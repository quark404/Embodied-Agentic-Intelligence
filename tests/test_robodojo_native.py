import json
import math
import sys
import tempfile
from types import ModuleType
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from embodied_agentic.evaluation import GoalOnlyPlanner
from embodied_agentic.harness import AgenticHarness, Decision, HarnessConfig
from embodied_agentic.image_memory import SubtaskImageMemory
from embodied_agentic.robodojo_native import (RoboDojoNativeAdapter,
    convert_ee_delta_actions, install_native_ik_guard, run_official_episode)


def action(amount=0.04):
    return {"left_arm_joint_state": [amount] + [0.0] * 5,
            "right_arm_joint_state": [0.0] * 6,
            "left_ee_joint_state": [0.0], "right_ee_joint_state": [0.0]}


def ee_delta(amount=0.02):
    return {"mode": "ee_delta", "left": [amount, 0, 0, 0, 0, 0],
            "right": [0, 0, 0, 0, 0, 0],
            "left_gripper": 0.5, "right_gripper": 0.5}


class FakeIKError(ValueError):
    event = {"type": "action_rejected_ik_continuity", "rejected_action_executed": False}


class FakeNativeEnv:
    def __init__(self, save_dir, *, can_succeed=True):
        self.num_envs = 1
        self.eval_batch = False
        self.task_name = "general_pickup"
        self.env_seeds = [2]
        self.eval_seed = 0
        self.run_id = "native-test"
        self.success_nums = 0
        self.fail_nums = 0
        self.save_dir = str(save_dir)
        self.success = [True]  # RoboDojo initializes this true before grading.
        self.terminal = False
        self.take_action_cnt = [0]
        self.step_lim = 5
        self.can_succeed = can_succeed
        self.state = {"left_arm_joint_state": [0.0] * 6,
                      "right_arm_joint_state": [0.0] * 6,
                      "left_ee_joint_state": [0.0],
                      "right_ee_joint_state": [0.0],
                      "left_ee_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      "right_ee_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
        self.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        self.reject_ik = False

    def get_obs(self):
        pixels = np.zeros((3, 4, 3), dtype=np.uint8)
        pixels[..., 0] = 125
        return {"instruction": "Pick up the requested object", "state": self.state,
                "vision": {name: {"color": pixels}
                           for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                "reward": 100, "success": self.success[0]}

    def validate_action_dict(self, command):
        expected = (set(action()) if "left_arm_joint_state" in command else
                    {"left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"})
        if set(command) != expected:
            raise ValueError("Native action keys differ")
        for value in command.values():
            assert isinstance(value, np.ndarray)

    def take_action(self, command):
        self.validate_action_dict(command)
        if self.reject_ik:
            raise FakeIKError("pre-queue IK rejection")
        self.take_action_cnt[0] += 1
        self.state.update({key: value.tolist() for key, value in command.items()})

    def is_episode_end(self):
        if self.can_succeed and (self.state["left_arm_joint_state"][0] > 0
                                 or self.state["left_ee_pose"][0] > 0):
            self.success[0] = True
            self.terminal = True
        if not self.success[0] or self.take_action_cnt[0] >= self.step_lim:
            self.success[0] = False
            self.terminal = True
        return self.terminal


class OneActionPolicy:
    def decide(self, context):
        return Decision("tool", "act", {"actions": [action()]})


class HoldPolicy:
    def decide(self, context):
        return Decision("tool", "act", {"actions": [action(0.0)]})


class EEPolicy:
    def decide(self, context):
        return Decision("tool", "act", {"actions": [ee_delta()]})


def build_harness(adapter, task, root, policy=OneActionPolicy):
    return AgenticHarness(environment=adapter, planner=GoalOnlyPlanner(),
        policy=policy(), memory=SubtaskImageMemory(root / "memory"),
        config=HarnessConfig(allowed_tools=("act",), max_decisions=1,
                             enable_image_memory=True, enable_replanning=False),
        trace_path=root / "trace.jsonl")


class NativeTests(unittest.TestCase):
    def test_native_adapter_preserves_rgb_state_and_verdict(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root)
            adapter = RoboDojoNativeAdapter(native, max_joint_delta=0.05)
            observation = adapter.reset({"task_id": "general_pickup"})
            self.assertEqual(set(observation["images"]),
                             {"cam_head", "cam_left_wrist", "cam_right_wrist"})
            self.assertTrue(all(data.startswith(b"\x89PNG\r\n\x1a\n")
                                for data in observation["images"].values()))
            self.assertNotIn("reward", observation)
            self.assertNotIn("success", observation)
            self.assertEqual(set(adapter.tool_schemas()), {"act"})
            result = adapter.execute("act", {"actions": [action()]})
            self.assertTrue(result.task_done)
            self.assertTrue(result.episode_end)
            self.assertEqual(result.receipt["executed_steps"], 1)

    def test_bad_native_action_is_rejected_without_sim_step(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root)
            adapter = RoboDojoNativeAdapter(native, max_joint_delta=0.05)
            adapter.reset({"task_id": "general_pickup"})
            result = adapter.execute("act", {"actions": [action(0.5)]})
            self.assertFalse(result.receipt["ok"])
            self.assertEqual(result.receipt["executed_steps"], 0)
            self.assertEqual(native.take_action_cnt, [0])
            result = adapter.execute("act", {"actions": [{**action(), "left_arm_joint_state": [float("nan")] + [0.0] * 5}]})
            self.assertFalse(result.receipt["ok"])
            self.assertEqual(native.take_action_cnt, [0])

    def test_invalid_act_shape_returns_zero_step_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root)
            adapter = RoboDojoNativeAdapter(native, max_chunk=2)
            observation = adapter.reset({"task_id": "general_pickup"})
            for args in ({}, {"actions": None}, {"actions": []},
                         {"actions": [action()] * 3},
                         {"actions": [action()], "extra": 1}):
                with self.subTest(args=args):
                    result = adapter.execute("act", args)
                    self.assertFalse(result.receipt["ok"])
                    self.assertEqual(result.receipt["executed_steps"], 0)
                    self.assertIs(result.observation, observation)
                    self.assertEqual(native.take_action_cnt, [0])
            with self.assertRaises(ValueError):
                adapter.execute("not_act", {"actions": [action()]})
            result = adapter.execute("act", {"actions": [{**action(), "left_ee_joint_state": [1.5]}]})
            self.assertFalse(result.receipt["ok"])
            self.assertEqual(native.take_action_cnt, [0])

    def test_official_entry_records_native_success(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root)
            seen = {}
            def factory(adapter, task, output_dir):
                seen.update(task)
                return build_harness(adapter, task, output_dir)
            with patch.dict("os.environ", {"EMBODIED_ROBODOJO_CONTROL_MODE": "joint"}):
                result = run_official_episode(native, factory=factory)
            self.assertTrue(result["native_success"])
            self.assertEqual(seen["layout_id"], 2)
            self.assertEqual(seen["goal"], "Pick up the requested object")
            self.assertEqual(native.take_action_cnt, [1])
            saved = list(Path(root).glob("embodied_agentic/*/summary.json"))
            self.assertEqual(len(saved), 1)
            self.assertTrue(json.loads(saved[0].read_text())["native_success"])

    def test_early_harness_stop_forces_native_final_check(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root, can_succeed=False)
            with patch.dict("os.environ", {"EMBODIED_ROBODOJO_CONTROL_MODE": "joint"}):
                result = run_official_episode(native,
                    factory=lambda adapter, task, output_dir: build_harness(
                        adapter, task, output_dir, policy=HoldPolicy))
            self.assertEqual(result["status"], "decision_budget_exhausted")
            self.assertTrue(result["native_terminal"])
            self.assertFalse(result["native_success"])
            self.assertFalse(native.success[0])

    def test_roboicl_ee_delta_conversion_and_bounds(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root)
            native.ik_continuity_guard = object()
            adapter = RoboDojoNativeAdapter(native, action_type="ee_delta",
                ik_error_type=FakeIKError, max_translation_m=0.03,
                max_rotation_rad=0.15)
            observation = adapter.reset({"task_id": "general_pickup"})
            schema = adapter.tool_schemas()["act"]["properties"]["actions"]["items"]
            self.assertEqual(schema["properties"]["mode"]["enum"], ["ee_delta"])
            self.assertIn("World-frame", schema["properties"]["left"]["description"])
            self.assertIn("0.03 m", schema["properties"]["left"]["description"])
            self.assertIn("0 fully closed", schema["properties"]["left_gripper"]["description"])
            converted = convert_ee_delta_actions([ee_delta(), ee_delta()],
                observation["state"], max_chunk=15, max_translation_m=0.03,
                max_rotation_rad=0.15)
            self.assertAlmostEqual(float(converted[1]["left_ee_pose"][0]), 0.04)
            self.assertEqual(set(converted[0]),
                             {"left_ee_pose", "right_ee_pose",
                              "left_ee_joint_state", "right_ee_joint_state"})
            result = adapter.execute("act", {"actions": [ee_delta(0.04)]})
            self.assertFalse(result.receipt["ok"])
            self.assertEqual(native.take_action_cnt, [0])
            native.reject_ik = True
            result = adapter.execute("act", {"actions": [ee_delta()]})
            self.assertFalse(result.receipt["ok"])
            self.assertEqual(result.receipt["rejection"]["type"], "action_rejected_ik_continuity")
            self.assertEqual(native.take_action_cnt, [0])
            native.reject_ik = False
            result = adapter.execute("act", {"actions": [ee_delta()]})
            self.assertTrue(result.task_done)
            self.assertEqual(native.take_action_cnt, [1])

    def test_ee_delta_rotation_and_whole_chunk_prevalidation(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root, can_succeed=False)
            native.ik_continuity_guard = object()
            adapter = RoboDojoNativeAdapter(native, action_type="ee_delta",
                ik_error_type=FakeIKError)
            observation = adapter.reset({"task_id": "general_pickup"})
            rotate = ee_delta(0)
            rotate["left"][5] = 0.1
            converted = convert_ee_delta_actions([rotate], observation["state"],
                max_chunk=15, max_translation_m=0.03, max_rotation_rad=0.15)
            self.assertAlmostEqual(float(converted[0]["left_ee_pose"][3]), math.cos(0.05), places=6)
            self.assertAlmostEqual(float(converted[0]["left_ee_pose"][6]), math.sin(0.05), places=6)
            result = adapter.execute("act", {"actions": [ee_delta(), ee_delta(0.04)]})
            self.assertFalse(result.receipt["ok"])
            self.assertEqual(result.receipt["executed_steps"], 0)
            self.assertEqual(native.take_action_cnt, [0])
            invalid = ee_delta(0)
            invalid["right_gripper"] = float("nan")
            result = adapter.execute("act", {"actions": [invalid]})
            self.assertFalse(result.receipt["ok"])
            self.assertEqual(native.take_action_cnt, [0])

    def test_official_entry_ee_delta_uses_installed_guard(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root)
            def install(env, limit):
                env.ik_continuity_guard = object()
                self.assertEqual(limit, 0.15)
                return FakeIKError
            with patch("embodied_agentic.robodojo_native.install_native_ik_guard", install), \
                 patch.dict("os.environ", {"EMBODIED_ROBODOJO_CONTROL_MODE": "ee_delta",
                                        "EMBODIED_CONDITION": "hybrid_plan"}):
                result = run_official_episode(native,
                    factory=lambda adapter, task, output_dir: build_harness(
                        adapter, task, output_dir, policy=EEPolicy))
            self.assertTrue(result["native_success"])
            self.assertEqual(result["condition"], "hybrid_plan")
            self.assertEqual((result["seed"], result["layout_id"],
                              result["native_result_index"]), (0, 2, 0))

    def test_ee_guard_fails_closed_if_dispatch_cannot_be_inspected(self):
        with tempfile.TemporaryDirectory() as root:
            native = FakeNativeEnv(root)
            guard_module = ModuleType("src.eval_client.ik_continuity")
            guard_module.IKContinuityError = FakeIKError
            guard_module.IKContinuityGuard = object
            modules = {"src": ModuleType("src"),
                       "src.eval_client": ModuleType("src.eval_client"),
                       "src.eval_client.ik_continuity": guard_module}
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, "Cannot verify RoboDojo"):
                    install_native_ik_guard(native, 0.15)


if __name__ == "__main__":
    unittest.main()
