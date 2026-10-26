import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from embodied_agentic.harness import AgenticHarness, Decision, HarnessConfig, Subtask
from embodied_agentic.image_memory import SubtaskImageMemory
from embodied_agentic.synthetic import _png
from embodied_agentic.upstream_adapters import PyRUALeanAdapter, RoboICLAdapter


class _OneStepPlanner:
    def plan(self, task, observation, completed, failure):
        return [Subtask("complete", "Complete the native task")]


class _OneStepPolicy:
    def __init__(self, tool, args):
        self.tool = tool
        self.args = args

    def decide(self, context):
        return Decision("tool", self.tool, self.args)


class AdapterTests(unittest.TestCase):
    def test_pyrualean_arm_toolkit_bridge_preserves_native_verdict(self):
        class FakeRobot:
            task = "move the object"
            def __init__(self):
                self.done = False
            def _summary_dict(self):
                return {"step": int(self.done), "truncated": False}
            def artifact(self, name, step=-1):
                assert name == "agentview_high.png" and step == -1
                return _png((0, 0, 255) if self.done else (255, 0, 0))

        class FakeToolkit:
            arm = "cells"
            def __init__(self):
                self.robo = FakeRobot()
            def get_tools_spec(self):
                return [{"name": "python", "input_schema": {"type": "object"}},
                        {"name": "finish", "input_schema": {"type": "object"}}]
            def execute_tool(self, name, args):
                self.robo.done = args["code"] == "robo.move()"
                return SimpleNamespace(result={"cell": 1, "stdout": "",
                                               "episode_done": self.robo.done})

        toolkit = FakeToolkit()
        adapter = PyRUALeanAdapter(toolkit,
            image_artifacts={"agentview": "agentview_high.png"})
        self.assertEqual(list(adapter.tool_schemas()), ["python"])
        self.assertEqual(adapter.reset({"task_id": "t"})["images"]["agentview"], _png((255, 0, 0)))
        with self.assertRaises(ValueError):
            adapter.execute("finish", {"status": "success"})
        with tempfile.TemporaryDirectory() as root:
            result = AgenticHarness(environment=adapter, planner=_OneStepPlanner(),
                policy=_OneStepPolicy("python", {"code": "robo.move()"}),
                memory=SubtaskImageMemory(Path(root) / "memory"),
                config=HarnessConfig(allowed_tools=("python",))).run(
                    {"task_id": "t", "episode_id": "t-0", "goal": "move"})
            self.assertTrue(result["success"])
            self.assertEqual(result["tool_calls"], 1)

    def test_roboicl_bridge_validates_actions_and_reads_fresh_cameras(self):
        import numpy as np

        class FakeEnv:
            def __init__(self):
                self.steps = 0
                self.success = [False]
            def get_obs(self):
                color = np.full((2, 2, 3), 20 + self.steps, dtype=np.uint8)
                return {"instruction": "move", "state": {"left_arm_joint_state": np.zeros(6)},
                        "vision": {name: {"color": color} for name in RoboICLAdapter.CAMERAS}}
            def take_action(self, action):
                assert action == {"packed": True}
                self.steps += 1
                self.success[0] = self.steps == 1
            def is_episode_end(self):
                return self.steps >= 1

        env = FakeEnv()
        def converter(rows, state):
            assert len(rows) == 1 and len(rows[0]) == 14
            assert "left_arm_joint_state" in state
            return [{"packed": True}]

        adapter = RoboICLAdapter(env, action_converter=converter,
            action_executor=env.take_action, observation_reader=env.get_obs,
            success_reader=lambda native: native.success[0])
        self.assertEqual(adapter.tool_schemas()["act"]["properties"]["actions"]["items"]["minItems"], 14)
        initial = adapter.reset({"task_id": "t"})
        self.assertEqual(set(initial["images"]), set(RoboICLAdapter.CAMERAS))
        self.assertTrue(all(image.startswith(b"\xff\xd8\xff") for image in initial["images"].values()))
        with tempfile.TemporaryDirectory() as root:
            result = AgenticHarness(environment=adapter, planner=_OneStepPlanner(),
                policy=_OneStepPolicy("act", {"actions": [[0] * 14]}),
                memory=SubtaskImageMemory(Path(root) / "memory"),
                config=HarnessConfig(allowed_tools=("act",))).run(
                    {"task_id": "t", "episode_id": "t-1", "goal": "move"})
            self.assertTrue(result["success"])
            self.assertEqual(env.steps, 1)

    def test_roboicl_rejected_action_reports_zero_execution(self):
        import numpy as np

        class FakeEnv:
            def get_obs(self):
                return {"state": {}, "vision": {camera: {"color": np.ones((1, 1, 3), dtype=np.uint8)}
                                                 for camera in RoboICLAdapter.CAMERAS}}
            def is_episode_end(self):
                return False

        adapter = RoboICLAdapter(FakeEnv(), action_converter=lambda rows, state: [{"packed": True}],
            action_executor=lambda action: {"ok": False, "reason": "IK rejected"},
            observation_reader=FakeEnv().get_obs)
        adapter.reset({"task_id": "t"})
        outcome = adapter.execute("act", {"actions": [[0] * 14]})
        self.assertFalse(outcome.receipt["ok"])
        self.assertEqual(outcome.receipt["executed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
