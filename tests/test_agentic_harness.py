import base64
import json
import tempfile
import time
import unittest
from pathlib import Path

from embodied_agentic.harness import AgenticHarness, Decision, HarnessConfig
from embodied_agentic.evaluation import (Condition, GoalOnlyPlanner,
                                         make_ablation_conditions, run_paired)
from embodied_agentic.harness_model import HarnessChatPlanner, HarnessChatPolicy
from embodied_agentic.image_memory import SubtaskImageMemory
from embodied_agentic.synthetic import FixedPlanner, RulePolicy, SyntheticEnvironment, _png, run_demo


TASK = {"task_id": "test-1", "episode_id": "test-1-seed0",
        "goal": "Open cabinet and deliver object"}
TOOLS = ("lookup", "unlock", "place")


class HarnessTests(unittest.TestCase):
    def harness(self, root, policy=None, *, config=None, environment=None):
        return AgenticHarness(environment=environment or SyntheticEnvironment(),
            planner=FixedPlanner(), policy=policy or RulePolicy(),
            memory=SubtaskImageMemory(Path(root) / "memory"),
            config=config or HarnessConfig(allowed_tools=TOOLS),
            trace_path=Path(root) / "trace.jsonl")

    def test_two_subtasks_capture_initial_images_and_recall_exact_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_demo(root)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["completed_subtasks"], ["open", "deliver"])
            self.assertEqual((result["tool_calls"], result["recall_calls"]), (3, 1))
            bank = SubtaskImageMemory(Path(root) / "memory")
            self.assertEqual(bank.recall(task_id="synthetic-1", episode_id="synthetic-1-seed0",
                                         subtask_id="open")[0].data,
                             _png((255, 0, 0)))
            self.assertEqual(bank.recall(task_id="synthetic-1", episode_id="synthetic-1-seed0",
                                         subtask_id="deliver")[0].data,
                             _png((0, 0, 255)))
            self.assertEqual(bank.recall(task_id="other", episode_id="synthetic-1-seed0",
                                         subtask_id="open"), [])
            self.assertEqual(bank.recall(task_id="synthetic-1", episode_id="other-seed",
                                         subtask_id="open"), [])
            trace = (Path(root) / "trace.jsonl").read_text()
            self.assertNotIn("data:image", trace)
            self.assertNotIn("iVBOR", trace)

    def test_snapshot_is_immutable_and_checksummed(self):
        with tempfile.TemporaryDirectory() as root:
            bank = SubtaskImageMemory(root)
            bank.capture(task_id="t", episode_id="e", subtask_id="s", instruction="start", tags=("pick",),
                         images={"front": _png((255, 0, 0))})
            bank.capture(task_id="t", episode_id="e", subtask_id="s", instruction="start", tags=("pick",),
                         images={"front": _png((0, 0, 255))})
            self.assertEqual(bank.recall(task_id="t", episode_id="e", tags=("pick",))[0].data,
                             _png((255, 0, 0)))
            path = Path(root) / bank.records[0].filename
            path.write_bytes(_png((0, 0, 255)))
            with self.assertRaises(ValueError):
                bank.recall(task_id="t", episode_id="e", subtask_id="s")

    def test_failed_action_can_be_corrected_using_receipt(self):
        class CorrectingPolicy(RulePolicy):
            def decide(self, context):
                if context.subtask.id == "open" and not any(e.get("event") == "tool" for e in context.history):
                    return Decision("tool", "unlock", {"code": "wrong"})
                return super().decide(context)
        with tempfile.TemporaryDirectory() as root:
            result = self.harness(root, policy=CorrectingPolicy()).run(TASK)
            self.assertTrue(result["success"])
            self.assertTrue(any(e.get("event") == "tool" and not e["receipt"]["ok"]
                                for e in result["events"]))

    def test_budget_and_invalid_tool_fail_closed(self):
        class BadPolicy:
            def decide(self, context):
                return Decision("tool", "unapproved")
        with tempfile.TemporaryDirectory() as root:
            limited = self.harness(root, config=HarnessConfig(max_tool_calls=1,
                allowed_tools=TOOLS)).run(TASK)
            self.assertEqual(limited["status"], "tool_budget_exhausted")
        with tempfile.TemporaryDirectory() as root:
            bad = self.harness(root, policy=BadPolicy()).run(TASK)
            self.assertEqual(bad["status"], "error")
            self.assertIn("allowlist", bad["error"])

    def test_missing_recall_and_replan_use_failure_receipt(self):
        class RecoveringPlanner(FixedPlanner):
            failures = []
            def plan(self, task, observation, completed, failure):
                self.failures.append(failure)
                return super().plan(task, observation, completed, failure)

        class RecoveringPolicy(RulePolicy):
            def decide(self, context):
                if not any(e.get("event") == "recall" for e in context.history):
                    return Decision("recall", recall_subtask_id="unknown")
                if not any(e.get("event") == "tool" for e in context.history):
                    return Decision("tool", "unlock", {"code": "wrong"})
                if not any(e.get("event") == "replan" for e in context.history):
                    return Decision("replan")
                return super().decide(context)

        with tempfile.TemporaryDirectory() as root:
            planner = RecoveringPlanner()
            harness = self.harness(root, policy=RecoveringPolicy())
            harness.planner = planner
            result = harness.run(TASK)
            self.assertTrue(result["success"])
            self.assertEqual(result["replans"], 1)
            self.assertEqual(planner.failures[1], {"ok": False})
            self.assertEqual(next(e for e in result["events"] if e["event"] == "recall")["count"], 0)

    def test_replan_discards_recalled_images(self):
        class RecallThenReplan(RulePolicy):
            def decide(self, context):
                if not any(e["event"] == "recall" for e in context.history):
                    return Decision("recall", recall_subtask_id="open")
                if not any(e["event"] == "replan" for e in context.history):
                    self.assert_recalled = bool(context.recalled_images)
                    return Decision("replan")
                if context.subtask.id == "open":
                    assert not context.recalled_images
                return super().decide(context)

        with tempfile.TemporaryDirectory() as root:
            policy = RecallThenReplan()
            result = self.harness(root, policy=policy).run(TASK)
            self.assertTrue(result["success"])
            self.assertTrue(policy.assert_recalled)

    def test_native_task_verdict_required(self):
        class IncompleteEnvironment(SyntheticEnvironment):
            def execute(self, tool, args):
                result = super().execute(tool, args)
                if result.task_done:
                    from embodied_agentic.harness import StepResult
                    return StepResult(result.observation, result.receipt, subtask_done=True,
                                      task_done=False)
                return result

        with tempfile.TemporaryDirectory() as root:
            result = self.harness(root, environment=IncompleteEnvironment()).run(TASK)
            self.assertEqual(result["status"], "plan_exhausted_without_task_success")
            self.assertFalse(result["success"])

        class EarlySuccessEnvironment(SyntheticEnvironment):
            def execute(self, tool, args):
                result = super().execute(tool, args)
                if tool == "unlock" and result.receipt["ok"]:
                    from embodied_agentic.harness import StepResult
                    return StepResult(result.observation, result.receipt,
                                      subtask_done=True, task_done=True)
                return result

        with tempfile.TemporaryDirectory() as root:
            result = self.harness(root, environment=EarlySuccessEnvironment()).run(TASK)
            self.assertTrue(result["success"])
            self.assertEqual(result["completed_subtasks"], ["open"])

        class TerminalFailureEnvironment(SyntheticEnvironment):
            def execute(self, tool, args):
                from embodied_agentic.harness import StepResult
                outcome = super().execute(tool, args)
                return StepResult(outcome.observation, outcome.receipt, episode_end=True)

        with tempfile.TemporaryDirectory() as root:
            result = self.harness(root, environment=TerminalFailureEnvironment()).run(TASK)
            self.assertEqual(result["status"], "environment_terminated_without_success")

    def test_episode_isolation_and_time_budget(self):
        with tempfile.TemporaryDirectory() as root:
            harness = self.harness(root)
            self.assertTrue(harness.run(TASK)["success"])
            repeated = self.harness(root).run(TASK)
            self.assertEqual(repeated["status"], "error")
            self.assertIn("fresh episode ID", repeated["error"])
            new_episode = dict(TASK, episode_id="test-1-seed1")
            self.assertTrue(self.harness(root).run(new_episode)["success"])

        class SlowPolicy:
            def decide(self, context):
                time.sleep(0.005)
                return Decision("tool", "lookup")
        with tempfile.TemporaryDirectory() as root:
            result = self.harness(root, policy=SlowPolicy(),
                config=HarnessConfig(max_seconds=0.001, allowed_tools=TOOLS)).run(TASK)
            self.assertEqual(result["status"], "time_budget_exhausted")
            self.assertEqual(result["tool_calls"], 0)

    def test_vlm_payload_contains_current_and_labeled_historical_images(self):
        payloads = []
        def transport(payload):
            payloads.append(payload)
            return {"choices": [{"message": {"content": json.dumps({
                "kind": "tool", "tool": "place", "args": {}})}}]}
        with tempfile.TemporaryDirectory() as root:
            bank = SubtaskImageMemory(root)
            bank.capture(task_id="test-1", episode_id="test-1-seed0", subtask_id="open",
                         instruction="open", tags=(),
                         images={"front": _png((255, 0, 0))})
            from embodied_agentic.harness import AgentContext, Subtask
            policy = HarnessChatPolicy("http://localhost:8000/v1", "qwen", allowed_tools=TOOLS,
                                       transport=transport)
            context = AgentContext(TASK, Subtask("deliver", "deliver"),
                {"phase": 1, "images": {"front": _png((0, 0, 255))}}, (),
                tuple(bank.recall(task_id="test-1", episode_id="test-1-seed0",
                                  subtask_id="open")), ("open",))
            self.assertEqual(policy.decide(context).tool, "place")
            parts = payloads[0]["messages"][1]["content"]
            images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
            self.assertEqual(len(images), 2)
            self.assertEqual(base64.b64decode(images[0].split(",", 1)[1]), _png((0, 0, 255)))
            self.assertEqual(base64.b64decode(images[1].split(",", 1)[1]), _png((255, 0, 0)))
            self.assertIn("Historical subtask-start image", str(parts))

    def test_vlm_planner_parses_subtasks_and_current_image(self):
        payloads = []
        def transport(payload):
            payloads.append(payload)
            return {"choices": [{"message": {"content": json.dumps([
                {"id": "open", "instruction": "Open cabinet", "tags": ["cabinet"]},
                {"id": "deliver", "instruction": "Deliver object"}])}}]}
        planner = HarnessChatPlanner("http://localhost:8000/v1", "qwen", allowed_tools=TOOLS,
                                      transport=transport)
        subtasks = planner.plan(TASK, {"images": {"front": _png((255, 0, 0))}}, (), None)
        self.assertEqual([s.id for s in subtasks], ["open", "deliver"])
        self.assertEqual(subtasks[0].tags, ("cabinet",))
        parts = payloads[0]["messages"][1]["content"]
        self.assertEqual(sum(p["type"] == "image_url" for p in parts), 1)

    def test_stubbed_multimodal_endpoint_runs_full_loop(self):
        def reply(value):
            return {"choices": [{"message": {"content": json.dumps(value)}}]}

        decisions = iter([
            {"kind": "tool", "tool": "lookup", "args": {}},
            {"kind": "tool", "tool": "unlock", "args": {"code": "4172"}},
            {"kind": "recall", "recall_subtask_id": "open"},
            {"kind": "tool", "tool": "place", "args": {}},
        ])
        planner = HarnessChatPlanner("http://localhost:8000/v1", "qwen", allowed_tools=TOOLS,
            transport=lambda payload: reply([
                {"id": "open", "instruction": "Open cabinet"},
                {"id": "deliver", "instruction": "Deliver object"}]))
        policy = HarnessChatPolicy("http://localhost:8000/v1", "qwen", allowed_tools=TOOLS,
            transport=lambda payload: reply(next(decisions)))
        with tempfile.TemporaryDirectory() as root:
            harness = self.harness(root, policy=policy)
            harness.planner = planner
            result = harness.run(TASK)
            self.assertTrue(result["success"])
            self.assertEqual(result["recall_calls"], 1)

    def test_paired_runner_keeps_task_seed_and_budgets_aligned(self):
        class BlindPolicy:
            def decide(self, context):
                return Decision("tool", "unlock", {"code": ""})

        conditions = [
            Condition("direct", FixedPlanner, BlindPolicy, ("unlock", "place")),
            Condition("tool_memory", FixedPlanner, RulePolicy, TOOLS),
        ]
        with tempfile.TemporaryDirectory() as root:
            rows = run_paired(tasks=[dict(TASK, seed=0), dict(TASK, seed=1)],
                conditions=conditions, environment_factory=SyntheticEnvironment,
                base_config=HarnessConfig(max_decisions=6, allowed_tools=TOOLS),
                output_dir=root, run_id="synthetic-v1")
            self.assertEqual(len(rows), 4)
            self.assertEqual({(r["seed"], r["condition"]) for r in rows},
                             {(0, "direct"), (0, "tool_memory"),
                              (1, "direct"), (1, "tool_memory")})
            self.assertEqual({r["budget"]["max_decisions"] for r in rows}, {6})
            self.assertEqual(len({r["episode_id"] for r in rows}), 4)
            self.assertEqual([r["success"] for r in rows if r["condition"] == "tool_memory"],
                             [True, True])
            self.assertTrue((Path(root) / "results.json").exists())

    def test_paired_runner_can_select_each_conditions_runtime(self):
        calls = []
        def make_env(label):
            calls.append(label)
            return SyntheticEnvironment()
        conditions = [
            Condition("pyrualean_arm", FixedPlanner, RulePolicy, TOOLS,
                      environment_factory=lambda: make_env("pyrualean_arm")),
            Condition("roboicl_arm", FixedPlanner, RulePolicy, TOOLS,
                      environment_factory=lambda: make_env("roboicl_arm")),
        ]
        with tempfile.TemporaryDirectory() as root:
            rows = run_paired(tasks=[dict(TASK, seed=0)], conditions=conditions,
                environment_factory=lambda: make_env("fallback"),
                base_config=HarnessConfig(allowed_tools=TOOLS), output_dir=root,
                run_id="two-runtimes")
        self.assertEqual(calls, ["pyrualean_arm", "roboicl_arm"])
        self.assertEqual(len(rows), 2)

    def test_ablation_disables_memory_and_replanning_in_code(self):
        class RecallPolicy:
            def decide(self, context):
                return Decision("recall", recall_subtask_id="whole_task")

        with tempfile.TemporaryDirectory() as root:
            rows = run_paired(tasks=[dict(TASK, seed=0)], conditions=[
                Condition("direct", GoalOnlyPlanner, RecallPolicy, TOOLS,
                          enable_image_memory=False, enable_replanning=False)],
                environment_factory=SyntheticEnvironment,
                base_config=HarnessConfig(allowed_tools=TOOLS), output_dir=root,
                run_id="no-memory")
            self.assertFalse(rows[0]["success"])
            self.assertIn("Image recall is disabled", rows[0]["error"])
            self.assertFalse(rows[0]["enable_image_memory"])
            self.assertFalse(rows[0]["enable_replanning"])
            self.assertFalse((Path(rows[0]["artifact_dir"]) / "memory" / "manifest.jsonl").exists())

        policy = HarnessChatPolicy("http://localhost:8000/v1", "qwen", allowed_tools=TOOLS,
            allowed_decisions=("tool",),
            transport=lambda payload: {"choices": [{"message": {"content":
                json.dumps({"kind": "recall", "recall_subtask_id": "whole_task"})}}]})
        with tempfile.TemporaryDirectory() as root:
            harness = self.harness(root, policy=policy,
                config=HarnessConfig(allowed_tools=TOOLS, enable_image_memory=False,
                                     enable_replanning=False))
            harness.planner = GoalOnlyPlanner()
            result = harness.run(TASK)
            self.assertIn("disabled decision kind", result["error"])

    def test_standard_ablation_matrix_shares_model_planner_and_tools(self):
        conditions = make_ablation_conditions(
            direct_policy_factory=RulePolicy, tool_policy_factory=RulePolicy,
            planned_policy_factory=RulePolicy, memory_policy_factory=RulePolicy,
            planner_factory=FixedPlanner, direct_tools=("unlock", "place"),
            tool_tools=TOOLS)
        self.assertEqual([c.name for c in conditions],
                         ["A_direct", "B_tools", "C_plan_tools", "D_plan_tools_memory"])
        self.assertEqual([c.enable_image_memory for c in conditions],
                         [False, False, False, True])
        self.assertEqual([c.enable_replanning for c in conditions],
                         [False, False, True, True])
        self.assertEqual({c.allowed_tools for c in conditions[1:]}, {TOOLS})
        self.assertIs(conditions[2].planner_factory, conditions[3].planner_factory)


if __name__ == "__main__":
    unittest.main()
