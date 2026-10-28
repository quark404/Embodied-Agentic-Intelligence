import json
import tempfile
import unittest
from pathlib import Path

from embodied_agentic.advisory import (AdviceLimits, AdvisedPlanner, AdvisedPolicy,
                                        BoundedAdvisor, JsonEndpoint)
from embodied_agentic.evaluation import Condition, run_paired
from embodied_agentic.experiment import CONDITIONS, build_agent_pair, load_experiment_config
from embodied_agentic.harness import AgenticHarness, Decision, HarnessConfig, StepResult, Subtask
from embodied_agentic.harness_model import HarnessChatPlanner
from embodied_agentic.image_memory import SubtaskImageMemory
from embodied_agentic.synthetic import FixedPlanner, RulePolicy, SyntheticEnvironment, _png


TASK = {"task_id": "mixed-test", "episode_id": "mixed-test-0",
        "goal": "Open cabinet and deliver object"}
TOOLS = ("lookup", "unlock", "place")


class StubEndpoint:
    model = "gpt-6-astra"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def ask(self, system, prompt, images):
        self.prompts.append((system, prompt, images))
        return self.answers.pop(0), {"input_tokens": 12, "output_tokens": 3}


class MixedAgentTests(unittest.TestCase):
    def test_failed_qwen_call_is_logged_even_when_planning_stops(self):
        def timeout(_payload):
            raise TimeoutError("model endpoint unavailable")

        planner = HarnessChatPlanner("http://q/v1", "qwen", allowed_tools=TOOLS,
                                     transport=timeout)
        with tempfile.TemporaryDirectory() as root:
            result = AgenticHarness(environment=SyntheticEnvironment(), planner=planner,
                policy=RulePolicy(), memory=SubtaskImageMemory(Path(root) / "memory"),
                config=HarnessConfig(allowed_tools=TOOLS)).run(TASK)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model_calls"]["qwen:planner"]["calls"], 1)
        self.assertEqual(result["model_calls"]["qwen:planner"]["errors"], 1)

    def test_invalid_qwen_json_is_a_counted_model_protocol_error(self):
        planner = HarnessChatPlanner("http://q/v1", "qwen", allowed_tools=TOOLS,
            transport=lambda _payload: {"choices": [{"message": {"content": "not json"}}],
                                        "usage": {"prompt_tokens": 10}})
        with tempfile.TemporaryDirectory() as root:
            result = AgenticHarness(environment=SyntheticEnvironment(), planner=planner,
                policy=RulePolicy(), memory=SubtaskImageMemory(Path(root) / "memory"),
                config=HarnessConfig(allowed_tools=TOOLS)).run(TASK)
        self.assertIn("ModelOutputError", result["error"])
        self.assertEqual(result["model_calls"]["qwen:planner"]["input_tokens"], 10)
        self.assertEqual(result["model_calls"]["qwen:planner"]["errors"], 1)

    def test_advisor_is_bounded_and_cannot_return_actions(self):
        endpoint = StubEndpoint([
            {"subtasks": [{"id": "open", "instruction": "Open cabinet"}]},
            {"actions": [[0] * 14]},
        ])
        advisor = BoundedAdvisor(endpoint, limits=AdviceLimits(2, 2, 0))
        observation = {"images": {"front": _png((1, 2, 3))}}
        self.assertEqual(advisor.plan(TASK, observation, (), None),
                         [Subtask("open", "Open cabinet")])
        self.assertIsNone(advisor.plan(TASK, observation, (), None))
        self.assertIsNone(advisor.plan(TASK, observation, (), None))
        events = advisor.drain_events()
        self.assertEqual(len(events), 2)
        self.assertTrue(events[0]["ok"])
        self.assertFalse(events[1]["ok"])
        self.assertEqual(advisor.calls, {"plan": 2, "diagnosis": 0})

    def test_diagnosis_is_advice_and_qwen_actor_still_chooses_tools(self):
        class QwenLikePolicy(RulePolicy):
            def decide(self, context):
                if context.subtask.id == "open" and not any(
                    e.get("event") == "tool" for e in context.history):
                    return Decision("tool", "unlock", {"code": "wrong"})
                return super().decide(context)

        endpoint = StubEndpoint([{"diagnosis": "Wrong cabinet code",
                                 "next_step_hint": "Use the lookup receipt"}])
        advisor = BoundedAdvisor(endpoint, limits=AdviceLimits(1, 0, 1))
        with tempfile.TemporaryDirectory() as root:
            harness = AgenticHarness(environment=SyntheticEnvironment(),
                planner=FixedPlanner(), policy=AdvisedPolicy(QwenLikePolicy(), advisor),
                memory=SubtaskImageMemory(Path(root) / "memory"),
                config=HarnessConfig(allowed_tools=TOOLS),
                trace_path=Path(root) / "trace.jsonl")
            result = harness.run(TASK)
        self.assertTrue(result["success"])
        self.assertEqual(advisor.calls["diagnosis"], 1)
        self.assertEqual([e["tool"] for e in result["events"] if e["event"] == "tool"],
                         ["unlock", "lookup", "unlock", "place"])
        self.assertEqual(result["model_calls"]["gpt-6-astra:diagnosis"]["calls"], 1)
        self.assertTrue(all("actions" not in (e.get("advice") or {})
                            for e in result["events"] if e["event"] == "advisor_call"))

    def test_qwen_can_explicitly_consult_after_action(self):
        class ConsultPolicy(RulePolicy):
            def decide(self, context):
                if context.subtask.id == "open" and not any(
                    e.get("event") == "tool" for e in context.history):
                    return Decision("tool", "lookup")
                if not any(e.get("event") == "consult" for e in context.history):
                    return Decision("consult")
                return super().decide(context)

        endpoint = StubEndpoint([{"diagnosis": "Need the code", "next_step_hint": "Use 4172"}])
        advisor = BoundedAdvisor(endpoint, limits=AdviceLimits(1, 0, 1))
        with tempfile.TemporaryDirectory() as root:
            result = AgenticHarness(environment=SyntheticEnvironment(),
                planner=FixedPlanner(), policy=AdvisedPolicy(ConsultPolicy(), advisor),
                memory=SubtaskImageMemory(Path(root) / "memory"),
                config=HarnessConfig(allowed_tools=TOOLS)).run(TASK)
        self.assertTrue(result["success"])
        self.assertEqual(advisor.calls["diagnosis"], 1)
        self.assertEqual(endpoint.prompts[0][1]["trigger"], "actor_requested")
        self.assertEqual(next(e for e in result["events"] if e["event"] == "consult")
                         ["advice"]["diagnosis"], "Need the code")

    def test_provisional_advance_captures_new_subtask_without_claiming_success(self):
        class NoSubtaskVerifier(SyntheticEnvironment):
            def execute(self, tool, args):
                outcome = super().execute(tool, args)
                return StepResult(outcome.observation, outcome.receipt,
                                  task_done=outcome.task_done, episode_end=outcome.episode_end)

        class AdvancingPolicy(RulePolicy):
            def decide(self, context):
                if context.subtask.id == "open" and context.observation.get("phase") == 1:
                    return Decision("advance")
                return super().decide(context)

        with tempfile.TemporaryDirectory() as root:
            result = AgenticHarness(environment=NoSubtaskVerifier(), planner=FixedPlanner(),
                policy=AdvancingPolicy(), memory=SubtaskImageMemory(Path(root) / "memory"),
                config=HarnessConfig(allowed_tools=TOOLS)).run(TASK)
            records = SubtaskImageMemory(Path(root) / "memory").records
        self.assertTrue(result["success"])
        self.assertEqual([r.subtask_id for r in records], ["open", "deliver"])
        self.assertEqual(result["completed_subtasks"], [])
        self.assertEqual(next(e for e in result["events"] if e["event"] == "subtask_advance")
                         ["verification"], "policy_unverified")

    def test_advance_without_action_cannot_create_a_false_memory_checkpoint(self):
        class PrematurePolicy:
            def decide(self, context):
                return Decision("advance")

        with tempfile.TemporaryDirectory() as root:
            bank = SubtaskImageMemory(Path(root) / "memory")
            result = AgenticHarness(environment=SyntheticEnvironment(),
                planner=FixedPlanner(), policy=PrematurePolicy(), memory=bank,
                config=HarnessConfig(allowed_tools=TOOLS)).run(TASK)
            records = SubtaskImageMemory(Path(root) / "memory").records
        self.assertIn("Advance requires a successful action", result["error"])
        self.assertEqual({record.subtask_id for record in records}, {"open"})

    def test_responses_transport_and_condition_router(self):
        response = {"output": [{"content": [{"type": "output_text", "text": "{\"kind\":\"advance\"}"}]}],
                    "usage": {"input_tokens": 9, "output_tokens": 2}}
        captured = []
        endpoint = JsonEndpoint("http://localhost:1234/v1", "gpt-6-astra", mode="responses",
            transport=lambda payload: captured.append(payload) or response)
        result = endpoint.as_chat_transport()({"model": "gpt-6-astra", "messages": [
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": [{"type": "text", "text": "See image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]})
        self.assertEqual(json.loads(result["choices"][0]["message"]["content"])["kind"], "advance")
        self.assertEqual(captured[0]["input"][1]["content"][1]["type"], "input_image")

        config = {"qwen": {"base_url": "http://localhost:8000/v1", "model": "qwen"},
                  "astra": {"base_url": "http://localhost:9000/v1", "model": "gpt-6-astra",
                            "mode": "responses"}}
        for name in CONDITIONS:
            planner, actor = build_agent_pair(config, name, allowed_tools=("act",),
                tool_schemas={"act": {"type": "object"}})
            self.assertEqual(getattr(actor, "qwen_actor", actor).model,
                             "gpt-6-astra" if name == "astra_reference" else "qwen")
            self.assertIsInstance(planner, AdvisedPlanner) if "plan" in name or name.startswith("hybrid_both") else None
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps(config))
            self.assertEqual(load_experiment_config(path)["qwen"]["model"], "qwen")


if __name__ == "__main__":
    unittest.main()
