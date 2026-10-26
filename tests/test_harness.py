import unittest

from embodied_agentic.core import CabinetWorld, Task, run_episode
from embodied_agentic.memory import MemoryBank
from embodied_agentic.model import ChatActionPolicy
from embodied_agentic.policies import DirectPolicy, PlannerMemoryPolicy, ToolPolicy


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.task = Task("test", "test", "blue cabinet", "cup", "table", "1234")

    def test_equal_budget_and_memory_gain(self):
        bank = MemoryBank()
        bank.add(task_id="train", split="train", cabinet="blue cabinet", code="1234", success=True)
        direct = run_episode(CabinetWorld(), DirectPolicy(), self.task, max_steps=4)
        tool = run_episode(CabinetWorld(), ToolPolicy(), self.task, max_steps=4)
        memory = run_episode(CabinetWorld(), PlannerMemoryPolicy(bank), self.task, max_steps=4)
        self.assertFalse(direct["success"])
        self.assertTrue(tool["success"])
        self.assertTrue(memory["success"])
        self.assertEqual((tool["steps"], memory["steps"]), (4, 3))

    def test_bad_memory_falls_back_to_tool(self):
        bank = MemoryBank()
        bank.add(task_id="train", split="train", cabinet="blue cabinet", code="stale", success=True)
        result = run_episode(CabinetWorld(), PlannerMemoryPolicy(bank), self.task, max_steps=5)
        self.assertTrue(result["success"])
        self.assertEqual(result["failed_actions"], 1)
        self.assertEqual(result["tool_calls"], 1)

    def test_no_test_leakage(self):
        bank = MemoryBank()
        with self.assertRaises(ValueError):
            bank.add(task_id="test", split="test", cabinet="blue cabinet", code="1234", success=True)
        self.assertEqual(bank.retrieve("blue cabinet"), [])

    def test_model_adapter_validates_actions(self):
        policy = ChatActionPolicy("http://localhost:8000/v1", "qwen-test",
                                  transport=lambda payload: {"choices": [{"message": {
                                      "content": '{"type":"lookup_code"}'}}]})
        self.assertEqual(policy.act({"cabinet": "blue cabinet"}, []), {"type": "lookup_code"})
        policy.transport = lambda payload: {"choices": [{"message": {"content": '{"type":"fly"}'}}]}
        with self.assertRaises(ValueError):
            policy.act({}, [])


if __name__ == "__main__":
    unittest.main()
