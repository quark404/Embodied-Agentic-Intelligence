from __future__ import annotations

from .memory import MemoryBank


class DirectPolicy:
    """No lookup tool; deterministic stand-in for a direct action interface."""

    def act(self, observation: dict, history: list[dict]) -> dict:
        if not observation["unlocked"]:
            return {"type": "unlock", "code": ""}
        if not observation["held"]:
            return {"type": "grasp", "object": observation["object"]}
        return {"type": "place", "destination": observation["destination"]}


class ToolPolicy(DirectPolicy):
    def act(self, observation: dict, history: list[dict]) -> dict:
        if not observation["unlocked"]:
            for item in reversed(history):
                receipt = item["receipt"]
                if receipt["action"] == "lookup_code" and receipt["ok"]:
                    return {"type": "unlock", "code": receipt["code"]}
            return {"type": "lookup_code"}
        return super().act(observation, history)


class PlannerMemoryPolicy(ToolPolicy):
    def __init__(self, bank: MemoryBank):
        self.bank = bank

    def act(self, observation: dict, history: list[dict]) -> dict:
        if not observation["unlocked"]:
            # Failed recalled facts trigger a fresh lookup; receipts drive correction.
            attempted = {item["action"].get("code") for item in history
                         if item["action"].get("type") == "unlock" and not item["receipt"]["ok"]}
            matches = self.bank.retrieve(observation["cabinet"])
            if matches and matches[0]["code"] not in attempted:
                return {"type": "unlock", "code": matches[0]["code"]}
        return super().act(observation, history)
