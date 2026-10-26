from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    cabinet: str
    object_name: str
    destination: str
    code: str


class Environment(Protocol):
    def reset(self, task: Task) -> dict: ...
    def step(self, action: dict) -> tuple[dict, bool, dict]: ...


class Policy(Protocol):
    def act(self, observation: dict, history: list[dict]) -> dict: ...


class CabinetWorld:
    """Synthetic capability probe; no claim of physical realism."""

    def reset(self, task: Task) -> dict:
        self.task = task
        self.unlocked = False
        self.held = False
        self.done = False
        return self.observe()

    def observe(self) -> dict:
        return {
            "instruction": f"Move {self.task.object_name} from {self.task.cabinet} to {self.task.destination}",
            "cabinet": self.task.cabinet,
            "object": self.task.object_name,
            "destination": self.task.destination,
            "unlocked": self.unlocked,
            "held": self.held,
        }

    def step(self, action: dict) -> tuple[dict, bool, dict]:
        kind = action.get("type")
        receipt = {"action": kind, "ok": False}
        if kind == "lookup_code":
            receipt.update(ok=True, code=self.task.code)
        elif kind == "unlock":
            self.unlocked = action.get("code") == self.task.code
            receipt["ok"] = self.unlocked
        elif kind == "grasp":
            self.held = self.unlocked and action.get("object") == self.task.object_name
            receipt["ok"] = self.held
        elif kind == "place":
            self.done = self.held and action.get("destination") == self.task.destination
            receipt["ok"] = self.done
        else:
            receipt["error"] = "unknown action"
        return self.observe(), self.done, receipt


def run_episode(env: Environment, policy: Policy, task: Task, max_steps: int = 5) -> dict:
    observation = env.reset(task)
    history: list[dict] = []
    success = False
    for _ in range(max_steps):
        action = policy.act(observation, history)
        observation, success, receipt = env.step(action)
        history.append({"observation": observation, "action": action, "receipt": receipt})
        if success:
            break
    return {
        "task_id": task.task_id,
        "split": task.split,
        "success": success,
        "steps": len(history),
        "tool_calls": sum(item["action"].get("type") == "lookup_code" for item in history),
        "failed_actions": sum(not item["receipt"]["ok"] for item in history),
        "trace": history,
    }
