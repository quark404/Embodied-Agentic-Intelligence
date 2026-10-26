"""Small deterministic fixture for exercising the full harness without robotics."""

from __future__ import annotations

import json
import struct
import tempfile
import zlib
from pathlib import Path

from .harness import AgentContext, AgenticHarness, Decision, HarnessConfig, StepResult, Subtask
from .image_memory import SubtaskImageMemory


def _png(rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, *rgb]))) + chunk(b"IEND", b""))


class SyntheticEnvironment:
    def __init__(self):
        self.phase = 0
        self.code = "4172"

    def _observe(self) -> dict:
        return {"phase": self.phase, "images": {"front": _png((255, 0, 0) if self.phase == 0
                                                            else (0, 0, 255))}}

    def reset(self, task: dict) -> dict:
        self.phase = 0
        return self._observe()

    def execute(self, tool: str, args: dict) -> StepResult:
        if tool == "lookup" and self.phase == 0:
            return StepResult(self._observe(), {"ok": True, "code": self.code})
        if tool == "unlock" and self.phase == 0:
            ok = args.get("code") == self.code
            if ok:
                self.phase = 1
            return StepResult(self._observe(), {"ok": ok}, subtask_done=ok)
        if tool == "place" and self.phase == 1:
            self.phase = 2
            return StepResult(self._observe(), {"ok": True}, subtask_done=True, task_done=True)
        return StepResult(self._observe(), {"ok": False, "reason": "invalid action for phase"})


class FixedPlanner:
    def plan(self, task: dict, observation: dict, completed: tuple[str, ...],
             failure: dict | None) -> list[Subtask]:
        all_subtasks = [Subtask("open", "Find code and unlock cabinet", ("cabinet",)),
                        Subtask("deliver", "Move object to destination", ("delivery",))]
        return [item for item in all_subtasks if item.id not in completed]


class RulePolicy:
    """Demonstrates control flow only; it is not an ML baseline."""

    def decide(self, context: AgentContext) -> Decision:
        if context.subtask.id == "open":
            receipt = next((event["receipt"] for event in reversed(context.history)
                            if event.get("event") == "tool" and event["tool"] == "lookup"
                            and event["receipt"]["ok"]), None)
            return Decision("tool", "unlock", {"code": receipt["code"]}) if receipt else Decision("tool", "lookup")
        if not any(event.get("event") == "recall" and event.get("current_subtask_id") == "deliver"
                   for event in context.history):
            return Decision("recall", recall_subtask_id="open")
        return Decision("tool", "place")


def run_demo(root: str | Path) -> dict:
    root = Path(root)
    harness = AgenticHarness(environment=SyntheticEnvironment(), planner=FixedPlanner(),
        policy=RulePolicy(), memory=SubtaskImageMemory(root / "memory"),
        config=HarnessConfig(allowed_tools=("lookup", "unlock", "place")),
        trace_path=root / "trace.jsonl")
    return harness.run({"task_id": "synthetic-1", "episode_id": "synthetic-1-seed0",
                        "goal": "Open cabinet and deliver object"})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    output = args.output_dir or tempfile.mkdtemp(prefix="embodied-agentic-")
    result = run_demo(output)
    print(json.dumps({"output_dir": output, "status": result["status"],
                      "completed_subtasks": result["completed_subtasks"],
                      "tool_calls": result["tool_calls"], "recall_calls": result["recall_calls"]},
                     ensure_ascii=False, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
