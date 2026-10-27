"""Bounded Planner–Tool–Memory agent loop with environment-owned verdicts."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .image_memory import RecalledImage, SubtaskImageMemory


@dataclass(frozen=True)
class Subtask:
    id: str
    instruction: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    kind: str  # tool | recall | replan
    tool: str = ""
    args: dict = field(default_factory=dict)
    recall_subtask_id: str = ""
    recall_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepResult:
    observation: dict
    receipt: dict
    subtask_done: bool = False
    task_done: bool = False
    episode_end: bool = False


@dataclass(frozen=True)
class HarnessConfig:
    max_decisions: int = 40
    max_tool_calls: int = 30
    max_recall_calls: int = 8
    max_replans: int = 3
    max_seconds: float = 600.0
    max_subtasks: int = 20
    recall_limit: int = 4
    allowed_tools: tuple[str, ...] = ()
    enable_image_memory: bool = True
    enable_replanning: bool = True

    def __post_init__(self):
        if (any(x < 1 for x in (self.max_decisions, self.max_tool_calls,
                                self.max_subtasks, self.recall_limit)) or self.max_seconds <= 0):
            raise ValueError("Decision, tool, time, subtask and recall limits must be positive")
        if self.max_recall_calls < 0 or self.max_replans < 0:
            raise ValueError("Recall and replan limits cannot be negative")


@dataclass(frozen=True)
class AgentContext:
    task: dict
    subtask: Subtask
    observation: dict
    history: tuple[dict, ...]
    recalled_images: tuple[RecalledImage, ...]
    completed_subtasks: tuple[str, ...]


class HarnessEnvironment(Protocol):
    def reset(self, task: dict) -> dict: ...
    def execute(self, tool: str, args: dict) -> StepResult: ...


class Planner(Protocol):
    def plan(self, task: dict, observation: dict, completed: tuple[str, ...],
             failure: dict | None) -> list[Subtask]: ...


class AgentPolicy(Protocol):
    def decide(self, context: AgentContext) -> Decision: ...


class AgenticHarness:
    def __init__(self, *, environment: HarnessEnvironment, planner: Planner,
                 policy: AgentPolicy, memory: SubtaskImageMemory,
                 config: HarnessConfig, trace_path: str | Path | None = None):
        self.environment = environment
        self.planner = planner
        self.policy = policy
        self.memory = memory
        self.config = config
        self.trace_path = Path(trace_path) if trace_path else None

    @staticmethod
    def _images(observation: dict) -> dict[str, bytes]:
        images = observation.get("images")
        if not isinstance(images, dict) or not images:
            raise ValueError("Observation must contain camera-name to PNG/JPEG bytes in images")
        return images

    def _plan(self, task: dict, observation: dict, completed: tuple[str, ...],
              failure: dict | None) -> list[Subtask]:
        plan = self.planner.plan(deepcopy(task), deepcopy(observation), completed, deepcopy(failure))
        if (not isinstance(plan, list) or not 0 < len(plan) <= self.config.max_subtasks or
                any(not isinstance(s, Subtask) or not s.id or not s.instruction for s in plan) or
                len({s.id for s in plan}) != len(plan) or
                any(s.id in completed for s in plan)):
            raise ValueError("Planner returned an invalid remaining-subtask list")
        return plan

    def run(self, task: dict) -> dict:
        if (not isinstance(task, dict) or
                any(not isinstance(task.get(key), str) or not task[key]
                    for key in ("task_id", "episode_id"))):
            raise ValueError("task_id and unique episode_id are required")
        start = time.monotonic()
        events: list[dict] = []
        completed: list[str] = []
        decisions = tool_calls = recall_calls = replans = 0
        recalled: tuple[RecalledImage, ...] = ()
        status = "error"
        error = None
        cleanup_error = None
        observation = None
        plan: list[Subtask] = []
        index = 0
        active_subtask_id = None

        def emit(event: dict):
            # Raw images are persisted only in SubtaskImageMemory, never in trace JSON.
            event = {"task_id": task["task_id"], "episode_id": task["episode_id"], **event}
            event = deepcopy(event)
            events.append(event)
            if self.trace_path:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")

        try:
            if any(r.task_id == task["task_id"] and r.episode_id == task["episode_id"]
                   for r in self.memory.records):
                raise ValueError("Episode ID already has image memory; use a fresh episode ID")
            observation = self.environment.reset(task)
            self._images(observation)
            plan = self._plan(task, observation, (), None)
            emit({"event": "plan", "subtasks": [asdict(s) for s in plan]})
            while True:
                if time.monotonic() - start >= self.config.max_seconds:
                    status = "time_budget_exhausted"
                    break
                if decisions >= self.config.max_decisions:
                    status = "decision_budget_exhausted"
                    break
                subtask = plan[index]
                if active_subtask_id != subtask.id:
                    begin = getattr(self.environment, "begin_subtask", None)
                    if begin is not None:
                        begin(subtask)
                    active_subtask_id = subtask.id
                if not any(e.get("event") == "subtask_start" and e.get("subtask_id") == subtask.id for e in events):
                    snapshots = (self.memory.capture(task_id=task["task_id"],
                                                    episode_id=task["episode_id"], subtask_id=subtask.id,
                                                    instruction=subtask.instruction, tags=subtask.tags,
                                                    images=self._images(observation))
                                 if self.config.enable_image_memory else [])
                    emit({"event": "subtask_start", "subtask_id": subtask.id,
                          "images": [{"camera": r.camera, "sha256": r.sha256,
                                      "filename": r.filename} for r in snapshots]})
                    recalled = ()
                context = AgentContext(deepcopy(task), subtask, deepcopy(observation),
                                       tuple(deepcopy(events)), recalled, tuple(completed))
                decision = self.policy.decide(context)
                decisions += 1
                if time.monotonic() - start >= self.config.max_seconds:
                    status = "time_budget_exhausted"
                    break
                if not isinstance(decision, Decision):
                    raise ValueError("Policy must return a Decision")
                if decision.kind == "recall":
                    if not self.config.enable_image_memory:
                        raise ValueError("Image recall is disabled for this condition")
                    if recall_calls >= self.config.max_recall_calls:
                        status = "recall_budget_exhausted"
                        break
                    recalled = tuple(self.memory.recall(task_id=task["task_id"],
                        episode_id=task["episode_id"],
                        subtask_id=decision.recall_subtask_id or None,
                        tags=decision.recall_tags, limit=self.config.recall_limit))
                    recall_calls += 1
                    emit({"event": "recall", "current_subtask_id": subtask.id,
                          "requested_subtask_id": decision.recall_subtask_id,
                          "count": len(recalled),
                          "images": [{"subtask_id": r.record.subtask_id,
                                      "camera": r.record.camera, "sha256": r.record.sha256}
                                     for r in recalled]})
                elif decision.kind == "replan":
                    if not self.config.enable_replanning:
                        raise ValueError("Replanning is disabled for this condition")
                    if replans >= self.config.max_replans:
                        status = "replan_budget_exhausted"
                        break
                    failure = next((e["receipt"] for e in reversed(events)
                                    if e.get("event") == "tool" and not e["receipt"].get("ok")), None)
                    plan = self._plan(task, observation, tuple(completed), failure)
                    index = 0
                    active_subtask_id = None
                    replans += 1
                    recalled = ()
                    emit({"event": "replan", "subtasks": [asdict(s) for s in plan],
                          "failure": failure})
                elif decision.kind == "tool":
                    if decision.tool not in self.config.allowed_tools or not isinstance(decision.args, dict):
                        raise ValueError("Policy requested a tool outside the allowlist or invalid arguments")
                    if tool_calls >= self.config.max_tool_calls:
                        status = "tool_budget_exhausted"
                        break
                    json.dumps(decision.args)
                    result = self.environment.execute(decision.tool, decision.args)
                    if (not isinstance(result, StepResult) or not isinstance(result.receipt, dict)
                            or not isinstance(result.receipt.get("ok"), bool)
                            or not isinstance(result.observation, dict)):
                        raise ValueError("Environment returned an invalid StepResult")
                    if result.subtask_done and not result.receipt["ok"]:
                        raise ValueError("Environment returned contradictory completion flags")
                    json.dumps(result.receipt)
                    tool_calls += 1
                    observation = result.observation
                    self._images(observation)
                    emit({"event": "tool", "subtask_id": subtask.id, "tool": decision.tool,
                          "args": decision.args, "receipt": result.receipt,
                          "subtask_done": result.subtask_done, "task_done": result.task_done,
                          "episode_end": result.episode_end})
                    recalled = ()
                    if time.monotonic() - start >= self.config.max_seconds:
                        status = "time_budget_exhausted"
                        break
                    if result.subtask_done:
                        completed.append(subtask.id)
                        emit({"event": "subtask_complete", "subtask_id": subtask.id})
                    if result.task_done:
                        status = "success"
                        break
                    if result.episode_end:
                        status = "environment_terminated_without_success"
                        break
                    if result.subtask_done:
                        if index == len(plan) - 1:
                            status = "plan_exhausted_without_task_success"
                            break
                        index += 1
                else:
                    raise ValueError(f"Unknown decision kind: {decision.kind}")
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            emit({"event": "error", "message": error})
        finally:
            close = getattr(self.environment, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
        summary = {"task_id": task["task_id"], "episode_id": task["episode_id"],
                   "status": status, "success": status == "success",
                   "completed_subtasks": completed, "decisions": decisions,
                   "tool_calls": tool_calls, "recall_calls": recall_calls,
                   "replans": replans, "elapsed_seconds": round(time.monotonic() - start, 3),
                   "error": error, "cleanup_error": cleanup_error, "events": events}
        return summary
