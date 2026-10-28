"""Paired-condition runner for controlled harness experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .harness import AgentPolicy, AgenticHarness, HarnessConfig, HarnessEnvironment, Planner, Subtask
from .image_memory import SubtaskImageMemory


@dataclass(frozen=True)
class Condition:
    name: str
    planner_factory: Callable[[], Planner]
    policy_factory: Callable[[], AgentPolicy]
    allowed_tools: tuple[str, ...]
    environment_factory: Callable[[], HarnessEnvironment] | None = None
    enable_image_memory: bool = True
    enable_replanning: bool = True
    agent_factory: Callable[[], tuple[Planner, AgentPolicy]] | None = None


class GoalOnlyPlanner:
    """One episode-wide subtask for direct and tool-only baselines."""

    def plan(self, task: dict, observation: dict, completed: tuple[str, ...],
             failure: dict | None) -> list[Subtask]:
        return [Subtask("whole_task", task.get("goal") or task["task_id"])]


def run_paired(*, tasks: list[dict], conditions: list[Condition],
               environment_factory: Callable[[], HarnessEnvironment],
               base_config: HarnessConfig, output_dir: str | Path, run_id: str) -> list[dict]:
    """Run every condition on every task/seed with identical decision/time limits.

    Factories provide fresh state for each episode. The caller is responsible for
    choosing scientifically comparable model checkpoints, observations, tools,
    and native benchmark adapters; the runner records rather than assumes them.
    """
    if not isinstance(run_id, str) or not run_id or not tasks or not conditions:
        raise ValueError("A run ID, tasks and conditions are required")
    if len({c.name for c in conditions}) != len(conditions) or any(not c.name for c in conditions):
        raise ValueError("Condition names must be unique and nonempty")
    keys = []
    for task in tasks:
        if (not isinstance(task, dict) or not isinstance(task.get("task_id"), str)
                or not task["task_id"] or type(task.get("seed")) not in (int, str)):
            raise ValueError("Each task requires task_id and seed")
        keys.append((task["task_id"], str(task["seed"])))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate task/seed pairs would bias paired results")
    root = Path(output_dir)
    if (root / "results.json").exists():
        raise ValueError("Output already contains results; use a fresh output directory")
    rows = []
    for task in tasks:
        for condition in conditions:
            identity = json.dumps([run_id, task["task_id"], task["seed"], condition.name],
                                  ensure_ascii=False, separators=(",", ":"))
            artifact_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            episode_id = f"{run_id}-{artifact_id}"
            episode_root = root / artifact_id
            config = replace(base_config, allowed_tools=condition.allowed_tools,
                             enable_image_memory=condition.enable_image_memory,
                             enable_replanning=condition.enable_replanning)
            planner, policy = (condition.agent_factory() if condition.agent_factory else
                               (condition.planner_factory(), condition.policy_factory()))
            harness = AgenticHarness(environment=(condition.environment_factory or environment_factory)(),
                planner=planner, policy=policy,
                memory=SubtaskImageMemory(episode_root / "memory"), config=config,
                trace_path=episode_root / "trace.jsonl")
            result = harness.run({**task, "episode_id": episode_id})
            row = {key: value for key, value in result.items() if key != "events"}
            row.update({"condition": condition.name, "seed": task["seed"],
                        "artifact_dir": str(episode_root),
                        "budget": {"max_decisions": config.max_decisions,
                                   "max_tool_calls": config.max_tool_calls,
                                   "max_recall_calls": config.max_recall_calls,
                                   "max_replans": config.max_replans,
                                   "max_seconds": config.max_seconds},
                        "allowed_tools": list(condition.allowed_tools)})
            row.update({"enable_image_memory": config.enable_image_memory,
                        "enable_replanning": config.enable_replanning})
            rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return rows
