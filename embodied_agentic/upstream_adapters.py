"""Adapters for initialized PyRUA-Lean and RoboICL runtime objects.

These adapters have no import-time dependency on either upstream project.
They do not boot simulators or replace native benchmark scoring.
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Callable

from .harness import StepResult, Subtask


def _plain(value):
    """Keep public observation state JSON-safe without exposing image arrays."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class PyRUALeanAdapter:
    """Drive PyRUA's supported `ArmToolkit` cells arm after its host boots it.

    `toolkit.robo` owns artifacts and the benchmark state. `finish` is excluded:
    an agent's declared success is not a native verdict. `subtask_verifier` is
    a project-defined progress check; use the benchmark's own final success for
    the task verdict. One fresh toolkit/robot is needed for each episode.
    """

    def __init__(self, toolkit, *, image_artifacts: dict[str, str],
                 subtask_verifier: Callable | None = None,
                 close_callback: Callable | None = None):
        if getattr(toolkit, "arm", None) != "cells":
            raise ValueError("PyRUA adapter needs the multi-turn cells arm")
        if not image_artifacts:
            raise ValueError("At least one PyRUA image artifact is required")
        self.toolkit = toolkit
        self.image_artifacts = dict(image_artifacts)
        self.subtask_verifier = subtask_verifier
        self.close_callback = close_callback
        self.current_subtask: Subtask | None = None

    def close(self):
        if self.close_callback:
            self.close_callback()

    def begin_subtask(self, subtask: Subtask):
        self.current_subtask = subtask

    def tool_schemas(self) -> dict:
        return {item["name"]: item["input_schema"]
                for item in self.toolkit.get_tools_spec() if item["name"] == "python"}

    def _observe(self) -> dict:
        robo = self.toolkit.robo
        return {"instruction": str(robo.task),
                "state": _plain(robo._summary_dict()),
                "images": {camera: robo.artifact(artifact, step=-1)
                           for camera, artifact in self.image_artifacts.items()}}

    def reset(self, task: dict) -> dict:
        # PyRUA's boot()/XPolicyLab host already reset the underlying simulator.
        self.current_subtask = None
        return self._observe()

    def execute(self, tool: str, args: dict) -> StepResult:
        if tool != "python" or not isinstance(args.get("code"), str):
            raise ValueError("PyRUA cells adapter only accepts python(code=...)")
        arm_result = self.toolkit.execute_tool("python", args)
        payload = getattr(arm_result, "result", None)
        if not isinstance(payload, dict):
            raise ValueError("PyRUA ArmToolkit returned no result dictionary")
        receipt = _plain(payload)
        receipt["ok"] = not (payload.get("error") or payload.get("exception"))
        observation = self._observe()
        success = payload.get("episode_done") is True
        ended = bool(payload.get("episode_finished")) or bool(observation["state"].get("truncated"))
        subtask_done = bool(self.subtask_verifier(self.current_subtask, observation, receipt)) if self.subtask_verifier else False
        return StepResult(observation, receipt, subtask_done=subtask_done,
                          task_done=success, episode_end=ended or success)


class RoboICLAdapter:
    """Use RoboICL's RoboDojo observation/action contract in an initialized env.

    The official evaluator owns reset, guarded execution and final scoring.
    Pass its guarded `action_executor`, camera-ready `observation_reader`, and
    validated `action_converter`. For the standard DialogueModel [T,14] format,
    use its `_convert_tensor(rows, state)` with the matching frozen config; a
    raw tensor must never be sent directly to `env.take_action`.
    """

    CAMERAS = ("cam_left_wrist", "cam_head", "cam_right_wrist")
    STATE_KEYS = ("left_arm_joint_state", "left_ee_joint_state", "left_ee_pose",
                  "right_arm_joint_state", "right_ee_joint_state", "right_ee_pose")

    def __init__(self, env, *, action_converter: Callable | None = None,
                 action_executor: Callable, observation_reader: Callable,
                 success_reader: Callable | None = None,
                 subtask_verifier: Callable | None = None,
                 action_format: str = "tensor", max_chunk: int = 32,
                 close_callback: Callable | None = None):
        if action_format not in ("tensor", "structured") or max_chunk < 1:
            raise ValueError("Unsupported RoboICL action format or chunk limit")
        if action_converter is None:
            if action_format != "structured":
                raise ValueError("Tensor actions require RoboICL's validated converter")
            def action_converter(rows, state):
                from roboicl.policy.astra_policy import validate_actions
                return validate_actions(rows, max_chunk=max_chunk, state=state)
        self.env = env
        self.action_converter = action_converter
        self.action_executor = action_executor
        self.observation_reader = observation_reader
        self.success_reader = success_reader
        self.subtask_verifier = subtask_verifier
        self.close_callback = close_callback
        self.action_format = action_format
        self.max_chunk = max_chunk
        self.current_subtask: Subtask | None = None
        self.last_raw_observation = None
        self.last_observation = None

    def close(self):
        if self.close_callback:
            self.close_callback()

    def begin_subtask(self, subtask: Subtask):
        self.current_subtask = subtask

    def tool_schemas(self) -> dict:
        row = ({"type": "array", "minItems": 14, "maxItems": 14,
                "items": {"type": "number"}} if self.action_format == "tensor" else
               {"type": "object", "description": "RoboICL validated joint/ee/ee_delta action"})
        return {"act": {"type": "object", "properties": {
            "actions": {"type": "array", "minItems": 1, "maxItems": self.max_chunk,
                        "items": row},
            "execution_note": {"type": "string"}}, "required": ["actions"]}}

    def _observe(self) -> dict:
        from PIL import Image
        import numpy as np

        raw = self.observation_reader()
        self.last_raw_observation = raw
        images = {}
        for camera in self.CAMERAS:
            pixels = np.asarray(raw["vision"][camera]["color"])
            if (pixels.ndim != 3 or pixels.shape[-1] not in (3, 4) or
                    not pixels.size or not np.isfinite(pixels).all() or
                    not np.any(pixels[..., :3])):
                raise ValueError(f"RoboICL camera {camera} is blank or invalid")
            stream = BytesIO()
            Image.fromarray(pixels[..., :3].astype(np.uint8)).save(stream, format="JPEG", quality=90)
            images[camera] = stream.getvalue()
        state = raw.get("state") or {}
        self.last_observation = {"instruction": str(raw.get("instruction", "")),
                                 "state": {key: _plain(state[key]) for key in self.STATE_KEYS if key in state},
                                 "images": images}
        return self.last_observation

    def reset(self, task: dict) -> dict:
        # RoboICL's robodojo.py/official evaluator has already reset the layout.
        self.current_subtask = None
        return self._observe()

    def execute(self, tool: str, args: dict) -> StepResult:
        if tool != "act" or not isinstance(args.get("actions"), list):
            raise ValueError("RoboICL adapter only accepts act(actions=[...])")
        rows = args["actions"]
        if not 1 <= len(rows) <= self.max_chunk:
            raise ValueError("RoboICL action chunk is outside the configured limit")
        if self.action_format == "tensor" and any(
            not isinstance(row, list) or len(row) != 14 or
            any(type(value) not in (int, float) or not math.isfinite(value) for value in row)
            for row in rows
        ):
            raise ValueError("RoboICL tensor actions must be finite numeric [T,14]")
        if self.env.is_episode_end():
            raise ValueError("RoboICL episode already ended")
        packed = self.action_converter(rows, self.last_raw_observation["state"])
        if not isinstance(packed, list) or not packed:
            raise ValueError("RoboICL action converter returned no commands")
        executed = 0
        observation = self.last_observation
        controller_events = []
        for action in packed:
            if self.env.is_episode_end():
                break
            feedback = self.action_executor(action)
            if isinstance(feedback, dict) and feedback.get("ok") is False:
                controller_events.append(_plain(feedback))
                break
            executed += 1
            observation = self._observe()  # A fresh post-action image, as in RoboICL rollout.
            if isinstance(feedback, dict) and feedback.get("interrupt"):
                controller_events.append(_plain(feedback))
                break
        terminal = bool(self.env.is_episode_end())
        success = bool(self.success_reader(self.env)) if terminal and self.success_reader else False
        receipt = {"ok": not any(e.get("ok") is False for e in controller_events),
                   "submitted_steps": len(packed), "executed_steps": executed,
                   "discarded_steps": len(packed) - executed, "terminal": terminal,
                   "controller_events": controller_events,
                   "execution_note": str(args.get("execution_note", ""))[:400]}
        subtask_done = bool(self.subtask_verifier(self.current_subtask, observation, receipt)) if self.subtask_verifier else False
        return StepResult(observation, receipt, subtask_done=subtask_done,
                          task_done=success, episode_end=terminal)


def boot_pyrualean_libero(cell, out_dir: str | Path, *, rpent_root: str | None = None,
                          max_turns: int = 40, primitives: str = "no-vla",
                          env_only: bool = False) -> PyRUALeanAdapter:
    """Assemble the LIBERO cells arm using PyRUA's documented RPent host.

    This function has not been run in this workspace because RPent/GPU assets
    are unavailable. The returned adapter closes its toolkit and daemons when
    `AgenticHarness.run` ends. Call it once per episode via a fresh factory.
    """
    from pyrualean._backend import Ledger
    from pyrualean.arms import ArmToolkit
    from pyrualean.hosts.common import RpentBackend, stop_all
    from pyrualean.hosts.rpent_libero import boot, dump_card
    from pyrualean.libero import LiberoRobot

    out_dir = Path(out_dir)
    toolkit, daemons = boot(cell, out_dir, rpent_root=rpent_root, env_only=env_only)
    try:
        dump_card(toolkit, cell, out_dir)  # materialize step-0 camera artifacts
        robo = LiberoRobot(RpentBackend(toolkit), ledger=Ledger(), primitives=primitives)
        arm_toolkit = ArmToolkit(robo, arm="cells", workspace=out_dir / "workspace",
                                images="on-demand", feedback="rich", max_turns=max_turns)
        return PyRUALeanAdapter(arm_toolkit,
            image_artifacts={"agentview": "agentview_high.png", "wrist": "wrist_high.png"},
            close_callback=lambda: stop_all(toolkit, daemons))
    except BaseException:
        stop_all(toolkit, daemons)
        raise
