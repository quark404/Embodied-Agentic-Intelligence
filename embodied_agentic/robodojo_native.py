"""Run a harness inside RoboDojo's official XPolicyLab eval client process.

This module is imported only by the simulator-side deploy.py entry. RoboDojo
owns scene reset, task rewards, terminal state, video, and official _result.json.
No success or privileged reward is included in observations sent to the model.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Callable

from .harness import StepResult, Subtask


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = ("left_arm_joint_state", "left_ee_joint_state", "left_ee_pose",
              "right_arm_joint_state", "right_ee_joint_state", "right_ee_pose")


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _camera_png(color) -> bytes:
    import numpy as np
    from PIL import Image

    pixels = np.asarray(color)
    if (pixels.ndim != 3 or pixels.shape[2] not in (3, 4)
            or pixels.shape[0] < 1 or pixels.shape[1] < 1
            or pixels.dtype != np.uint8 or not pixels[..., :3].any()):
        raise ValueError("RoboDojo camera image is blank or not uint8 RGB(A)")
    output = BytesIO()
    Image.fromarray(pixels[..., :3]).save(output, format="PNG")
    return output.getvalue()


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw)


def convert_ee_delta_actions(rows: list[dict], state: dict, *, max_chunk: int,
                             max_translation_m: float, max_rotation_rad: float) -> list[dict]:
    """World-frame ee_delta -> native EE poses, equivalent to RoboICL validate_actions.

    Translation/rotation bounds match RoboICL DialogueModel._convert_tensor.
    Each row is relative to the previous proposed EE target within the chunk.
    This conversion does not solve IK; RoboDojo's execution-side guard must do so.
    """
    import numpy as np

    if not isinstance(rows, list) or not 1 <= len(rows) <= max_chunk:
        raise ValueError(f"actions must contain 1..{max_chunk} control steps")
    targets = {}
    for side in ("left", "right"):
        pose = np.asarray(state.get(f"{side}_ee_pose", []), dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"Observed {side} EE pose is required for ee_delta")
        norm = float(np.linalg.norm(pose[3:]))
        if norm < 1e-6 or abs(norm - 1) > 0.05:
            raise ValueError(f"Observed {side} quaternion must be unit w,x,y,z")
        targets[side] = pose.copy()
    converted = []
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) !=
                {"mode", "left", "right", "left_gripper", "right_gripper"}
                or row["mode"] != "ee_delta"):
            raise ValueError("ee_delta action requires exactly mode,left,right,left_gripper,right_gripper")
        native = {}
        for side in ("left", "right"):
            raw = row[side]
            if (not isinstance(raw, (list, tuple)) or len(raw) != 6
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in raw)):
                raise ValueError(f"actions[{index}].{side} must have six finite numbers")
            values = np.asarray(raw, dtype=np.float64)
            translation = float(np.linalg.norm(values[:3]))
            rotation = float(np.linalg.norm(values[3:]))
            if translation > max_translation_m + 1e-6 or rotation > max_rotation_rad + 1e-6:
                raise ValueError(f"actions[{index}].{side} exceeds EE translation/rotation norm")
            gripper = row[f"{side}_gripper"]
            if type(gripper) not in (int, float) or not math.isfinite(gripper) or not 0 <= gripper <= 1:
                raise ValueError(f"actions[{index}].{side}_gripper must be in [0,1]")
            previous = targets[side]
            dq = np.r_[math.cos(rotation / 2),
                       values[3:] * (math.sin(rotation / 2) / rotation
                                     if rotation > 1e-10 else 0.5)]
            quat = np.asarray(_quat_mul(dq, previous[3:]), dtype=np.float64)
            norm = float(np.linalg.norm(quat))
            if norm < 1e-6 or abs(norm - 1) > 0.05:
                raise ValueError("Converted EE quaternion is invalid")
            pose = np.r_[previous[:3] + values[:3], quat / norm].astype(np.float32)
            targets[side] = pose.astype(np.float64)
            native[f"{side}_ee_pose"] = pose
            native[f"{side}_ee_joint_state"] = np.asarray([gripper], dtype=np.float32)
        converted.append(native)
    return converted


def install_native_ik_guard(env, max_joint_increment_rad: float):
    """Require RoboICL's guard-capable RoboDojo fork before EE execution.

    Merely attaching a guard to stock RoboDojo is ineffective: its action
    dispatcher has no pre-queue IK check. Fail before starting the policy.
    """
    try:
        from src.eval_client.ik_continuity import IKContinuityError, IKContinuityGuard
    except ImportError as exc:
        raise RuntimeError(
            "ee_delta requires RoboICL's pinned RoboDojo fork with "
            "src.eval_client.ik_continuity and pre-queue IK checks"
        ) from exc

    try:
        dispatch = inspect.getsource(type(env).take_action_batch)
    except (AttributeError, OSError, TypeError) as exc:
        raise RuntimeError(
            "Cannot verify RoboDojo's pre-queue IK guard in take_action_batch; "
            "ee_delta requires RoboICL's pinned RoboDojo fork"
        ) from exc
    if "ik_guard.check(" not in dispatch or "IKContinuityError" not in dispatch:
        raise RuntimeError("RoboDojo dispatch lacks RoboICL's pre-queue IK guard; use its pinned RoboDojo fork")
    env.ik_continuity_guard = IKContinuityGuard(
        max_joint_increment_rad,
        log_path=Path(env.save_dir) / "_ik_continuity.jsonl")
    return IKContinuityError


class RoboDojoNativeAdapter:
    """Single-environment bridge around RoboDojo's already active TASK_ENV.

    `act` accepts only the named native action dictionary, never an untyped 14D
    vector. RoboDojo's own `validate_action_dict` and `take_action` remain in
    charge. `max_joint_delta` can reject too-large joint targets without
    advancing physics; set a measured limit appropriate to the controller.
    """

    def __init__(self, env, *, initial_raw: dict | None = None,
                 action_type: str = "joint", max_chunk: int = 16,
                 max_joint_delta: float | None = None,
                 max_translation_m: float = 0.03,
                 max_rotation_rad: float = 0.15,
                 ik_error_type: type[Exception] | None = None,
                 subtask_verifier: Callable | None = None,
                 max_camera_retries: int = 30):
        if getattr(env, "num_envs", None) != 1 or getattr(env, "eval_batch", False):
            raise ValueError("RoboDojo harness requires one non-batched native environment")
        if action_type not in ("joint", "ee_delta"):
            raise ValueError("Supported native action modes are joint and ee_delta")
        if max_chunk < 1 or max_camera_retries < 0:
            raise ValueError("Invalid action chunk or camera retry limit")
        if max_joint_delta is not None and (not math.isfinite(max_joint_delta)
                                            or max_joint_delta <= 0):
            raise ValueError("Joint step bound must be positive and finite")
        if (not math.isfinite(max_translation_m) or max_translation_m <= 0
                or not math.isfinite(max_rotation_rad) or max_rotation_rad <= 0):
            raise ValueError("EE motion bounds must be positive and finite")
        if action_type == "ee_delta" and (getattr(env, "ik_continuity_guard", None) is None
                                           or ik_error_type is None):
            raise ValueError("ee_delta requires a native pre-queue IK guard and rejection type")
        if action_type == "ee_delta":
            info = getattr(env, "robot_action_dim_info", None)
            if (not isinstance(info, dict) or info.get("arm_dim") != [6, 6]
                    or info.get("ee_dim") != [1, 1]):
                raise ValueError("ee_delta requires the RoboICL dual-arm ARX X5 action dimensions")
        if not callable(getattr(env, "validate_action_dict", None)):
            raise ValueError("RoboDojo native action validator is required")
        self.env = env
        self._initial_raw = initial_raw
        self.max_chunk = max_chunk
        self.max_joint_delta = max_joint_delta
        self.action_type = action_type
        self.max_translation_m = max_translation_m
        self.max_rotation_rad = max_rotation_rad
        self.ik_error_type = ik_error_type
        self.subtask_verifier = subtask_verifier
        self.max_camera_retries = max_camera_retries
        self.current_subtask: Subtask | None = None
        self.last_observation: dict | None = None

    def tool_schemas(self) -> dict:
        if self.action_type == "ee_delta":
            vector = {"type": "array", "minItems": 6, "maxItems": 6,
                      "items": {"type": "number"},
                      "description": (
                          "World-frame incremental end-effector target "
                          "[dx,dy,dz,rx,ry,rz]: translation in metres, then "
                          "axis-angle rotation vector in radians. Apply each row "
                          "relative to the preceding proposed target, starting "
                          "from the observed end-effector pose; native pose "
                          "quaternions are ordered w,x,y,z. "
                          f"Translation L2 norm <= {self.max_translation_m} m and "
                          f"rotation L2 norm <= {self.max_rotation_rad} rad.")}
            row = {"type": "object", "properties": {
                "mode": {"type": "string", "enum": ["ee_delta"],
                         "description": "Harness EE delta mode; official RoboDojo action_type is ee."},
                "left": vector, "right": vector,
                "left_gripper": {"type": "number", "minimum": 0, "maximum": 1,
                                  "description": "Normalized opening target: 0 fully closed, 1 fully open."},
                "right_gripper": {"type": "number", "minimum": 0, "maximum": 1,
                                   "description": "Normalized opening target: 0 fully closed, 1 fully open."}},
                "required": ["mode", "left", "right", "left_gripper", "right_gripper"],
                "additionalProperties": False}
            return {"act": {"type": "object",
                "description": "Send a short dual-arm EE target sequence; rejected IK targets execute zero steps.",
                "properties": {
                "actions": {"type": "array", "minItems": 1,
                            "maxItems": self.max_chunk, "items": row,
                            "description": "Sequential RoboDojo control steps, at most the configured chunk size."}},
                "required": ["actions"], "additionalProperties": False}}
        info = self.env.robot_action_dim_info
        arm_dim = info["arm_dim"]
        ee_dim = info["ee_dim"]
        if len(arm_dim) != 2 or len(ee_dim) != 2:
            raise ValueError("Expected a dual-arm ARX X5 embodiment")
        properties = {}
        for index, prefix in enumerate(("left", "right")):
            for field, dimension in (("arm_joint_state", arm_dim[index]),
                                     ("ee_joint_state", ee_dim[index])):
                properties[f"{prefix}_{field}"] = {
                    "type": "array", "minItems": dimension, "maxItems": dimension,
                    "items": {"type": "number"},
                    "description": (
                        "Normalized gripper opening target: 0 fully closed, 1 fully open."
                        if field == "ee_joint_state" else
                        "Absolute arm joint targets in radians; compare against current joint state."
                        + (f" Each joint change must be <= {self.max_joint_delta} rad."
                           if self.max_joint_delta is not None else ""))}
        row = {"type": "object", "properties": properties,
               "required": list(properties), "additionalProperties": False}
        return {"act": {"type": "object",
            "description": "Send sequential absolute dual-arm joint targets in radians.",
            "properties": {
            "actions": {"type": "array", "minItems": 1,
                        "maxItems": self.max_chunk, "items": row}},
            "required": ["actions"], "additionalProperties": False}}

    def begin_subtask(self, subtask: Subtask):
        self.current_subtask = subtask

    def _ready_raw(self) -> tuple[dict, dict[str, bytes]]:
        for attempt in range(self.max_camera_retries + 1):
            raw = self._initial_raw if self._initial_raw is not None else self.env.get_obs()
            self._initial_raw = None
            try:
                images = {name: _camera_png(raw["vision"][name]["color"])
                          for name in CAMERAS}
                return raw, images
            except (KeyError, TypeError, ValueError):
                if attempt == self.max_camera_retries:
                    raise ValueError("RoboDojo cameras unavailable after render-only retries")
        raise AssertionError("unreachable")

    def _observe(self) -> dict:
        raw, images = self._ready_raw()
        state = raw.get("state") or {}
        counts = getattr(self.env, "take_action_cnt", None)
        used = int(counts[0]) if isinstance(counts, (tuple, list)) else None
        limit = getattr(self.env, "step_lim", None)
        observation = {
            "instruction": str(raw.get("instruction") or raw.get("instructions") or ""),
            "state": {key: _plain(state[key]) for key in STATE_KEYS if key in state},
            "images": images,
        }
        if used is not None and isinstance(limit, int):
            observation["remaining_steps"] = max(0, limit - used)
        self.last_observation = observation
        return observation

    def reset(self, task: dict) -> dict:
        # The official client has already selected and reset this layout.
        if self.last_observation is not None:
            raise ValueError("A native RoboDojo adapter cannot be reused across episodes")
        self.current_subtask = None
        return self._observe()

    def _validate_action(self, action: dict) -> dict:
        import numpy as np

        if not isinstance(action, dict):
            raise ValueError("Each action must be a named native dictionary")
        schema = self.tool_schemas()["act"]["properties"]["actions"]["items"]
        if set(action) != set(schema["required"]):
            raise ValueError("Native joint action must have exactly four named keys")
        validated = {}
        for key, field in schema["properties"].items():
            values = action[key]
            if (not isinstance(values, (tuple, list)) or len(values) != field["minItems"]
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError(f"{key} must be a finite numeric vector of the expected size")
            if key.endswith("ee_joint_state") and any(not 0 <= v <= 1 for v in values):
                raise ValueError(f"{key} gripper target must be within [0, 1]")
            validated[key] = np.asarray(values, dtype=np.float32)
        self.env.validate_action_dict(validated)
        if self.max_joint_delta is not None:
            state = (self.last_observation or {}).get("state") or {}
            for key in ("left_arm_joint_state", "right_arm_joint_state"):
                measured = state.get(key)
                if not isinstance(measured, list) or len(measured) != len(validated[key]):
                    raise ValueError(f"Measured {key} is required for the joint step guard")
                if any(abs(float(target) - float(actual)) > self.max_joint_delta + 1e-6
                       for target, actual in zip(validated[key], measured)):
                    raise ValueError(f"{key} exceeds the configured joint step bound")
        return validated

    def execute(self, tool: str, args: dict) -> StepResult:
        if tool != "act":
            raise ValueError("RoboDojo adapter only accepts the act tool")
        if self.last_observation is None:
            raise ValueError("RoboDojo adapter must be reset before an action")
        if (not isinstance(args, dict) or set(args) != {"actions"}
                or not isinstance(args.get("actions"), list)):
            return StepResult(self.last_observation,
                {"ok": False, "submitted_steps": 0, "executed_steps": 0,
                 "discarded_steps": 0,
                 "rejection": "act requires exactly an actions list", "terminal": False})
        rows = args["actions"]
        if not 1 <= len(rows) <= self.max_chunk:
            return StepResult(self.last_observation,
                {"ok": False, "submitted_steps": len(rows), "executed_steps": 0,
                 "discarded_steps": len(rows),
                 "rejection": f"Native action chunk must contain 1..{self.max_chunk} steps",
                 "terminal": False})
        if self.env.is_episode_end():
            raise ValueError("RoboDojo episode already ended")
        executed = 0
        rejection = None
        observation = self.last_observation
        if self.action_type == "ee_delta":
            try:
                commands = convert_ee_delta_actions(
                    rows, (self.last_observation or {}).get("state") or {},
                    max_chunk=self.max_chunk,
                    max_translation_m=self.max_translation_m,
                    max_rotation_rad=self.max_rotation_rad)
                for command in commands:
                    self.env.validate_action_dict(command)
            except (TypeError, ValueError) as exc:
                commands = []
                rejection = str(exc)
        else:
            commands = rows
        for row in commands:
            if self.env.is_episode_end():
                break
            if self.action_type == "joint":
                try:
                    row = self._validate_action(row)
                except (TypeError, ValueError) as exc:
                    rejection = str(exc)
                    break
            try:
                self.env.take_action(row)
            except Exception as exc:
                if self.ik_error_type is None or not isinstance(exc, self.ik_error_type):
                    raise
                rejection = _plain(getattr(exc, "event", {"reason": str(exc)}))
                break
            executed += 1
            observation = self._observe()
            if self.env.is_episode_end():
                break
        terminal = bool(self.env.is_episode_end())
        success = bool(self.env.success[0]) if terminal else False
        receipt = {"ok": rejection is None,
                   "submitted_steps": len(rows), "executed_steps": executed,
                   "discarded_steps": len(rows) - executed,
                   "rejection": rejection, "terminal": terminal}
        subtask_done = bool(self.subtask_verifier(self.current_subtask, observation, receipt)) if self.subtask_verifier else False
        return StepResult(observation, receipt, subtask_done=subtask_done,
                          task_done=success, episode_end=terminal)


def _factory(spec: str):
    if ":" not in spec:
        raise ValueError("EMBODIED_HARNESS_FACTORY must be module:function")
    module, name = spec.split(":", 1)
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise ValueError("Harness factory is not callable")
    return result


def run_official_episode(task_env, *, factory: Callable | None = None) -> dict:
    """Entry called by XPolicyLab deploy.eval_one_episode, after native reset.

    When a harness budget ends before the simulator does, force RoboDojo's own
    final check. Its `success` array starts as True, so returning early without
    this step would make the official result incorrectly count as successful.
    """
    if getattr(task_env, "num_envs", None) != 1 or getattr(task_env, "eval_batch", False):
        raise ValueError("Use one RoboDojo environment; batch mode is unsupported")
    raw = task_env.get_obs()
    name = getattr(task_env, "task_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Native task name is required")
    layout = int(task_env.env_seeds[0])
    seed = int(task_env.eval_seed)
    run_id = str(task_env.run_id)
    index = int(task_env.success_nums + task_env.fail_nums)
    episode_id = f"{run_id}-{name}-seed{seed}-layout{layout}-trial{index}"
    task = {"task_id": name, "episode_id": episode_id,
            "goal": str(raw.get("instruction") or raw.get("instructions") or name),
            "seed": seed, "layout_id": layout}
    root = Path(task_env.save_dir) / "embodied_agentic" / episode_id
    action_type = os.environ.get("EMBODIED_ROBODOJO_CONTROL_MODE", "ee_delta")
    max_chunk = int(os.environ.get("EMBODIED_ROBODOJO_MAX_CHUNK", "15"))
    max_delta_raw = os.environ.get("EMBODIED_ROBODOJO_MAX_JOINT_DELTA", "0.05")
    max_delta = None if max_delta_raw.lower() == "none" else float(max_delta_raw)
    max_translation = float(os.environ.get("EMBODIED_ROBODOJO_MAX_TRANSLATION_M", "0.03"))
    max_rotation = float(os.environ.get("EMBODIED_ROBODOJO_MAX_ROTATION_RAD", "0.15"))
    max_ik_increment = float(os.environ.get("EMBODIED_ROBODOJO_MAX_IK_INCREMENT_RAD", "0.15"))
    ik_error_type = install_native_ik_guard(task_env, max_ik_increment) if action_type == "ee_delta" else None
    adapter = RoboDojoNativeAdapter(task_env, initial_raw=raw, action_type=action_type,
                                    max_chunk=max_chunk, max_joint_delta=max_delta,
                                    max_translation_m=max_translation,
                                    max_rotation_rad=max_rotation,
                                    ik_error_type=ik_error_type)
    build_harness = factory or _factory(os.environ.get("EMBODIED_HARNESS_FACTORY",
                                                   "embodied_agentic.experiment:build_harness"))
    harness = build_harness(adapter, task, root)
    summary = harness.run(task)
    if not isinstance(summary, dict) or "status" not in summary:
        raise ValueError("Harness factory returned no valid episode summary")
    if not task_env.is_episode_end():
        task_env.success[0] = False
        if not task_env.is_episode_end():
            raise RuntimeError("RoboDojo refused to close an interrupted episode")
    summary["native_terminal"] = True
    summary["native_success"] = bool(task_env.success[0])
    summary["seed"] = seed
    summary["layout_id"] = layout
    summary["native_result_index"] = index
    summary["condition"] = os.environ.get("EMBODIED_CONDITION", "q_only")
    summary["native_result_pending"] = str(Path(task_env.save_dir) / "_result.json")
    root.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in summary.items() if key != "events"}
    (root / "summary.json").write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return summary
