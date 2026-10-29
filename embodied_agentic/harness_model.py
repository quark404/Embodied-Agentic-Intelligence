"""VLM decision adapter for an OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import base64
import json
import os
import time
from urllib.request import Request, urlopen

from .harness import AgentContext, Decision, Subtask


class ModelOutputError(ValueError):
    """A model returned a response that violates the frozen agent protocol."""


def _strict_json(raw: str):
    def reject_constant(value):
        raise ModelOutputError(f"Non-JSON numeric constant {value} is forbidden")
    return json.loads(raw, parse_constant=reject_constant)


def _post_json(endpoint: str, api_key_env: str, payload: dict,
               timeout_seconds: float) -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def _call_model(transport, payload: dict, events: list[dict], model: str,
                role: str) -> dict:
    started = time.monotonic()
    response = None
    failure = None
    try:
        response = transport(payload)
        if not isinstance(response, dict):
            raise ModelOutputError("Model endpoint returned no JSON object")
        return response
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        event = {"event": "model_call", "role": role, "model": model,
                 "ok": failure is None,
                 "latency_seconds": round(time.monotonic() - started, 3),
                 "usage": response.get("usage", {}) if isinstance(response, dict) else {}}
        if failure is not None:
            event["error"] = failure
        events.append(event)


def _image(data: bytes) -> dict:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        raise ValueError("Unsupported image format")
    return {"type": "image_url", "image_url": {
        "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}}


class HarnessChatPlanner:
    """Ask a multimodal model for ordered remaining subtasks, including replans."""

    def __init__(self, endpoint: str, model: str, *, allowed_tools: tuple[str, ...],
                 api_key_env: str = "MODEL_API_KEY", transport=None,
                 tool_schemas: dict | None = None, max_output_tokens: int = 8192,
                 timeout_seconds: float = 600):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.allowed_tools = allowed_tools
        self.api_key_env = api_key_env
        self.transport = transport or self._post
        self.tool_schemas = tool_schemas or {}
        if max_output_tokens < 1 or timeout_seconds <= 0:
            raise ValueError("max_output_tokens and timeout_seconds must be positive")
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self._events = []
        if not set(self.tool_schemas).issubset(allowed_tools):
            raise ValueError("Tool schemas must match allowed tool names")

    def _post(self, payload: dict) -> dict:
        return _post_json(self.endpoint, self.api_key_env, payload, self.timeout_seconds)

    def drain_events(self) -> list[dict]:
        events, self._events = self._events, []
        return events

    def plan(self, task: dict, observation: dict, completed: tuple[str, ...],
             failure: dict | None) -> list[Subtask]:
        info = {"goal": task.get("goal", ""), "task_id": task["task_id"],
                "episode_id": task["episode_id"], "completed_subtasks": completed,
                "failure_receipt": failure,
                "current_state": {k: v for k, v in observation.items() if k != "images"},
                "allowed_tools": self.allowed_tools,
                "tool_schemas": self.tool_schemas}
        content = [{"type": "text", "text": json.dumps(info, ensure_ascii=False)}]
        for camera, data in observation["images"].items():
            content.extend(({"type": "text", "text": f"Current image, camera={camera}"}, _image(data)))
        response = _call_model(self.transport, {"model": self.model, "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "system", "content":
                "Return only a raw JSON array, without Markdown code fences or surrounding text, "
                "of remaining subtasks, each with unique id, instruction, "
                "and optional string-list tags. Do not repeat completed subtask IDs. "
                "Use the current image and failure receipt when replanning."},
                {"role": "user", "content": content}]}, self._events, self.model, "planner")
        try:
            raw = _strict_json(response["choices"][0]["message"]["content"])
            if not isinstance(raw, list):
                raise ModelOutputError("Planner must return a JSON list")
            plan = []
            for item in raw:
                if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                        or not isinstance(item.get("instruction"), str)
                        or not isinstance(item.get("tags", []), list)
                        or any(not isinstance(tag, str) for tag in item.get("tags", []))):
                    raise ModelOutputError("Planner returned an invalid subtask")
                plan.append(Subtask(item["id"], item["instruction"], tuple(item.get("tags", []))))
            return plan
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ModelOutputError) as exc:
            self._events[-1].update(ok=False, error=f"{type(exc).__name__}: {exc}")
            raise ModelOutputError(f"Planner response violated JSON protocol: {exc}") from exc


class HarnessChatPolicy:
    """Pass current and recalled images to a model; validate its one-step decision."""

    def __init__(self, endpoint: str, model: str, *, allowed_tools: tuple[str, ...],
                 api_key_env: str = "MODEL_API_KEY", transport=None,
                 tool_schemas: dict | None = None,
                 allowed_decisions: tuple[str, ...] = ("tool", "recall", "replan", "advance", "consult"),
                 max_output_tokens: int = 8192, timeout_seconds: float = 600):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.allowed_tools = allowed_tools
        self.api_key_env = api_key_env
        self.transport = transport or self._post
        self.tool_schemas = tool_schemas or {}
        self.allowed_decisions = allowed_decisions
        if max_output_tokens < 1 or timeout_seconds <= 0:
            raise ValueError("max_output_tokens and timeout_seconds must be positive")
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self._events = []
        if not set(self.tool_schemas).issubset(allowed_tools):
            raise ValueError("Tool schemas must match allowed tool names")
        if not allowed_decisions or not set(allowed_decisions).issubset({"tool", "recall", "replan", "advance", "consult"}):
            raise ValueError("Unsupported allowed decision kinds")

    def _post(self, payload: dict) -> dict:
        return _post_json(self.endpoint, self.api_key_env, payload, self.timeout_seconds)

    def drain_events(self) -> list[dict]:
        events, self._events = self._events, []
        return events

    @staticmethod
    def _image(data: bytes) -> dict:
        return _image(data)

    def decide(self, context: AgentContext) -> Decision:
        # The policy sees only bounded trace metadata; raw images travel in image parts.
        events = context.history[-10:]
        prompt = {"goal": context.task.get("goal", ""), "task_id": context.task["task_id"],
                  "episode_id": context.task["episode_id"],
                  "subtask": {"id": context.subtask.id,
                              "instruction": context.subtask.instruction},
                  "state": {k: v for k, v in context.observation.items() if k != "images"},
                  "recent_events": events, "completed_subtasks": context.completed_subtasks,
                  "allowed_tools": self.allowed_tools,
                  "allowed_decisions": self.allowed_decisions,
                  "remaining_subtasks": [{"id": item.id, "instruction": item.instruction}
                                         for item in context.remaining_subtasks],
                  "can_advance": len(context.remaining_subtasks) > 1,
                  "tool_schemas": self.tool_schemas,
                  "recalled_images": [{"subtask_id": r.record.subtask_id,
                                       "instruction": r.record.instruction,
                                       "camera": r.record.camera, "sha256": r.record.sha256}
                                      for r in context.recalled_images]}
        content = [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]
        for camera, data in context.observation["images"].items():
            content.append({"type": "text", "text": f"Current image, camera={camera}"})
            content.append(self._image(data))
        for recalled in context.recalled_images:
            content.append({"type": "text", "text":
                            f"Historical subtask-start image: {recalled.record.subtask_id}, "
                            f"camera={recalled.record.camera}. This is not the current observation."})
            content.append(self._image(recalled.data))
        response = _call_model(self.transport, {"model": self.model, "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "system", "content":
                "Return exactly one raw JSON object without Markdown code fences or surrounding text, "
                "using only allowed_decisions. Choices: "
                "{\"kind\":\"tool\",\"tool\":name,\"args\":{...}}, "
                "{\"kind\":\"recall\",\"recall_subtask_id\":id}, or "
                "{\"kind\":\"replan\"}, or {\"kind\":\"advance\"} to move to the next "
                "planned subtask after checking the latest observation, or "
                "{\"kind\":\"consult\"} to request bounded diagnosis after an action. "
                "An advance is provisional; "
                "it cannot claim benchmark success. Use only allowed tools. "
                "A historical image is a past starting view, not current state. "
                "Trust environment receipts for completion."},
                {"role": "user", "content": content}]}, self._events, self.model, "actor")
        try:
            parsed = _strict_json(response["choices"][0]["message"]["content"])
            if not isinstance(parsed, dict) or parsed.get("kind") not in (
                    "tool", "recall", "replan", "advance", "consult"):
                raise ModelOutputError("Model returned an invalid decision")
            kind = parsed["kind"]
            if kind not in self.allowed_decisions:
                raise ModelOutputError("Model requested a disabled decision kind")
            if kind == "tool":
                if parsed.get("tool") not in self.allowed_tools or not isinstance(parsed.get("args"), dict):
                    raise ModelOutputError("Model requested an invalid tool call")
                return Decision(kind, tool=parsed["tool"], args=parsed["args"])
            if kind == "recall":
                subtask_id = parsed.get("recall_subtask_id", "")
                if not isinstance(subtask_id, str) or not subtask_id:
                    raise ModelOutputError("Recall requires a subtask ID")
                return Decision(kind, recall_subtask_id=subtask_id)
            return Decision(kind)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ModelOutputError) as exc:
            self._events[-1].update(ok=False, error=f"{type(exc).__name__}: {exc}")
            raise ModelOutputError(f"Actor response violated JSON protocol: {exc}") from exc
