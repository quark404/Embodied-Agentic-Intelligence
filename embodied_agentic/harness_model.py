"""VLM decision adapter for an OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import base64
import json
import os
from urllib.request import Request, urlopen

from .harness import AgentContext, Decision, Subtask


def _post_json(endpoint: str, api_key_env: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    with urlopen(request, timeout=60) as response:
        return json.load(response)


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
                 api_key_env: str = "MODEL_API_KEY", transport=None):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.allowed_tools = allowed_tools
        self.api_key_env = api_key_env
        self.transport = transport or self._post

    def _post(self, payload: dict) -> dict:
        return _post_json(self.endpoint, self.api_key_env, payload)

    def plan(self, task: dict, observation: dict, completed: tuple[str, ...],
             failure: dict | None) -> list[Subtask]:
        info = {"goal": task.get("goal", ""), "task_id": task["task_id"],
                "episode_id": task["episode_id"], "completed_subtasks": completed,
                "failure_receipt": failure,
                "current_state": {k: v for k, v in observation.items() if k != "images"},
                "allowed_tools": self.allowed_tools}
        content = [{"type": "text", "text": json.dumps(info, ensure_ascii=False)}]
        for camera, data in observation["images"].items():
            content.extend(({"type": "text", "text": f"Current image, camera={camera}"}, _image(data)))
        response = self.transport({"model": self.model, "temperature": 0,
            "messages": [{"role": "system", "content":
                "Return a JSON array of remaining subtasks, each with unique id, instruction, "
                "and optional string-list tags. Do not repeat completed subtask IDs. "
                "Use the current image and failure receipt when replanning."},
                {"role": "user", "content": content}]})
        raw = json.loads(response["choices"][0]["message"]["content"])
        if not isinstance(raw, list):
            raise ValueError("Planner must return a JSON list")
        plan = []
        for item in raw:
            if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                    or not isinstance(item.get("instruction"), str)
                    or not isinstance(item.get("tags", []), list)
                    or any(not isinstance(tag, str) for tag in item.get("tags", []))):
                raise ValueError("Planner returned an invalid subtask")
            plan.append(Subtask(item["id"], item["instruction"], tuple(item.get("tags", []))))
        return plan


class HarnessChatPolicy:
    """Pass current and recalled images to a model; validate its one-step decision."""

    def __init__(self, endpoint: str, model: str, *, allowed_tools: tuple[str, ...],
                 api_key_env: str = "MODEL_API_KEY", transport=None):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.allowed_tools = allowed_tools
        self.api_key_env = api_key_env
        self.transport = transport or self._post

    def _post(self, payload: dict) -> dict:
        return _post_json(self.endpoint, self.api_key_env, payload)

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
        response = self.transport({"model": self.model, "temperature": 0,
            "messages": [{"role": "system", "content":
                "Return exactly one JSON object. Choices: "
                "{\"kind\":\"tool\",\"tool\":name,\"args\":{...}}, "
                "{\"kind\":\"recall\",\"recall_subtask_id\":id}, or "
                "{\"kind\":\"replan\"}. Use only allowed tools. "
                "A historical image is a past starting view, not current state. "
                "Trust environment receipts for completion."},
                {"role": "user", "content": content}]})
        raw = response["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or parsed.get("kind") not in ("tool", "recall", "replan"):
            raise ValueError("Model returned an invalid decision")
        kind = parsed["kind"]
        if kind == "tool":
            if parsed.get("tool") not in self.allowed_tools or not isinstance(parsed.get("args"), dict):
                raise ValueError("Model requested an invalid tool call")
            return Decision(kind, tool=parsed["tool"], args=parsed["args"])
        if kind == "recall":
            subtask_id = parsed.get("recall_subtask_id", "")
            if not isinstance(subtask_id, str) or not subtask_id:
                raise ValueError("Recall requires a subtask ID")
            return Decision(kind, recall_subtask_id=subtask_id)
        return Decision(kind)
