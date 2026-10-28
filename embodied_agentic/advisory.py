"""Bounded, auditable frontier-model advice; advice cannot drive the robot."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, replace
from urllib.request import Request, urlopen

from .harness import AgentContext, Decision, Subtask


def _image_part(data: bytes, *, responses: bool) -> dict:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        raise ValueError("Advisor accepts only PNG/JPEG camera images")
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return {"type": "input_image", "image_url": url} if responses else {
        "type": "image_url", "image_url": {"url": url}}


def _response_text(response: dict, mode: str) -> str:
    if mode == "chat_completions":
        return response["choices"][0]["message"]["content"]
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts = [part["text"] for item in response.get("output", [])
             for part in item.get("content", [])
             if part.get("type") == "output_text" and isinstance(part.get("text"), str)]
    if not parts:
        raise ValueError("Responses API returned no output_text")
    return "".join(parts)


class JsonEndpoint:
    """Small multimodal JSON client for Chat Completions or Responses APIs."""

    def __init__(self, base_url: str, model: str, *, mode: str = "responses",
                 api_key_env: str = "ASTRA_API_KEY", timeout_seconds: float = 600,
                 max_output_tokens: int = 1200, transport=None):
        if mode not in ("responses", "chat_completions"):
            raise ValueError("Unsupported model API mode")
        if not base_url or not model or timeout_seconds <= 0 or max_output_tokens < 1:
            raise ValueError("A model endpoint, model and positive timeout are required")
        self.endpoint = base_url.rstrip("/") + ("/responses" if mode == "responses" else "/chat/completions")
        self.model = model
        self.mode = mode
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.transport = transport or self._post

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response)

    def ask(self, system: str, prompt: dict, images: dict[str, bytes]) -> tuple[dict, dict]:
        responses = self.mode == "responses"
        text_type = "input_text" if responses else "text"
        content = [{"type": text_type, "text": json.dumps(prompt, ensure_ascii=False)}]
        for camera, data in images.items():
            content.append({"type": text_type, "text": f"Current public RGB observation, camera={camera}"})
            content.append(_image_part(data, responses=responses))
        payload = ({"model": self.model, "max_output_tokens": self.max_output_tokens, "input": [
            {"role": "system", "content": system}, {"role": "user", "content": content}]}
            if responses else {"model": self.model, "temperature": 0,
                "max_tokens": self.max_output_tokens, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": content}]})
        response = self.transport(payload)
        if not isinstance(response, dict):
            raise ValueError("Model endpoint did not return an object")
        def reject_constant(value):
            raise ValueError(f"Non-JSON numeric constant {value} is forbidden")
        value = json.loads(_response_text(response, self.mode), parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("Advisor must return one JSON object")
        return value, response.get("usage", {})

    def as_chat_transport(self):
        """Adapt a Responses-only provider to HarnessChatPlanner/Policy."""
        if self.mode == "chat_completions":
            return self.transport

        def transport(payload: dict) -> dict:
            messages = []
            for item in payload["messages"]:
                source = item["content"]
                if isinstance(source, str):
                    content = source
                else:
                    content = []
                    for part in source:
                        if part["type"] == "text":
                            content.append({"type": "input_text", "text": part["text"]})
                        elif part["type"] == "image_url":
                            content.append({"type": "input_image",
                                            "image_url": part["image_url"]["url"]})
                        else:
                            raise ValueError("Unsupported chat content for Responses API")
                messages.append({"role": item["role"], "content": content})
            response = self.transport({"model": payload["model"], "input": messages,
                                       "max_output_tokens": self.max_output_tokens})
            return {"choices": [{"message": {"content": _response_text(response, "responses")}}],
                    "usage": response.get("usage", {})}

        return transport


@dataclass(frozen=True)
class AdviceLimits:
    max_total_calls: int = 2
    max_plan_calls: int = 1
    max_diagnosis_calls: int = 1

    def __post_init__(self):
        if min(self.max_total_calls, self.max_plan_calls, self.max_diagnosis_calls) < 0:
            raise ValueError("Advice call limits cannot be negative")


class BoundedAdvisor:
    """Astra is callable only for structured plans and diagnoses, never actions."""

    def __init__(self, endpoint: JsonEndpoint, *, limits: AdviceLimits = AdviceLimits()):
        self.endpoint = endpoint
        self.limits = limits
        self.calls = {"plan": 0, "diagnosis": 0}
        self._events: list[dict] = []

    def drain_events(self) -> list[dict]:
        events, self._events = self._events, []
        return events

    def _request(self, kind: str, prompt: dict, observation: dict) -> dict | None:
        limit = self.limits.max_plan_calls if kind == "plan" else self.limits.max_diagnosis_calls
        if self.calls[kind] >= limit or sum(self.calls.values()) >= self.limits.max_total_calls:
            return None
        self.calls[kind] += 1  # A failed request also spends the call budget.
        system = ("Return only JSON: {\"subtasks\":[{\"id\":string,\"instruction\":string,"
                  "\"tags\":[string]}]}. Give at most 10 ordered remaining task goals. "
                  "Do not emit robot actions, tool calls, code or success verdicts."
                  if kind == "plan" else
                  "Return only JSON: {\"diagnosis\":string,\"next_step_hint\":string}. "
                  "Diagnose the last failed or stalled execution using public observations only. "
                  "Do not emit robot actions, tool calls, code or success verdicts.")
        started = time.monotonic()
        try:
            answer, usage = self.endpoint.ask(system, prompt, observation["images"])
            if kind == "plan":
                rows = answer.get("subtasks")
                if (set(answer) != {"subtasks"} or not isinstance(rows, list) or
                        not 1 <= len(rows) <= 10 or
                        any(not isinstance(row, dict) or
                            not set(row).issubset({"id", "instruction", "tags"}) or
                            not isinstance(row.get("id"), str) or not 1 <= len(row["id"]) <= 64 or
                            not isinstance(row.get("instruction"), str) or
                            not 1 <= len(row["instruction"]) <= 500 or
                            not isinstance(row.get("tags", []), list) or
                            len(row.get("tags", [])) > 10 or
                            any(not isinstance(tag, str) or len(tag) > 64
                                for tag in row.get("tags", []))
                            for row in rows) or len({row["id"] for row in rows}) != len(rows)):
                    raise ValueError("Advisor returned an invalid task plan")
            else:
                if (set(answer) != {"diagnosis", "next_step_hint"} or
                        any(not isinstance(value, str) or not value or len(value) > 1000
                            for value in answer.values())):
                    raise ValueError("Advisor returned an invalid diagnosis")
            self._events.append({"event": "advisor_call", "kind": kind,
                                 "model": self.endpoint.model, "ok": True,
                                 "latency_seconds": round(time.monotonic() - started, 3),
                                 "usage": usage, "advice": answer})
            return answer
        except Exception as exc:
            self._events.append({"event": "advisor_call", "kind": kind,
                                 "model": self.endpoint.model, "ok": False,
                                 "latency_seconds": round(time.monotonic() - started, 3),
                                 "error": f"{type(exc).__name__}: {exc}"})
            return None

    def plan(self, task: dict, observation: dict, completed: tuple[str, ...],
             failure: dict | None) -> list[Subtask] | None:
        prompt = {"goal": task.get("goal", ""), "task_id": task["task_id"],
                  "completed_subtasks": completed, "last_failure_receipt": failure,
                  "public_state": {k: v for k, v in observation.items() if k != "images"}}
        answer = self._request("plan", prompt, observation)
        if answer is None:
            return None
        rows = [row for row in answer["subtasks"] if row["id"] not in completed]
        if not rows:
            return None
        return [Subtask(row["id"], row["instruction"], tuple(row.get("tags", []))) for row in rows]

    def diagnose(self, context: AgentContext, receipt: dict, *, trigger: str) -> dict | None:
        prompt = {"goal": context.task.get("goal", ""),
                  "subtask": {"id": context.subtask.id,
                              "instruction": context.subtask.instruction},
                  "trigger": trigger, "last_execution_receipt": receipt,
                  "public_state": {k: v for k, v in context.observation.items() if k != "images"}}
        return self._request("diagnosis", prompt, context.observation)


class AdvisedPlanner:
    """Use one bounded Astra planning suggestion; Qwen handles any fallback plan."""

    def __init__(self, qwen_planner, advisor: BoundedAdvisor):
        self.qwen_planner = qwen_planner
        self.advisor = advisor

    def plan(self, task: dict, observation: dict, completed: tuple[str, ...],
             failure: dict | None) -> list[Subtask]:
        suggestion = self.advisor.plan(task, observation, completed, failure)
        return suggestion if suggestion is not None else self.qwen_planner.plan(
            task, observation, completed, failure)

    def drain_events(self) -> list[dict]:
        events = self.advisor.drain_events()
        drain = getattr(self.qwen_planner, "drain_events", None)
        return events + (drain() if drain else [])


class AdvisedPolicy:
    """On a failed robot receipt, ask Astra for advice then let Qwen decide."""

    def __init__(self, qwen_actor, advisor: BoundedAdvisor):
        self.qwen_actor = qwen_actor
        self.advisor = advisor
        self._diagnosed_event_count: set[int] = set()

    def decide(self, context: AgentContext) -> Decision:
        latest = next(((index, event) for index, event in reversed(list(enumerate(context.history)))
                       if event.get("event") == "tool"), None)
        if latest is not None:
            index, event = latest
            if index not in self._diagnosed_event_count and event.get("receipt", {}).get("ok") is False:
                self._diagnosed_event_count.add(index)
                answer = self.advisor.diagnose(context, event["receipt"], trigger="rejected_action")
                if answer is not None:
                    context = replace(context, history=context.history + ({
                        "event": "advisor_diagnosis", "model": self.advisor.endpoint.model,
                        **answer},))
        return self.qwen_actor.decide(context)

    def consult(self, context: AgentContext) -> dict | None:
        latest = next((event for event in reversed(context.history)
                       if event.get("event") == "tool"), None)
        if latest is None:
            raise ValueError("Diagnosis requires a prior robot action")
        return self.advisor.diagnose(context, latest["receipt"], trigger="actor_requested")

    def drain_events(self) -> list[dict]:
        events = self.advisor.drain_events()
        drain = getattr(self.qwen_actor, "drain_events", None)
        return events + (drain() if drain else [])
