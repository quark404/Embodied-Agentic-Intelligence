"""Live local-endpoint checks, not robot benchmark evaluation."""

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

from PIL import Image

from embodied_agentic.harness import AgentContext
from embodied_agentic.harness_model import HarnessChatPlanner, HarnessChatPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--output", type=Path, default=Path("runs/qwen35/smoke.json"))
    args = parser.parse_args()
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY", ""),
                                                  "localhost", "127.0.0.1"]))
    model = "Qwen/Qwen3.5-27B"
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "model": model,
              "base_url": args.base_url,
              "versions": {name: version(name) for name in ("vllm", "torch", "transformers")},
              "checks": [], "robot_benchmark": False}

    def request(payload):
        payload = {"model": model, "temperature": 0, "max_tokens": 256, **payload}
        headers = {"Content-Type": "application/json"}
        if os.environ.get("MODEL_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["MODEL_API_KEY"]
        started = time.monotonic()
        message = Request(args.base_url.rstrip("/") + "/chat/completions",
                          data=json.dumps(payload).encode(), headers=headers)
        with urlopen(message, timeout=180) as response:
            result = json.load(response)
        return result, round(time.monotonic() - started, 3)

    def record(name, response, elapsed):
        report["checks"].append({"name": name, "passed": True, "latency_seconds": elapsed,
                                  "usage": response.get("usage", {}),
                                  "message": response["choices"][0]["message"]})

    try:
        response, elapsed = request({"messages": [{"role": "user", "content":
            'Return exactly this JSON object, no markdown: {"ready":true}'}]})
        if json.loads(response["choices"][0]["message"]["content"]) != {"ready": True}:
            raise ValueError("Text JSON check failed")
        record("text_json", response, elapsed)

        images = {}
        content = [{"type": "text", "text":
            'Return JSON {"colors":[...]} listing the solid color of each image in order. '
            'Use lowercase English color names, without markdown.'}]
        for index, color in enumerate(("red", "green", "blue")):
            buffer = BytesIO()
            Image.new("RGB", (256, 256), color=color).save(buffer, format="PNG")
            images[f"camera_{index}"] = buffer.getvalue()
            content.append({"type": "image_url", "image_url": {"url":
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}})
        response, elapsed = request({"messages": [{"role": "user", "content": content}]})
        if json.loads(response["choices"][0]["message"]["content"]) != {
                "colors": ["red", "green", "blue"]}:
            raise ValueError("Three-image color check failed")
        record("three_images", response, elapsed)

        parameters = {"type": "object", "properties": {
            "camera": {"type": "string"},
            "color": {"type": "string", "enum": ["red", "green", "blue"]}},
            "required": ["camera", "color"], "additionalProperties": False}
        response, elapsed = request({"messages": [{"role": "user", "content":
            "Call report_color with camera camera_0 and color red. Do not answer in prose."}],
            "tools": [{"type": "function", "function": {"name": "report_color",
                "description": "Record the observed camera color.", "parameters": parameters}}],
            "tool_choice": "auto"})
        calls = response["choices"][0]["message"].get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "report_color" or json.loads(
                calls[0]["function"]["arguments"]) != {"camera": "camera_0", "color": "red"}:
            raise ValueError("Native function-call check failed")
        record("native_function_call", response, elapsed)

        task = {"task_id": "endpoint-smoke", "episode_id": "endpoint-smoke-1",
                "goal": "Report the color of camera_0 using report_color. Do not move anything."}
        observation = {"images": images}
        tools = {"report_color": parameters}
        planner = HarnessChatPlanner(args.base_url, model, allowed_tools=tuple(tools),
            tool_schemas=tools, max_output_tokens=512, timeout_seconds=180)
        started = time.monotonic()
        plan = planner.plan(task, observation, (), None)
        if not plan:
            raise ValueError("Harness planner returned an empty plan")
        policy = HarnessChatPolicy(args.base_url, model, allowed_tools=tuple(tools),
            tool_schemas=tools, allowed_decisions=("tool",), max_output_tokens=512,
            timeout_seconds=180)
        decision = policy.decide(AgentContext(task=task, subtask=plan[0],
            observation=observation, history=(), recalled_images=(), completed_subtasks=(),
            remaining_subtasks=tuple(plan)))
        if decision.kind != "tool" or decision.tool != "report_color" or decision.args != {
                "camera": "camera_0", "color": "red"}:
            raise ValueError(f"Harness actor returned an unexpected decision: {decision}")
        report["checks"].append({"name": "harness_planner_actor", "passed": True,
            "latency_seconds": round(time.monotonic() - started, 3),
            "plan": [asdict(subtask) for subtask in plan], "decision": asdict(decision),
            "model_calls": planner.drain_events() + policy.drain_events()})
        report["passed"] = True
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
