"""Frozen Qwen/Astra condition builder for RoboDojo native episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .advisory import AdviceLimits, AdvisedPlanner, AdvisedPolicy, BoundedAdvisor, JsonEndpoint
from .harness import AgenticHarness, HarnessConfig
from .harness_model import HarnessChatPlanner, HarnessChatPolicy
from .image_memory import SubtaskImageMemory


CONDITIONS = ("q_only", "astra_reference", "hybrid_plan", "hybrid_diagnose",
              "hybrid_both", "hybrid_both_memory")


def load_experiment_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be an object")
    for role in ("qwen", "astra"):
        spec = config.get(role)
        if (not isinstance(spec, dict) or
                any(not isinstance(spec.get(key), str) or not spec[key]
                    for key in ("base_url", "model"))):
            raise ValueError(f"{role} requires base_url and model")
        if spec.get("mode", "chat_completions") not in ("chat_completions", "responses"):
            raise ValueError(f"Invalid {role} API mode")
        for price_key in ("input_usd_per_million_tokens", "output_usd_per_million_tokens"):
            price = spec.get(price_key)
            if price is not None and (type(price) not in (int, float) or price < 0):
                raise ValueError(f"{role}.{price_key} must be a nonnegative number")
    if not isinstance(config.get("harness", {}), dict):
        raise ValueError("harness must be an object")
    if not isinstance(config.get("advisor_limits", {}), dict):
        raise ValueError("advisor_limits must be an object")
    HarnessConfig(**config.get("harness", {}))
    AdviceLimits(**config.get("advisor_limits", {}))
    return config


def build_agent_pair(config: dict, condition: str, *,
                     allowed_tools: tuple[str, ...], tool_schemas: dict) -> tuple[object, object]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose from {CONDITIONS}")
    if not allowed_tools or set(tool_schemas) != set(allowed_tools):
        raise ValueError("Robot tool schemas must match the exact allowed tool set")
    qwen = config["qwen"]
    qwen_url = qwen["base_url"]
    qwen_model = qwen["model"]
    qwen_key = qwen.get("api_key_env", "MODEL_API_KEY")
    qwen_output = qwen.get("max_output_tokens", 8192)
    qwen_timeout = qwen.get("timeout_seconds", 600)
    qwen_transport = None
    if qwen.get("mode", "chat_completions") == "responses":
        qwen_transport = JsonEndpoint(qwen_url, qwen_model, mode="responses",
            api_key_env=qwen_key, timeout_seconds=qwen_timeout,
            max_output_tokens=qwen_output).as_chat_transport()
    planner = HarnessChatPlanner(qwen_url, qwen_model, allowed_tools=allowed_tools,
                                 api_key_env=qwen_key, tool_schemas=tool_schemas,
                                 max_output_tokens=qwen_output,
                                 timeout_seconds=qwen_timeout, transport=qwen_transport)
    decisions = ("tool", "replan", "advance")
    if condition in ("hybrid_diagnose", "hybrid_both", "hybrid_both_memory"):
        decisions += ("consult",)
    if condition == "hybrid_both_memory":
        decisions += ("recall",)
    actor = HarnessChatPolicy(qwen_url, qwen_model, allowed_tools=allowed_tools,
                              api_key_env=qwen_key, tool_schemas=tool_schemas,
                              allowed_decisions=decisions, max_output_tokens=qwen_output,
                              timeout_seconds=qwen_timeout, transport=qwen_transport)
    if condition == "q_only":
        return planner, actor

    astra = config["astra"]
    astra_output = (astra.get("reference_max_output_tokens", 8192)
                    if condition == "astra_reference" else astra.get("advisor_max_output_tokens", 1200))
    endpoint = JsonEndpoint(astra["base_url"], astra["model"],
                            mode=astra.get("mode", "responses"),
                            api_key_env=astra.get("api_key_env", "ASTRA_API_KEY"),
                            timeout_seconds=astra.get("timeout_seconds", 600),
                            max_output_tokens=astra_output)
    if condition == "astra_reference":
        transport = endpoint.as_chat_transport()
        return (HarnessChatPlanner(astra["base_url"], astra["model"],
                    allowed_tools=allowed_tools, tool_schemas=tool_schemas,
                    transport=transport, max_output_tokens=astra_output),
                HarnessChatPolicy(astra["base_url"], astra["model"],
                    allowed_tools=allowed_tools, tool_schemas=tool_schemas,
                    allowed_decisions=("tool", "replan", "advance"), transport=transport,
                    max_output_tokens=astra_output))

    limits = AdviceLimits(**config.get("advisor_limits", {}))
    if condition == "hybrid_plan":
        limits = AdviceLimits(limits.max_total_calls, limits.max_plan_calls, 0)
    elif condition == "hybrid_diagnose":
        limits = AdviceLimits(limits.max_total_calls, 0, limits.max_diagnosis_calls)
    advisor = BoundedAdvisor(endpoint, limits=limits)
    if condition in ("hybrid_plan", "hybrid_both", "hybrid_both_memory"):
        planner = AdvisedPlanner(planner, advisor)
    if condition in ("hybrid_diagnose", "hybrid_both", "hybrid_both_memory"):
        actor = AdvisedPolicy(actor, advisor)
    return planner, actor


def build_harness(env_adapter, task: dict, output_dir: str | Path) -> AgenticHarness:
    """Official RoboDojo entry: construct exactly one arm for this native episode.

    The surrounding XPolicyLab evaluator owns task reset, terminal checks and
    score files. This factory only selects models, budgets and the agent loop.
    """
    config_path = os.environ.get("EMBODIED_EXPERIMENT_CONFIG")
    if not config_path:
        raise ValueError("Set EMBODIED_EXPERIMENT_CONFIG to a frozen JSON config")
    condition = os.environ.get("EMBODIED_CONDITION", "q_only")
    config = load_experiment_config(config_path)
    tool_schemas = env_adapter.tool_schemas()
    allowed_tools = tuple(tool_schemas)
    planner, actor = build_agent_pair(config, condition, allowed_tools=allowed_tools,
                                      tool_schemas=tool_schemas)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"condition": condition, "task_id": task["task_id"],
                "episode_id": task["episode_id"], "models": {
                    role: {"model": config[role]["model"], "base_url": config[role]["base_url"],
                           "mode": config[role].get("mode", "chat_completions"),
                           "input_usd_per_million_tokens": config[role].get("input_usd_per_million_tokens"),
                           "output_usd_per_million_tokens": config[role].get("output_usd_per_million_tokens")}
                    for role in ("qwen", "astra")},
                "advisor_limits": config.get("advisor_limits", {}),
                "allowed_tools": allowed_tools}
    (run_dir / "experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    settings = dict(config.get("harness", {}))
    settings.update(allowed_tools=allowed_tools,
                    enable_image_memory=condition == "hybrid_both_memory")
    harness_config = HarnessConfig(**settings)
    return AgenticHarness(environment=env_adapter, planner=planner, policy=actor,
                          memory=SubtaskImageMemory(run_dir / "memory"),
                          config=harness_config, trace_path=run_dir / "trace.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the frozen RoboDojo mixed-agent config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    print(json.dumps({"conditions": [args.condition] if args.condition else CONDITIONS,
                      "models": {role: config[role]["model"] for role in ("qwen", "astra")},
                      "harness": config.get("harness", {}),
                      "advisor_limits": config.get("advisor_limits", {})},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
