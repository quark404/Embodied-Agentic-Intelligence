"""Join harness traces to RoboDojo's native _result.json, then compare pairs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .schedule import native_result_dir


def collect_results(schedule_path: str | Path, robodojo_root: str | Path) -> dict:
    schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("Schedule must be a nonempty list")
    root = Path(robodojo_root)
    rows = []
    runs = []
    for item in schedule:
        native_dir = native_result_dir(root, item)
        native_path = native_dir / "_result.json"
        if not native_path.is_file():
            runs.append({"condition": item["condition"], "task_id": item["task_id"],
                         "seed": item["seed"], "status": "native_result_missing",
                         "path": str(native_path)})
            continue
        native = json.loads(native_path.read_text(encoding="utf-8"))
        details = native.get("details", {})
        if not isinstance(details, dict):
            raise ValueError(f"Invalid native result details: {native_path}")
        complete = len(details) >= item["eval_num"]
        runs.append({"condition": item["condition"], "task_id": item["task_id"],
                     "seed": item["seed"], "status": "scored" if complete else "incomplete",
                     "expected_episodes": item["eval_num"], "scored_episodes": len(details),
                     "path": str(native_path)})
        episode_root = native_dir / "embodied_agentic"
        summaries = {}
        for path in episode_root.glob("*/summary.json"):
            summary = json.loads(path.read_text(encoding="utf-8"))
            index = summary.get("native_result_index")
            if index is not None:
                if str(index) in summaries:
                    raise ValueError(f"Duplicate harness summary for native result {index}")
                summaries[str(index)] = (summary, path)
        for index, detail in details.items():
            if (not isinstance(detail, dict) or not isinstance(detail.get("success"), bool)
                    or not isinstance(detail.get("score"), (int, float))
                    or not math.isfinite(detail["score"])
                    or type(detail.get("layout_id")) is not int):
                raise ValueError(f"Invalid native episode detail {index}: {native_path}")
            summary_item = summaries.get(str(index))
            trace, summary_path = summary_item if summary_item is not None else (None, None)
            model_cost = None
            if trace is not None and isinstance(trace.get("model_calls"), dict):
                # A missing price manifest leaves monetary cost unknown, not zero.
                manifest_path = summary_path.parent / "experiment.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    prices = {spec["model"]: spec for spec in manifest.get("models", {}).values()}
                    costs = []
                    for name, usage in trace["model_calls"].items():
                        model = name.rsplit(":", 1)[0]
                        spec = prices.get(model, {})
                        input_price = spec.get("input_usd_per_million_tokens")
                        output_price = spec.get("output_usd_per_million_tokens")
                        if input_price is None or output_price is None:
                            costs = []
                            break
                        costs.append((usage.get("input_tokens", 0) * input_price +
                                      usage.get("output_tokens", 0) * output_price) / 1_000_000)
                    if costs or not trace["model_calls"]:
                        model_cost = round(sum(costs), 8)
            error = trace.get("error") if trace else None
            if not complete:
                exclusion = "incomplete_native_run"
            elif trace is None:
                exclusion = "harness_trace_missing"
            elif trace.get("status") == "error" and not str(error).startswith(
                    ("ModelOutputError:", "AgentDecisionError:")):
                exclusion = "service_or_runtime_error"
            else:
                exclusion = None
            rows.append({"condition": item["condition"], "task_id": item["task_id"],
                         "seed": item["seed"], "layout_id": detail["layout_id"],
                         "success": detail["success"], "score": detail["score"],
                         "harness_status": trace.get("status") if trace else None,
                         "harness_error": error,
                         "model_calls": trace.get("model_calls") if trace else None,
                         "estimated_model_cost_usd": model_cost,
                         "rejected_actions": trace.get("rejected_actions") if trace else None,
                         "executed_physical_steps": trace.get("executed_physical_steps") if trace else None,
                         "elapsed_seconds": trace.get("elapsed_seconds") if trace else None,
                         "trace_missing": trace is None,
                         "valid_for_comparison": exclusion is None,
                         "exclusion_reason": exclusion,
                         "native_result": str(native_path)})
    by_condition = defaultdict(list)
    raw_by_condition = defaultdict(list)
    exclusions = defaultdict(lambda: defaultdict(int))
    for row in rows:
        raw_by_condition[row["condition"]].append(row)
        if row["valid_for_comparison"]:
            by_condition[row["condition"]].append(row)
        else:
            exclusions[row["condition"]][row["exclusion_reason"]] += 1
    def summarize(groups):
        summarized = {}
        for name, group in groups.items():
            if not group:
                continue
            breakdown = defaultdict(lambda: {"calls": 0, "errors": 0,
                                              "input_tokens": 0, "output_tokens": 0,
                                              "latency_seconds": 0.0})
            for row in group:
                for model_role, slot in (row.get("model_calls") or {}).items():
                    for metric in breakdown[model_role]:
                        breakdown[model_role][metric] += slot.get(metric, 0) or 0
            summarized[name] = {
                "episodes": len(group),
                "success_rate": sum(row["success"] for row in group) / len(group),
                "mean_score": sum(row["score"] for row in group) / len(group),
                "model_call_breakdown": dict(breakdown),
                "estimated_model_cost_usd": (
                    round(sum(row["estimated_model_cost_usd"] for row in group), 8)
                    if all(row["estimated_model_cost_usd"] is not None for row in group)
                    else None),
                "executed_physical_steps": sum(row["executed_physical_steps"] or 0
                                               for row in group),
                "rejected_actions": sum(row["rejected_actions"] or 0 for row in group),
            }
        return summarized
    condition_summary = summarize(by_condition)
    indexed = defaultdict(dict)
    for row in rows:
        if not row["valid_for_comparison"]:
            continue
        key = (row["task_id"], row["seed"], row["layout_id"])
        if row["condition"] in indexed[key]:
            raise ValueError(f"Duplicate scored episode for {key} and {row['condition']}")
        indexed[key][row["condition"]] = row
    pairs = {}
    for treatment in ("hybrid_plan", "hybrid_diagnose", "hybrid_both",
                      "hybrid_both_memory", "astra_reference"):
        control = "hybrid_both" if treatment == "hybrid_both_memory" else "q_only"
        matched = [(arm[control], arm[treatment]) for arm in indexed.values()
                   if control in arm and treatment in arm]
        if matched:
            pairs[f"{control}_vs_{treatment}"] = {
                "paired_episodes": len(matched),
                "success_rate_difference": sum(b["success"] - a["success"]
                                               for a, b in matched) / len(matched),
                "score_difference": sum(b["score"] - a["score"]
                                        for a, b in matched) / len(matched),
                "treatment_wins": sum(not a["success"] and b["success"] for a, b in matched),
                "control_wins": sum(a["success"] and not b["success"] for a, b in matched),
            }
    return {"runs": runs, "episodes": rows, "conditions": condition_summary,
            "raw_native_conditions": summarize(raw_by_condition),
            "exclusions": {name: dict(reasons) for name, reasons in exclusions.items()},
            "paired_comparisons": pairs,
            "note": "Primary comparisons exclude missing traces, incomplete native runs and service/runtime errors; native-only scores remain separately visible. Model output and decision protocol errors count as model failures."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize native RoboDojo mixed-agent results")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect_results(args.schedule, args.robodojo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scored {len(result['episodes'])} episodes; wrote {args.output}")


if __name__ == "__main__":
    main()
