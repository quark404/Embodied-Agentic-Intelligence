import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agentic.reporting import collect_results
from embodied_agentic.schedule import load_task_manifest, main, make_schedule, native_result_dir


class ScheduleReportingTests(unittest.TestCase):
    def test_resume_uses_frozen_hash_and_skips_completed_native_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "scripts" / "robodojo.sh").write_text("#!/bin/sh\n")
            policy = root / "XPolicyLab" / "policy" / "Embodied_Agentic"
            policy.mkdir(parents=True)
            (policy / "eval.sh").write_text("#!/bin/sh\n")
            config = root / "config.json"
            config.write_text(json.dumps({"qwen": {"base_url": "http://q/v1", "model": "q"},
                "astra": {"base_url": "http://a/v1", "model": "a"}}))
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps({"tasks": [{"task_id": "general_pickup",
                                                     "seed": 0, "eval_num": 1}]}))
            output = root / "schedule.json"
            args = ["schedule", "--robodojo-root", str(root), "--config", str(config),
                    "--tasks", str(tasks), "--policy-env", "python", "--sim-env", "python",
                    "--run-id", "pilot", "--action-mode", "joint", "--conditions", "q_only",
                    "--output", str(output)]
            with patch("sys.argv", args):
                main()
            row = json.loads(output.read_text())[0]
            self.assertEqual(len(row["config_sha256"]), 64)
            native = native_result_dir(root, row)
            native.mkdir(parents=True)
            (native / "_result.json").write_text(json.dumps({"details": {
                "0": {"layout_id": 0, "success": False, "score": 0.0}}}))
            with patch("sys.argv", args + ["--resume", "--execute"]), \
                    patch("embodied_agentic.schedule.subprocess.run") as launched:
                main()
            launched.assert_not_called()
            config.write_text(config.read_text() + "\n")
            with patch("sys.argv", args + ["--resume"]), \
                    self.assertRaisesRegex(ValueError, "differs"):
                main()

    def test_manifest_rejects_duplicate_pairs(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tasks.json"
            path.write_text(json.dumps({"tasks": [
                {"task_id": "general_pickup", "seed": 0, "eval_num": 2},
                {"task_id": "general_pickup", "seed": 0, "eval_num": 2}]}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_task_manifest(path)

    def test_native_result_is_authoritative_and_missing_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "scripts" / "robodojo.sh").write_text("#!/bin/sh\n")
            policy = root / "XPolicyLab" / "policy" / "Embodied_Agentic"
            policy.mkdir(parents=True)
            (policy / "eval.sh").write_text("#!/bin/sh\n")
            config = root / "config.json"
            config.write_text(json.dumps({"qwen": {"base_url": "http://q/v1", "model": "qwen"},
                "astra": {"base_url": "http://a/v1", "model": "astra"}}))
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps({"tasks": [{"task_id": "general_pickup",
                                                     "seed": 0, "eval_num": 1}]}))
            schedule = make_schedule(robodojo_root=root, experiment_config=config,
                task_manifest=tasks, policy_env="policy-python", sim_env="sim-python",
                run_id="paired-test", action_mode="joint",
                conditions=("q_only", "hybrid_plan", "hybrid_diagnose"))
            self.assertEqual(len(schedule), 3)
            for condition, success in (("q_only", False), ("hybrid_plan", True)):
                item = next(row for row in schedule if row["condition"] == condition)
                native = (root / "eval_result" / "RoboDojo" / "general_pickup" /
                          "Embodied_Agentic" / "arx_x5" /
                          f"0_ckpt_name={condition},action_type=joint" / item["native_run_id"])
                native.mkdir(parents=True)
                (native / "_result.json").write_text(json.dumps({"details": {
                    "0": {"layout_id": 0, "success": success, "score": float(success)}}}))
                summary = native / "embodied_agentic" / "episode" / "summary.json"
                summary.parent.mkdir(parents=True)
                summary.write_text(json.dumps({"native_result_index": 0, "status": "success",
                                               "native_success": not success,
                                               "model_calls": ({"qwen:actor": {"calls": 1,
                                                   "input_tokens": 1_000_000,
                                                   "output_tokens": 500_000}}
                                                   if condition == "q_only" else {})}))
                if condition == "q_only":
                    (summary.parent / "experiment.json").write_text(json.dumps({"models": {
                        "qwen": {"model": "qwen", "input_usd_per_million_tokens": 1,
                                 "output_usd_per_million_tokens": 2}}}))
            schedule_path = root / "schedule.json"
            schedule_path.write_text(json.dumps(schedule))
            report = collect_results(schedule_path, root)
            self.assertEqual(len(report["episodes"]), 2)
            self.assertEqual(report["conditions"]["q_only"]["success_rate"], 0)
            self.assertEqual(report["conditions"]["hybrid_plan"]["success_rate"], 1)
            self.assertEqual(report["conditions"]["q_only"]["estimated_model_cost_usd"], 2)
            self.assertEqual(report["paired_comparisons"]["q_only_vs_hybrid_plan"]
                             ["treatment_wins"], 1)
            missing = report["paired_comparisons"]["q_only_vs_hybrid_diagnose"]
            self.assertEqual(missing["status"], "no_valid_pairs")
            self.assertEqual(missing["unmatched_control_episodes"], 1)
            self.assertIsNone(missing["success_rate_difference"])
            self.assertNotIn("hybrid_plan_vs_hybrid_both", report["paired_comparisons"])
            self.assertEqual(next(run for run in report["runs"]
                                  if run["condition"] == "hybrid_diagnose")["status"],
                             "native_result_missing")

    def test_infrastructure_error_is_not_a_robot_failure_in_primary_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schedule = [{"condition": "q_only", "task_id": "general_pickup", "seed": 0,
                         "eval_num": 1, "native_run_id": "pilot_q", "action_mode": "joint"}]
            native = (root / "eval_result" / "RoboDojo" / "general_pickup" /
                      "Embodied_Agentic" / "arx_x5" /
                      "0_ckpt_name=q_only,action_type=joint" / "pilot_q")
            native.mkdir(parents=True)
            (native / "_result.json").write_text(json.dumps({"details": {
                "0": {"layout_id": 0, "success": False, "score": 0.2}}}))
            summary = native / "embodied_agentic" / "episode" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({"native_result_index": 0, "status": "error",
                                           "error": "TimeoutError: model service unavailable"}))
            path = root / "schedule.json"
            path.write_text(json.dumps(schedule))
            report = collect_results(path, root)
            self.assertNotIn("q_only", report["conditions"])
            self.assertEqual(report["raw_native_conditions"]["q_only"]["episodes"], 1)
            self.assertEqual(report["episodes"][0]["exclusion_reason"],
                             "service_or_runtime_error")

    def test_planning_plus_diagnosis_comparison_excludes_service_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schedule = []
            for condition in ("hybrid_plan", "hybrid_both"):
                item = {"condition": condition, "task_id": "general_pickup", "seed": 0,
                        "eval_num": 2, "native_run_id": condition, "action_mode": "joint"}
                schedule.append(item)
                native = native_result_dir(root, item)
                native.mkdir(parents=True)
                success = condition == "hybrid_both"
                (native / "_result.json").write_text(json.dumps({"details": {
                    str(index): {"layout_id": index, "success": success, "score": float(success)}
                    for index in range(2)}}))
                for index in range(2):
                    summary = native / "embodied_agentic" / str(index) / "summary.json"
                    summary.parent.mkdir(parents=True)
                    trace = {"native_result_index": index, "status": "success"}
                    if condition == "hybrid_both" and index == 1:
                        trace.update(status="error", error="TimeoutError: service unavailable")
                    summary.write_text(json.dumps(trace))
            path = root / "schedule.json"
            path.write_text(json.dumps(schedule))
            report = collect_results(path, root)
            paired = report["paired_comparisons"]["hybrid_plan_vs_hybrid_both"]
            self.assertEqual(paired["paired_episodes"], 1)
            self.assertEqual(paired["valid_control_episodes"], 2)
            self.assertEqual(paired["valid_treatment_episodes"], 1)
            self.assertEqual(paired["unmatched_control_episodes"], 1)
            self.assertEqual(paired["treatment_wins"], 1)
            self.assertEqual(report["exclusions"]["hybrid_both"]["service_or_runtime_error"], 1)

    def test_missing_token_usage_is_not_priced_as_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            item = {"condition": "q_only", "task_id": "general_pickup", "seed": 0,
                    "eval_num": 1, "native_run_id": "cost-check", "action_mode": "joint"}
            path = root / "schedule.json"
            path.write_text(json.dumps([item]))
            native = native_result_dir(root, item)
            native.mkdir(parents=True)
            (native / "_result.json").write_text(json.dumps({"details": {
                "0": {"layout_id": 0, "success": True, "score": 1.0}}}))
            summary = native / "embodied_agentic" / "episode" / "summary.json"
            summary.parent.mkdir(parents=True)
            price_path = summary.parent / "experiment.json"
            cases = [
                ({"output_tokens": 0}, 1, None),
                ({"input_tokens": 0}, 1, None),
                ({"input_tokens": True, "output_tokens": 0}, 1, None),
                ({"input_tokens": -1, "output_tokens": 0}, 1, None),
                ({"input_tokens": 0, "output_tokens": 0}, -1, None),
                ({"input_tokens": 0, "output_tokens": 0}, None, None),
                ({"input_tokens": 0, "output_tokens": 0}, 1, 0),
            ]
            for usage, input_price, expected in cases:
                with self.subTest(usage=usage, input_price=input_price):
                    summary.write_text(json.dumps({"native_result_index": 0, "status": "success",
                        "model_calls": {"qwen:actor": usage}}))
                    price_path.write_text(json.dumps({"models": {"qwen": {"model": "qwen",
                        "input_usd_per_million_tokens": input_price,
                        "output_usd_per_million_tokens": 1}}}))
                    report = collect_results(path, root)
                    self.assertEqual(report["episodes"][0]["estimated_model_cost_usd"], expected)
                    self.assertEqual(report["conditions"]["q_only"]["estimated_model_cost_usd"], expected)
                    self.assertEqual(report["conditions"]["q_only"]["success_rate"], 1)


if __name__ == "__main__":
    unittest.main()
