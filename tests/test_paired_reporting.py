import unittest

from embodied_agentic.reporting import summarize_paired_comparison


def episode(success, **metrics):
    return {"success": success, "score": float(success), **metrics}


class PairedReportingTests(unittest.TestCase):
    def report(self, indexed):
        return summarize_paired_comparison(indexed, "q_only", "hybrid_plan")

    def test_all_four_outcomes_and_conditional_denominators(self):
        indexed = {
            ("pickup", 0, 0): {"q_only": episode(False), "hybrid_plan": episode(True)},
            ("pickup", 0, 1): {"q_only": episode(False), "hybrid_plan": episode(True)},
            ("pickup", 0, 2): {"q_only": episode(True), "hybrid_plan": episode(False)},
            ("pickup", 0, 3): {"q_only": episode(True), "hybrid_plan": episode(True)},
            ("pickup", 0, 4): {"q_only": episode(False), "hybrid_plan": episode(False)},
        }
        report = self.report(indexed)
        self.assertEqual(report["status"], "paired")
        self.assertEqual(report["paired_episodes"], 5)
        self.assertEqual(report["treatment_wins"], 2)
        self.assertEqual(report["control_wins"], 1)
        self.assertEqual(report["both_succeeded"], 1)
        self.assertEqual(report["both_failed"], 1)
        self.assertEqual(report["control_failure_episodes"], 3)
        self.assertEqual(report["control_success_episodes"], 2)
        self.assertAlmostEqual(report["baseline_failure_rescue_rate"], 2 / 3)
        self.assertEqual(report["baseline_success_harm_rate"], 0.5)
        self.assertEqual(report["control_success_rate"], 0.4)
        self.assertEqual(report["treatment_success_rate"], 0.6)
        self.assertEqual(report["success_rate_difference"], 0.2)
        self.assertEqual(report["score_difference"], 0.2)

    def test_unmatched_episodes_do_not_change_pair_metrics_or_cost(self):
        indexed = {
            ("pickup", 0, 0): {
                "q_only": episode(False, estimated_model_cost_usd=2),
                "hybrid_plan": episode(True, estimated_model_cost_usd=3)},
            ("pickup", 1, 0): {"q_only": episode(True, estimated_model_cost_usd=100)},
            ("tower", 0, 0): {"hybrid_plan": episode(False, estimated_model_cost_usd=200)},
        }
        report = self.report(indexed)
        self.assertEqual(report["paired_episodes"], 1)
        self.assertEqual(report["success_rate_difference"], 1)
        for role in ("control", "treatment"):
            self.assertEqual(report[f"valid_{role}_episodes"], 2)
            self.assertEqual(report[f"unmatched_{role}_episodes"], 1)
            self.assertEqual(report[f"{role}_pair_coverage"], 0.5)
        cost = report["paired_resources"]["estimated_model_cost_usd"]
        self.assertEqual(cost["complete_pairs"], 1)
        self.assertEqual(cost["control_mean"], 2)
        self.assertEqual(cost["treatment_mean"], 3)
        self.assertEqual(cost["mean_difference"], 1)

    def test_no_pairs_and_zero_opportunities_are_unknown_not_zero(self):
        for indexed in ({}, {("pickup", 0, 0): {"q_only": episode(False)}}):
            with self.subTest(indexed=indexed):
                report = self.report(indexed)
                self.assertEqual(report["status"], "no_valid_pairs")
                self.assertEqual(report["paired_episodes"], 0)
                for metric in ("success_rate_difference", "score_difference",
                               "baseline_failure_rescue_rate", "baseline_success_harm_rate"):
                    self.assertIsNone(report[metric])
                for resource in report["paired_resources"].values():
                    self.assertEqual(resource["complete_pairs"], 0)
                    self.assertIsNone(resource["mean_difference"])
        successful = self.report({("pickup", 0, 0): {
            "q_only": episode(True), "hybrid_plan": episode(True)}})
        self.assertIsNone(successful["baseline_failure_rescue_rate"])
        self.assertEqual(successful["baseline_success_harm_rate"], 0)
        failed = self.report({("pickup", 0, 0): {
            "q_only": episode(False), "hybrid_plan": episode(False)}})
        self.assertEqual(failed["baseline_failure_rescue_rate"], 0)
        self.assertIsNone(failed["baseline_success_harm_rate"])

    def test_resources_use_complete_pairs_and_keep_zero_as_known(self):
        indexed = {
            ("pickup", 0, 0): {
                "q_only": episode(False, elapsed_seconds=10, executed_physical_steps=2),
                "hybrid_plan": episode(True, elapsed_seconds=6, executed_physical_steps=0)},
            ("pickup", 0, 1): {
                "q_only": episode(True, elapsed_seconds=1000),
                "hybrid_plan": episode(True, elapsed_seconds=None)},
        }
        resources = self.report(indexed)["paired_resources"]
        self.assertEqual(resources["elapsed_seconds"], {
            "complete_pairs": 1, "missing_or_invalid_pairs": 1,
            "control_mean": 10, "treatment_mean": 6, "mean_difference": -4})
        self.assertEqual(resources["executed_physical_steps"]["treatment_mean"], 0)
        self.assertEqual(resources["executed_physical_steps"]["mean_difference"], -2)
        self.assertIsNone(resources["estimated_model_cost_usd"]["control_mean"])

    def test_invalid_resource_values_do_not_become_measurements(self):
        for invalid in (None, True, "10", -1, float("nan"), float("inf")):
            for role in ("q_only", "hybrid_plan"):
                with self.subTest(invalid=invalid, role=role):
                    arms = {"q_only": episode(False, elapsed_seconds=10),
                            "hybrid_plan": episode(True, elapsed_seconds=20)}
                    arms[role]["elapsed_seconds"] = invalid
                    report = self.report({("pickup", 0, 0): arms})
                    timing = report["paired_resources"]["elapsed_seconds"]
                    self.assertEqual(timing["complete_pairs"], 0)
                    self.assertEqual(timing["missing_or_invalid_pairs"], 1)
                    self.assertIsNone(timing["mean_difference"])
                    self.assertEqual(report["paired_episodes"], 1)

    def test_repeated_seeds_do_not_inflate_layout_group_count(self):
        indexed = {(task, seed, 0): {"q_only": episode(False), "hybrid_plan": episode(True)}
                   for task in ("pickup", "tower") for seed in (0, 1)}
        report = self.report(indexed)
        self.assertEqual(report["paired_episodes"], 4)
        self.assertEqual(report["paired_tasks"], 2)
        self.assertEqual(report["paired_task_layout_groups"], 2)

    def test_score_difference_uses_native_partial_score(self):
        report = self.report({("pickup", 0, 0): {
            "q_only": episode(False, score=0.1),
            "hybrid_plan": episode(False, score=0.7)}})
        self.assertEqual(report["success_rate_difference"], 0)
        self.assertAlmostEqual(report["score_difference"], 0.6)


if __name__ == "__main__":
    unittest.main()
