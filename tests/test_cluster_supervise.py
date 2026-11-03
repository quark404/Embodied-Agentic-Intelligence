import unittest

from cluster_eval.supervise import platform_time, summarize


class SupervisionTests(unittest.TestCase):
    def report(self, status="Succeeded", **changes):
        result = {"job_id": "example", "queue_name": "reviewed-queue", "status": status,
                  "accepted_at": "2026-10-29T10:00:00+00:00",
                  "start_time": "2026-10-29 18:01:00", "finish_time": "2026-10-29 18:02:00",
                  "inventory": {"inventory_passed": True, "single_5090_visible": True,
                                "jfs_origin_verified": True}}
        result.update(changes)
        return result

    def test_times_use_platform_timezone(self):
        self.assertEqual(platform_time("2026-10-29 18:01:00").isoformat(), "2026-10-29T10:01:00+00:00")
        summary = summarize([self.report()])
        self.assertEqual(summary["jobs"][0]["queue_seconds"], 60)
        self.assertEqual(summary["jobs"][0]["runtime_seconds"], 60)
        self.assertEqual(summary["jobs"][0]["accept_to_finish_seconds"], 120)
        self.assertFalse(summary["native_evaluation_ready"])

    def test_queued_or_running_is_not_completed(self):
        for status in ("Queuing", "Running", None):
            summary = summarize([self.report(status=status, finish_time=None)])
            self.assertFalse(summary["all_terminal"])
            self.assertIsNone(summary["first_verified_inventory"])

    def test_failed_first_is_not_successful_inventory(self):
        summary = summarize([self.report(status="Failed", job_id="failed"),
                             self.report(job_id="passed", finish_time="2026-10-29 18:03:00")])
        self.assertEqual(summary["first_terminal"]["job_id"], "failed")
        self.assertEqual(summary["first_verified_inventory"]["job_id"], "passed")
        self.assertTrue(summary["all_terminal"])

    def test_success_without_verified_inventory_is_not_pass(self):
        for inventory in ({}, {"inventory_passed": True}, {"inventory_passed": False}):
            summary = summarize([self.report(inventory=inventory)])
            self.assertIsNone(summary["first_verified_inventory"])

    def test_empty_reports_are_not_all_terminal(self):
        self.assertFalse(summarize([])["all_terminal"])


if __name__ == "__main__":
    unittest.main()
