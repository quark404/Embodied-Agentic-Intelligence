from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_robodojo import check_endpoint, classify_gpu, readiness_summary
from scripts.prepare_robodojo_sources import prepare


class DeploymentHelperTests(unittest.TestCase):
    def test_metadata_success_is_not_a_native_result(self):
        report = readiness_summary([])
        self.assertTrue(report["metadata_preflight_passed"])
        self.assertFalse(report["native_ready"])
        self.assertTrue(report["remaining_verification"])

    def test_blockers_are_preserved(self):
        report = readiness_summary(["Missing simulator"])
        self.assertFalse(report["metadata_preflight_passed"])
        self.assertEqual(report["blockers"], ["Missing simulator"])

    def test_gpu_classification_does_not_promise_rendering(self):
        self.assertEqual(classify_gpu("NVIDIA H20"), "unsupported_no_rt_cores")
        self.assertEqual(classify_gpu("NVIDIA GeForce RTX 5090"),
                         "requires_isaac_compatibility_test")

    def test_nonlocal_endpoint_is_not_contacted(self):
        with patch("scripts.check_robodojo.urlopen") as request:
            self.assertFalse(check_endpoint("https://external.invalid/v1", "Qwen")["ok"])
            self.assertFalse(check_endpoint("http://user:password@localhost/v1", "Qwen")["ok"])
            request.assert_not_called()

    def test_source_preparation_never_reuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "keep"
            marker.write_text("original")
            with patch("scripts.prepare_robodojo_sources.checkout") as checkout:
                with self.assertRaises(FileExistsError):
                    prepare(root)
                checkout.assert_not_called()
            self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
