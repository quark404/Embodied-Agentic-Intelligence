import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from cluster_eval import probe


class ClusterProbeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.environment = {"EAI_CLUSTER_CONFIRMED": "1", "WORKING_PATH": str(probe.ROOT),
                            "EAI_RUN_ID": "diagnostic-run", "EAI_RESULT_DIR": "/jfs-public/test-run"}
        self.mountinfo = (
            "21 1 0:1 / / rw - overlay overlay rw,private-option=hidden\n"
            "22 21 0:2 / /jfs-public/smartauto rw - fuse.juicefs private-source rw\n"
            "23 21 0:3 / /another-project rw - ext4 private-source rw\n"
        )

    def test_mount_summary_excludes_sources_options_and_unrelated_paths(self):
        report = probe.inspect_mounts(self.mountinfo, Path("/jfs-public/test-run"))
        self.assertEqual(report, [{"mountpoint": "/", "filesystem": "overlay"},
                                  {"mountpoint": "/jfs-public/smartauto", "filesystem": "fuse.juicefs"}])
        self.assertNotIn("private", json.dumps(report))

    def test_malformed_mount_records_are_ignored(self):
        self.assertEqual(probe.inspect_mounts("invalid\n1 2 - \n", Path("/jfs-public/run")), [])

    def test_missing_shared_directory_does_not_pass_or_write(self):
        with patch.dict(os.environ, self.environment, clear=True), \
                patch("cluster_eval.probe.Path.is_dir", return_value=False), \
                patch("cluster_eval.probe.Path.is_file", return_value=False), \
                patch("cluster_eval.probe.Path.read_text", return_value=self.mountinfo), \
                patch("cluster_eval.probe.os.path.ismount", return_value=False), \
                patch("cluster_eval.probe.run_command", return_value={"returncode": 0}), \
                patch("cluster_eval.probe.library_status", return_value={"loadable": False}), \
                patch("cluster_eval.probe.os.open") as open_file:
            report = probe.context_diagnostics()
        self.assertFalse(report["jfs_root_exists"])
        self.assertFalse(report["jfs_root_is_mount"])
        self.assertFalse(report["result_directory_exists"])
        self.assertFalse(report["origin_regular_file"])
        self.assertFalse(report["inventory_passed"])
        self.assertFalse(report["native_evaluation_ready"])
        open_file.assert_not_called()

    def test_matching_origin_is_diagnostic_only_and_content_is_not_logged(self):
        origin = json.dumps({"run_id": "diagnostic-run", "writer": "development_host",
                             "nonce": "private-marker"}).encode()
        self.environment["EAI_ORIGIN_SHA256"] = hashlib.sha256(origin).hexdigest()
        with patch.dict(os.environ, self.environment, clear=True), \
                patch("cluster_eval.probe.Path.is_dir", return_value=True), \
                patch("cluster_eval.probe.Path.is_file", return_value=True), \
                patch("cluster_eval.probe.Path.is_symlink", return_value=False), \
                patch("cluster_eval.probe.Path.read_bytes", return_value=origin), \
                patch("cluster_eval.probe.Path.read_text", return_value=self.mountinfo), \
                patch("cluster_eval.probe.os.path.ismount", return_value=False), \
                patch("cluster_eval.probe.run_command", return_value={"returncode": 0}), \
                patch("cluster_eval.probe.library_status", return_value={"loadable": False}):
            report = probe.context_diagnostics()
        self.assertTrue(report["origin_sha256_matches"])
        self.assertTrue(report["origin_identity_matches"])
        self.assertFalse(report["inventory_passed"])
        self.assertNotIn("private-marker", json.dumps(report))

    def test_diagnostics_refuse_development_host_execution(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("cluster_eval.probe.run_command") as command, \
                self.assertRaises(ValueError):
            probe.context_diagnostics()
        command.assert_not_called()

    def test_main_keeps_failure_and_does_not_create_fallback_inventory(self):
        output = io.StringIO()
        with patch.dict(os.environ, self.environment, clear=True), \
                patch("cluster_eval.probe.check_context", side_effect=ValueError("missing mount")), \
                patch("cluster_eval.probe.context_diagnostics", return_value={"jfs_root_exists": False}), \
                patch("cluster_eval.probe.os.open") as open_file, \
                contextlib.redirect_stdout(output), self.assertRaises(ValueError):
            probe.main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["event"], "preflight_context_failed")
        self.assertFalse(report["inventory_passed"])
        self.assertFalse(report["native_evaluation_ready"])
        open_file.assert_not_called()

    def test_diagnostic_failure_preserves_original_context_error(self):
        output = io.StringIO()
        with patch.dict(os.environ, self.environment, clear=True), \
                patch("cluster_eval.probe.check_context", side_effect=ValueError("missing mount")), \
                patch("cluster_eval.probe.context_diagnostics", side_effect=PermissionError()), \
                contextlib.redirect_stdout(output), self.assertRaisesRegex(ValueError, "missing mount"):
            probe.main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["diagnostics"], {"error_type": "PermissionError"})
        self.assertFalse(report["inventory_passed"])


if __name__ == "__main__":
    unittest.main()
