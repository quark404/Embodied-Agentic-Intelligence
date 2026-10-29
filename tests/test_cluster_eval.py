import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cluster_eval.prepare import validate_template, write_private
from cluster_eval.probe import check_context, inspect_gpus
from cluster_eval.credentials import load_token, save_token
from cluster_eval.submit_preflight import job_environment, verify_artifacts
from cluster_eval.aidi_cli import validate_arguments


ROOT = Path(__file__).resolve().parents[1]


class ClusterEvalTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/aidi_5090_preflight.example.json").read_text())

    def test_default_is_single_gpu_inventory_only(self):
        validate_template(self.config)
        self.assertEqual(self.config["python_launcher"], "python3")
        self.assertEqual(self.config["python_executable"], "cluster_eval/probe.py")

    def test_manager_wrapper_refuses_mutations(self):
        self.assertEqual(validate_arguments(["manager", "queues"]), ["queues"])
        self.assertEqual(validate_arguments(["manager"]), ["--help"])
        for arguments in ([], ["submit"], ["manager", "stop"], ["manager", "delete"],
                          ["manager", "urgent"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_arguments(arguments)

    def test_rejects_execution_clear_and_resource_changes(self):
        for key, value in (("execute", True), ("clear_workspace", True),
                           ("num_workers", 2), ("gpu_per_worker", 8),
                           ("queue_name", "another-queue"), ("project_id", "robot-lab"),
                           ("wall_time", 60), ("python_launcher", "torchrun"),
                           ("input_bucket", "robot_lab"), ("max_retries", 2),
                           ("priority", 5), ("cpu_per_worker", 64),
                           ("to_upload", ["."]), ("follow_symlinks", True),
                           ("gpu_per_worker", True), ("job_password", "unsafe")):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_template(config)

    def test_no_cluster_actions_without_confirmation(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            check_context()
        with patch.dict(os.environ, {"EAI_CLUSTER_CONFIRMED": "1", "WORKING_PATH": "/tmp"}, clear=True):
            with self.assertRaises(ValueError):
                check_context()

    def test_cluster_requires_real_shared_mount(self):
        variables = {"EAI_CLUSTER_CONFIRMED": "1", "WORKING_PATH": str(ROOT),
                     "EAI_RESULT_DIR": "/jfs-public/test-run"}
        with patch.dict(os.environ, variables, clear=True), patch("os.path.ismount", return_value=False):
            with self.assertRaises(ValueError):
                check_context()

    def test_gpu_inventory_parsing(self):
        devices = inspect_gpus("0, NVIDIA GeForce RTX 5090, 580.65.06, 32607\n")
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["memory_mib"], 32607)

    def test_private_file_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_private(path, {"execute": False})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_private(path, {})

    def test_credentials_private_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials/aidi.json"
            save_token("test-not-a-real-token", path)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(load_token(path), "test-not-a-real-token")
            with self.assertRaises(FileExistsError):
                save_token("replacement", path)
            path.chmod(0o644)
            with self.assertRaises(PermissionError):
                load_token(path)

    def test_credentials_reject_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aidi.json"
            path.symlink_to(Path(directory) / "missing-target")
            with self.assertRaises(FileExistsError):
                save_token("test-not-a-real-token", path)
            self.assertFalse((Path(directory) / "missing-target").exists())

    def test_workload_credentials_are_rejected(self):
        config = copy.deepcopy(self.config)
        config["env"]["EAI_ORIGIN_SHA256"] = "test-digest"
        environment = job_environment(config, "test-secret")
        self.assertEqual(environment["EAI_CLUSTER_CONFIRMED"], "1")
        self.assertEqual(config["env"]["EAI_CLUSTER_CONFIRMED"], "0")
        config["env"]["AIDI_TOKEN"] = "test-secret"
        with self.assertRaises(ValueError):
            job_environment(config, "test-secret")
        del config["env"]["AIDI_TOKEN"]
        config["env"]["EAI_RUN_ID"] = "test-secret"
        with self.assertRaises(ValueError):
            job_environment(config, "test-secret")

    def test_frozen_artifacts_reject_changes_and_additions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "payload"
            path.write_bytes(b"source")
            expected = {"payload": hashlib.sha256(b"source").hexdigest()}
            verify_artifacts(root, expected)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_artifacts(root, expected)
            path.write_bytes(b"source")
            (root / "extra").write_bytes(b"new")
            with self.assertRaises(ValueError):
                verify_artifacts(root, expected)


if __name__ == "__main__":
    unittest.main()
