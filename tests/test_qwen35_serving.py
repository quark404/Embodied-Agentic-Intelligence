import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/serve_qwen35.sh"


class QwenServingTests(unittest.TestCase):
    def run_launcher(self, used_memory, *, selected_gpu="6", allocation=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            environment = root / "environment"
            tools.mkdir()
            (environment / "bin").mkdir(parents=True)
            files = {
                tools / "nvidia-smi": '#!/bin/sh\nprintf "%s\\n" "$FAKE_GPU_MEMORY"\n',
                environment / "bin/ninja": "#!/bin/sh\nexit 0\n",
                environment / "bin/vllm": (
                    "#!/usr/bin/env python3\n"
                    "import json, os, shutil, sys\n"
                    "print(json.dumps({'args': sys.argv[1:], "
                    "'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), "
                    "'order': os.environ.get('CUDA_DEVICE_ORDER'), "
                    "'ninja': shutil.which('ninja')}))\n"
                ),
            }
            for path, content in files.items():
                path.write_text(content)
                path.chmod(0o755)
            variables = dict(os.environ, QWEN_ENV=str(environment), QWEN_GPU="6",
                             QWEN_PORT="18000", FAKE_GPU_MEMORY=str(used_memory),
                             PATH=f"{tools}:{os.environ['PATH']}")
            variables.pop("CUDA_VISIBLE_DEVICES", None)
            if allocation is not None:
                variables["CUDA_VISIBLE_DEVICES"] = allocation
            if selected_gpu is None:
                variables.pop("QWEN_GPU")
            else:
                variables["QWEN_GPU"] = selected_gpu
            result = subprocess.run(["bash", str(LAUNCHER)], env=variables,
                                    capture_output=True, text=True, timeout=10)
            expected_ninja = str(environment / "bin/ninja")
            return result, expected_ninja

    def test_idle_gpu_uses_environment_tools_and_loopback_only(self):
        result, expected_ninja = self.run_launcher(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        launch = json.loads(result.stdout)
        self.assertEqual(launch["visible"], "6")
        self.assertEqual(launch["order"], "PCI_BUS_ID")
        self.assertEqual(launch["ninja"], expected_ninja)
        self.assertEqual(launch["args"][0:2], ["serve", "Qwen/Qwen3.5-27B"])
        self.assertEqual(launch["args"][launch["args"].index("--host") + 1], "127.0.0.1")

    def test_busy_gpu_is_rejected_without_launching(self):
        result, _ = self.run_launcher(60000)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to share or stop another job", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_requires_explicit_gpu_selection(self):
        result, _ = self.run_launcher(0, selected_gpu=None)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no default GPU", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_respects_inherited_allocation(self):
        for allocation in ("0,1", "", "GPU-not-a-numeric-index"):
            with self.subTest(allocation=allocation):
                result, _ = self.run_launcher(0, allocation=allocation)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("refusing to override", result.stderr)
                self.assertEqual(result.stdout, "")
        result, _ = self.run_launcher(0, allocation="5,6")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unreadable_gpu_memory_is_rejected(self):
        result, _ = self.run_launcher("N/A")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not verify GPU memory", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
