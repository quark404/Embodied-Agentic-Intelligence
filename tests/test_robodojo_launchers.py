import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / 'integrations/xpolicylab/Embodied_Agentic/setup_eval_env_client.sh'


class RoboDojoLauncherTests(unittest.TestCase):
    def invoke(self, *, environment_kind='directory', mode='sim'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bench = root / 'RoboDojo'
            policy = bench / 'XPolicyLab/policy/Embodied_Agentic'
            policy.mkdir(parents=True)
            launcher = policy / LAUNCHER.name
            shutil.copyfile(LAUNCHER, launcher)
            (policy / 'deploy.yml').write_text('eval_batch: false\nprotocol: ws\n')
            sim = root / 'private-sim'
            (sim / 'bin').mkdir(parents=True)
            (sim / 'bin/python').symlink_to(sys.executable)
            metadata_python = sim / 'bin/python3'
            metadata_python.write_text('#!/bin/sh\n[ "$1" = "-c" ] && [ -f "$3" ]\n')
            metadata_python.chmod(0o755)
            tools = root / 'tools'
            tools.mkdir()
            conda = tools / 'conda'
            conda.write_text('#!/bin/sh\necho "Unexpected Conda invocation" >&2\nexit 99\n')
            conda.chmod(0o755)
            native = bench / 'scripts/eval_policy.sh'
            native.parent.mkdir()
            native.write_text("#!/bin/bash\nexec python -c 'import json,os,sys; print(json.dumps({"
                              '"python":sys.executable,"args":sys.argv[1:],'
                              '"retries":os.environ.get("ROBODOJO_MAX_BASH_RETRIES")}))'
                              "' \"$@\"\n")
            fallback = bench / 'XPolicyLab/utils/setup_env_client.sh'
            fallback.parent.mkdir()
            fallback.write_text('#!/bin/sh\necho legacy-conda-route\n')
            selection = {'directory': str(sim), 'interpreter': str(sim / 'bin/python'),
                         'missing': str(root / 'nonexistent'), 'conda': 'named-sim'}[environment_kind]
            variables = dict(os.environ, PATH=f'{tools}:{os.environ["PATH"]}', EVAL_ENV_TYPE=mode)
            variables.pop('ROBODOJO_MAX_BASH_RETRIES', None)
            result = subprocess.run(['bash', str(launcher), 'RoboDojo', 'general_pickup',
                'q_only', 'arx_x5', 'ee', '0', '2', selection,
                'ckpt_name=q_only,action_type=ee', '18001', 'localhost'],
                env=variables, capture_output=True, text=True, timeout=10)
            return result, str(sim / 'bin/python')

    def test_explicit_venv_reaches_native_eval_without_conda(self):
        for kind in ('directory', 'interpreter'):
            with self.subTest(kind=kind):
                result, expected_python = self.invoke(environment_kind=kind)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(result.stdout)
                self.assertEqual(record['python'], expected_python)
                self.assertEqual(record['retries'], '0')
                for flag, value in (('--task_name', 'general_pickup'), ('--device_id', '2'),
                                     ('--protocol', 'ws'), ('--eval_batch', 'false')):
                    self.assertEqual(record['args'][record['args'].index(flag) + 1], value)

    def test_missing_explicit_environment_never_falls_back_to_conda(self):
        result, _ = self.invoke(environment_kind='missing')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refusing Conda fallback', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_explicit_environment_does_not_run_debug_or_real_hardware(self):
        for mode in ('debug', 'real'):
            with self.subTest(mode=mode):
                result, _ = self.invoke(mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('simulation only', result.stderr)

    def test_legacy_named_environment_keeps_the_upstream_route(self):
        result, _ = self.invoke(environment_kind='conda')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'legacy-conda-route')


if __name__ == '__main__':
    unittest.main()
