import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cluster_eval import prebuilt_runtime
from cluster_eval.baked_runtime import installation_plan
from cluster_eval.native_job import qwen_install_plan
from scripts.render_dockerfiles import manifest


class PrebuiltRuntimeTests(unittest.TestCase):
    def test_unset_image_flag_retains_the_existing_installation_path(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(prebuilt_runtime, 'verify') as verify:
            self.assertIsNone(prebuilt_runtime.attach(Path('/unused'), {}))
        verify.assert_not_called()

    def test_arbitrary_image_prefix_is_rejected_without_executing(self):
        for prefix in ('', '/tmp/runtime', '/opt/eai/../eai', '/opt/eai/'):
            with self.subTest(prefix=prefix), patch.dict(os.environ, {'EAI_PREBUILT_RUNTIME': prefix}), \
                    patch.object(prebuilt_runtime, 'verify') as verify, self.assertRaises(ValueError):
                prebuilt_runtime.attach(Path('/unused'), {})
            verify.assert_not_called()

    def test_conflicting_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / 'bridge').symlink_to('/nonexistent/old-environment')
            with patch.dict(os.environ, {'EAI_PREBUILT_RUNTIME': '/opt/eai'}), \
                    patch.object(prebuilt_runtime, 'verify') as verify, self.assertRaises(FileExistsError):
                prebuilt_runtime.attach(runtime, {})
            verify.assert_not_called()
            self.assertFalse((runtime / 'bootstrap').is_symlink())

    def test_manifest_mismatch_does_not_fall_back_to_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            with patch.dict(os.environ, {'EAI_PREBUILT_RUNTIME': '/opt/eai'}), \
                    patch.object(prebuilt_runtime, 'verify', side_effect=ValueError('mismatch')), \
                    self.assertRaisesRegex(ValueError, 'mismatch'):
                prebuilt_runtime.attach(runtime, {})
            self.assertEqual(list(runtime.iterdir()), [])

    def test_verified_environments_are_linked_only_into_the_job(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            with patch.dict(os.environ, {'EAI_PREBUILT_RUNTIME': '/opt/eai'}), \
                    patch.object(prebuilt_runtime, 'verify', return_value={'verified': True}) as verify:
                report = prebuilt_runtime.attach(runtime, {'HOME': directory})
            verify.assert_called_once_with(Path('/opt/eai'), {'HOME': directory})
            self.assertEqual(report, {'verified': True})
            for name in prebuilt_runtime.ENVIRONMENTS:
                self.assertEqual(os.readlink(runtime / name), '/opt/eai/' + name)

    def test_manifest_and_live_package_versions_both_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / 'runtime.json').write_text(json.dumps(manifest()))
            packages = prebuilt_runtime.expected_packages()
            with patch.object(prebuilt_runtime, 'PREFIX', directory), \
                    patch.object(prebuilt_runtime.subprocess, 'check_output',
                                 side_effect=[json.dumps(value) for value in packages.values()]):
                report = prebuilt_runtime.verify(prefix, {})
            self.assertEqual(report['packages'], packages)
            self.assertFalse(report['native_evaluation_ready'])
            packages['qwen']['torch'] += '+cu128'
            with patch.object(prebuilt_runtime, 'PREFIX', directory), \
                    patch.object(prebuilt_runtime.subprocess, 'check_output',
                                 side_effect=[json.dumps(value) for value in packages.values()]):
                report = prebuilt_runtime.verify(prefix, {})
            self.assertEqual(report['packages']['qwen']['torch'], '2.10.0+cu128')
            with patch.object(prebuilt_runtime, 'PREFIX', directory), \
                    patch.object(prebuilt_runtime.subprocess, 'check_output', return_value='{}'), \
                    self.assertRaisesRegex(ValueError, 'package versions differ'):
                prebuilt_runtime.verify(prefix, {})
            (prefix / 'runtime.json').write_text('{}')
            with patch.object(prebuilt_runtime, 'PREFIX', directory), \
                    patch.object(prebuilt_runtime.subprocess, 'check_output') as execute, \
                    self.assertRaisesRegex(ValueError, 'manifest differs'):
                prebuilt_runtime.verify(prefix, {})
            execute.assert_not_called()

    def test_reuse_skips_installation_but_not_dependency_and_gpu_checks(self):
        runtime = Path('/job_data/eai_test/runtime')
        self.assertEqual([stage for stage, command in installation_plan(runtime, preinstalled=True)],
                         ['bridge_dependency_check', 'simulation_cuda_check'])
        self.assertEqual([stage for stage, command in qwen_install_plan(
            runtime, prebuilt=True, preinstalled=True)], ['qwen_dependency_check', 'qwen_cuda_check'])
        self.assertIn('simulation_overlay', dict(installation_plan(runtime)))
        self.assertIn('qwen_dependencies', dict(qwen_install_plan(runtime, prebuilt=True)))
        with self.assertRaises(ValueError):
            qwen_install_plan(runtime, preinstalled=True)


if __name__ == '__main__':
    unittest.main()
