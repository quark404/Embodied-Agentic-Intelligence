import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cluster_eval import baked_runtime
from cluster_eval.native_job import native_environment, qwen_environment, qwen_install_plan
from cluster_eval.prepare import ROOT, validate_template
from cluster_eval.submit_preflight import job_environment


class BakedNativeTests(unittest.TestCase):
    def test_explicit_native_profile_keeps_three_gpus_and_read_only_inputs(self):
        config = json.loads((ROOT / 'configs/aidi_5090_baked_native.example.json').read_text())
        validate_template(config, profile='baked_native')
        for key, value in (('gpu_per_worker', 1), ('output_bucket', 'robot_lab'), ('wall_time', 241)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_template({**config, key: value}, profile='baked_native')
        config['env'].update(EAI_EXPECTED_GPUS='3', EAI_ORIGIN_SHA256='hash', EAI_STORAGE_MODE='job_local')
        environment = job_environment(config, 'not-a-real-token')
        self.assertEqual(environment['EAI_RUNTIME_SOURCE'], 'prebuilt_image')
        self.assertNotIn('not-a-real-token', str(environment))
        config['input_bucket'] = None
        with self.assertRaises(ValueError):
            job_environment(config, 'not-a-real-token')

    def test_installs_use_private_destinations_and_never_change_the_image_prefix(self):
        runtime = Path('/job_data/eai_test/runtime')
        for name, arguments in baked_runtime.installation_plan(runtime):
            with self.subTest(name=name):
                self.assertNotIn('sudo', arguments)
                self.assertNotIn('conda', arguments)
                if 'venv' not in arguments:
                    self.assertTrue(Path(arguments[0]).is_relative_to(runtime))
        plan = dict(qwen_install_plan(runtime, prebuilt=True))
        self.assertIn(str(baked_runtime.SIM_PYTHON), plan['qwen_venv'])
        self.assertIn(baked_runtime.INTERNAL_INDEX, plan['qwen_dependencies'])
        self.assertIn('12.8', plan['qwen_cuda_check'][-1])
        self.assertTrue(plan['qwen_dependencies'][-1].endswith('qwen35-serving-cu128.txt'))

    def test_curobo_installs_only_in_the_private_overlay_with_inherited_versions_pinned(self):
        runtime = Path('/job_data/eai_test/runtime')
        plan = dict(baked_runtime.installation_plan(runtime))
        install = plan['private_curobo_dependencies']
        self.assertEqual(install[0], str(runtime / 'sim/bin/python'))
        self.assertIn(str(runtime / 'inherited-constraints.txt'), install)
        self.assertEqual(install[-1], str(runtime / 'RoboDojo/third_party/curobo'))
        self.assertNotIn('--upgrade', install)
        self.assertNotIn('--ignore-installed', install)
        self.assertNotIn('--no-deps', install)
        check = plan['simulation_cuda_check'][-1]
        compile(check, '<cuda-check>', 'exec')
        self.assertIn('InverseKinematics', check)
        self.assertIn('program.compile("cubin"', check)

    def test_constraints_preserve_base_cuda_and_reject_requirement_injection(self):
        packages = {'torch': '2.7.0+cu128', 'cuda-core': '1.1.0', 'websockets': '12.0',
                    'nvidia_curobo': 'old', 'numpy': '1.26.0'}
        constraints = baked_runtime.constraint_lines(packages)
        for expected in ('torch==2.7.0+cu128', 'cuda-core==1.1.0', 'websockets==12.0', 'numpy==1.26.0'):
            self.assertIn(expected, constraints)
        self.assertNotIn('nvidia-curobo', constraints)
        for invalid in ({'torch\n--index-url': '1.0'}, {'torch': '1.0\n--upgrade'}):
            with self.assertRaises(ValueError):
                baked_runtime.constraint_lines(invalid)

    def test_curobo_package_version_is_explicit_local_metadata_not_an_upstream_release(self):
        with patch.dict(os.environ, {'PIP_TARGET': '/outside', 'PIP_EXTRA_INDEX_URL': 'unapproved'}):
            environment = baked_runtime.baked_environment(Path('/job_data/eai_test/runtime'))
        self.assertNotIn('PIP_TARGET', environment)
        self.assertNotIn('PIP_EXTRA_INDEX_URL', environment)
        self.assertEqual(environment['PIP_REQUIRE_VIRTUALENV'], 'true')
        self.assertEqual(environment['CUROBO_USE_PYBIND'], '0')
        self.assertEqual(environment['SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO'],
                         '0.8.0+eai.baked')
        self.assertNotIn('SETUPTOOLS_SCM_PRETEND_VERSION', environment)

    def test_prebuilt_workers_keep_disjoint_gpus_and_private_kit_storage(self):
        runtime = Path('/job_data/eai_test/runtime')
        with patch.dict(os.environ, {'EAI_RUNTIME_SOURCE': 'prebuilt_image'}, clear=True):
            qwen = qwen_environment(runtime)
            simulation = native_environment(runtime, runtime / 'config.json', {'native_run_id': 'native-test'})
        self.assertEqual(qwen['CUDA_VISIBLE_DEVICES'], '1,2')
        self.assertEqual(simulation['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(qwen['UV_DEFAULT_INDEX'], baked_runtime.INTERNAL_INDEX)
        self.assertNotIn('UV_PYTHON_PREFERENCE', qwen)
        self.assertIn(str(runtime / 'kit-native'), simulation['ROBODOJO_KIT_ARGS'])
        self.assertEqual(simulation['ROBODOJO_ENV_CONFIG_DIR'], str(runtime / 'RoboDojo/env_cfg'))

    def test_reviewed_launcher_variants_only_patch_the_private_original(self):
        for forwarding in (False, True):
            with self.subTest(forwarding=forwarding), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'source'
                launcher = source / 'scripts/eval_policy.sh'
                launcher.parent.mkdir(parents=True)
                content = 'KIT_ARGS="${ROBODOJO_KIT_ARGS:-}"\n' if forwarding else 'KIT_ARGS=""\n'
                launcher.write_text(content)
                digest = hashlib.sha256(content.encode()).hexdigest()
                (root / 'configs').mkdir()
                (root / 'configs/robodojo-baked-source-lock.json').write_text(json.dumps({
                    'reviewed_launchers': {digest: {'forwards_kit_args': forwarding}}}))
                with patch('cluster_eval.baked_runtime.ROOT', root):
                    reviewed = baked_runtime.review_launcher(launcher)
                    result = baked_runtime.configure_private_launcher(source, reviewed)
                self.assertEqual(result['modified'], not forwarding)
                self.assertEqual(result['sha256'], digest)
                self.assertEqual(launcher.read_text(), 'KIT_ARGS="${ROBODOJO_KIT_ARGS:-}"\n')

    def test_similar_but_unreviewed_launcher_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / 'eval_policy.sh'
            launcher.write_text('KIT_ARGS="${ROBODOJO_KIT_ARGS:-}"\necho unreviewed\n')
            with self.assertRaisesRegex(ValueError, 'both reviewed source versions'):
                baked_runtime.review_launcher(launcher)

    def test_private_launcher_refuses_a_symlink_back_to_the_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'base.sh'
            original.write_text('KIT_ARGS=""\n')
            source = root / 'private'
            (source / 'scripts').mkdir(parents=True)
            (source / 'scripts/eval_policy.sh').symlink_to(original)
            with self.assertRaisesRegex(ValueError, 'private source copy'):
                baked_runtime.configure_private_launcher(source, {})
            self.assertEqual(original.read_text(), 'KIT_ARGS=""\n')

    def test_setup_refuses_control_host_without_installing_or_copying(self):
        with patch('cluster_eval.baked_runtime.validate_local_destination', side_effect=ValueError('not allocated')), \
                patch('cluster_eval.baked_runtime.subprocess.run') as installer, \
                patch('cluster_eval.baked_runtime.shutil.copytree') as copy, self.assertRaises(ValueError):
            baked_runtime.setup(Path('/tmp/runtime'), {})
        installer.assert_not_called()
        copy.assert_not_called()

    def test_mismatched_task_or_reward_code_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'task/RoboDojo/tasks/general_pickup.py'
            source.parent.mkdir(parents=True)
            source.write_text('altered reward code')
            with self.assertRaisesRegex(ValueError, 'differs from the reference'):
                baked_runtime.validate_core_sources(root)


if __name__ == '__main__':
    unittest.main()
