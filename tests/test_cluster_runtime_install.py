import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cluster_eval.network import proxy_environment
from cluster_eval.runtime_install import execute_installation, installation_plan, runtime_environment
from cluster_eval.storage import BOOTSTRAP_WHEEL_URL, PUBLIC_DEPENDENCY_URLS


class RuntimeInstallTests(unittest.TestCase):
    def test_plan_uses_only_private_environment_installers_and_pinned_sources(self):
        runtime = Path('/job_data/eai_test/runtime')
        plan = installation_plan(runtime, package=Path('/running_package/code_package'), python='/usr/bin/python3')
        self.assertEqual(plan[0]['argv'], ['/usr/bin/python3', '-m', 'venv', str(runtime / 'bootstrap')])
        self.assertIn('3.11.14', next(step for step in plan if step['stage'] == 'managed_python')['argv'])
        for step in plan[1:]:
            with self.subTest(stage=step['stage']):
                if step['stage'] != 'pinned_sources':
                    self.assertTrue(Path(step['argv'][0]).is_relative_to(runtime))
                self.assertFalse(set(step['argv']) & {'sudo', 'apt-get', 'conda', '--remote', '--upgrade'})
        source = next(step for step in plan if step['stage'] == 'pinned_sources')
        self.assertIn('scripts.prepare_robodojo_sources', source['argv'])
        sim = next(step for step in plan if step['stage'] == 'simulation_dependencies')
        bridge = next(step for step in plan if step['stage'] == 'bridge_dependencies')
        self.assertIn(str(runtime / 'sim/bin/python'), sim['argv'])
        self.assertIn(str(runtime / 'bridge/bin/python'), bridge['argv'])
        self.assertIn('-c', sim['argv'])
        self.assertNotIn('-c', bridge['argv'])

    def test_execute_refuses_development_host_without_starting_any_installer(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('cluster_eval.runtime_install.subprocess.run') as command, \
                self.assertRaises(ValueError):
            execute_installation(Path('/tmp/must-not-be-created'))
        command.assert_not_called()

    def test_unreachable_dependencies_stop_before_creating_an_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime'
            (root / 'inventory.json').write_text(json.dumps({
                'run_id': 'network_test', 'storage_mode': 'job_local',
                'inventory_passed': True, 'single_5090_visible': True,
                'network': {'https://pypi.org/simple/': {'reachable': False, 'error_type': 'URLError'}},
            }))
            with patch.dict(os.environ, {'EAI_RUN_ID': 'network_test'}, clear=True), \
                    patch('cluster_eval.runtime_install.validate_local_destination', return_value=runtime), \
                    patch('cluster_eval.runtime_install.local_result_path', return_value=root), \
                    patch('cluster_eval.runtime_install.subprocess.run') as command, \
                    self.assertRaisesRegex(ValueError, 'Public dependency endpoints are unreachable'):
                execute_installation(runtime)
            command.assert_not_called()
            self.assertFalse(runtime.exists())

    def test_installers_keep_explicit_proxy_while_isolating_home_and_caches(self):
        network = proxy_environment('http://10.0.0.2:18000')
        runtime = Path('/job_data/eai_network_test/runtime')
        with patch.dict(os.environ, {**network, 'HOME': '/some-existing-home'}, clear=True):
            environment = runtime_environment(runtime)
        self.assertEqual({name: environment[name] for name in network}, network)
        self.assertEqual(environment['HOME'], str(runtime / 'home'))
        self.assertEqual(environment['PIP_CACHE_DIR'], str(runtime / 'cache/pip'))
        self.assertEqual(environment['HF_HUB_DISABLE_IMPLICIT_TOKEN'], '1')

    def test_accessible_index_does_not_allow_a_denied_wheel_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime'
            network = {url: {'reachable': True, 'status': 200} for url in PUBLIC_DEPENDENCY_URLS}
            network[BOOTSTRAP_WHEEL_URL]['status'] = 403
            (root / 'inventory.json').write_text(json.dumps({
                'run_id': 'network_test', 'storage_mode': 'job_local',
                'inventory_passed': True, 'single_5090_visible': True, 'network': network,
            }))
            with patch.dict(os.environ, {'EAI_RUN_ID': 'network_test'}, clear=True), \
                    patch('cluster_eval.runtime_install.validate_local_destination', return_value=runtime), \
                    patch('cluster_eval.runtime_install.local_result_path', return_value=root), \
                    patch('cluster_eval.runtime_install.subprocess.run') as command, \
                    self.assertRaisesRegex(ValueError, 'Bootstrap wheel download endpoint failed'):
                execute_installation(runtime)
            command.assert_not_called()
            self.assertFalse(runtime.exists())

    def test_curobo_installs_its_cuda_backend_before_runtime_check(self):
        plan = installation_plan(Path('/job_data/eai_test/runtime'))
        stages = [step['stage'] for step in plan]
        curobo = plan[stages.index('curobo')]
        self.assertTrue(curobo['argv'][-1].endswith('/third_party/curobo[cu12]'))
        self.assertLess(stages.index('curobo'), stages.index('cuda_backend_check'))
        constraints = Path('requirements/robodojo-runtime-constraints.txt').read_text().splitlines()
        self.assertIn('cuda-core==0.7.0', constraints)
        dependencies = Path('requirements/robodojo-sim.txt').read_text().splitlines()
        self.assertIn('setuptools-scm==8.3.1', dependencies)


if __name__ == '__main__':
    unittest.main()
