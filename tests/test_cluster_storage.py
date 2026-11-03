import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cluster_eval import probe, storage
from cluster_eval.status import collect_status
from cluster_eval.submit_preflight import ENV_KEYS, job_environment
from cluster_eval.supervise import summarize


class ClusterStorageTests(unittest.TestCase):
    def test_platform_job_data_alias_is_accepted_only_under_the_verified_mount(self):
        platform_root = self.root / 'running_root'
        platform_root.mkdir()
        storage.JOB_DATA.symlink_to(platform_root, target_is_directory=True)
        canonical_result = platform_root / f'eai_{self.run_id}/results'
        with patch('cluster_eval.storage.PLATFORM_DATA_ROOT', platform_root), \
                patch('cluster_eval.storage.os.path.ismount', return_value=True), \
                patch.dict(os.environ, self.environment, clear=True):
            result = storage.create_local_results(self.package, self.result, self.run_id, self.digest)
            self.assertEqual(result, canonical_result)
            logical_runtime = self.result.parent / 'runtime'
            canonical_runtime = canonical_result.parent / 'runtime'
            self.assertEqual(storage.validate_local_destination(logical_runtime, self.package), canonical_runtime)
            self.assertEqual(storage.validate_local_destination(canonical_runtime, self.package), canonical_runtime)
            foreign = platform_root / 'another-user'
            foreign.mkdir()
            canonical_runtime.symlink_to(foreign, target_is_directory=True)
            with self.assertRaises(ValueError):
                storage.validate_local_destination(logical_runtime, self.package)

    def test_unmounted_platform_alias_is_not_a_local_fallback(self):
        platform_root = self.root / 'running_root'
        platform_root.mkdir()
        storage.JOB_DATA.symlink_to(platform_root, target_is_directory=True)
        with patch('cluster_eval.storage.PLATFORM_DATA_ROOT', platform_root), \
                patch('cluster_eval.storage.os.path.ismount', return_value=False), \
                self.assertRaises(ValueError):
            storage.create_local_results(self.package, self.result, self.run_id, self.digest)
        self.assertEqual(list(platform_root.iterdir()), [])

    def test_three_gpu_probe_requires_explicit_native_topology(self):
        output = io.StringIO()
        query = {'returncode': 0, 'stderr': '', 'stdout': '\n'.join(
            f'{index}, NVIDIA GeForce RTX 5090, 570.169, 32607' for index in range(3))}
        with patch.dict(os.environ, {**self.environment, 'EAI_EXPECTED_GPUS': '3'}, clear=True), \
                patch('cluster_eval.probe.run_command', return_value=query), \
                patch('cluster_eval.probe.library_status', return_value={'loadable': True}), \
                patch('cluster_eval.probe.metadata.version', return_value='test'), \
                patch('cluster_eval.probe.network_inventory', return_value={}), \
                patch('cluster_eval.probe.resource_inventory', return_value={}), \
                patch('cluster_eval.probe.inspect_image_runtime', return_value={}), \
                contextlib.redirect_stdout(output):
            probe.main()
        envelope = storage.parse_inventory_export(output.getvalue(), self.run_id, self.digest)
        self.assertTrue(envelope['inventory']['inventory_passed'])
        self.assertTrue(envelope['inventory']['gpu_topology_verified'])
        self.assertFalse(envelope['inventory']['single_5090_visible'])
        self.assertEqual(envelope['inventory']['expected_gpus'], 3)

    def test_prebuilt_image_inspection_does_not_depend_on_any_download_endpoint(self):
        output = io.StringIO()
        query = {'returncode': 0, 'stdout': '0, NVIDIA GeForce RTX 5090, 570.169, 32607', 'stderr': ''}
        with patch.dict(os.environ, self.environment, clear=True), \
                patch('cluster_eval.probe.run_command', return_value=query), \
                patch('cluster_eval.probe.library_status', return_value={'loadable': True}), \
                patch('cluster_eval.probe.metadata.version', return_value='test'), \
                patch('cluster_eval.probe.network_inventory') as network, \
                patch('cluster_eval.probe.resource_inventory') as resources, \
                patch('cluster_eval.probe.inspect_image_runtime', return_value={'modified_image': False}), \
                contextlib.redirect_stdout(output):
            probe.main(check_network=False)
        network.assert_not_called()
        resources.assert_not_called()
        envelope = storage.parse_inventory_export(output.getvalue(), self.run_id, self.digest)
        self.assertTrue(envelope['inventory']['inventory_passed'])
        self.assertFalse(envelope['inventory']['network_checked'])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / 'running_package/code_package'
        self.package.mkdir(parents=True)
        self.run_id = 'local_storage_test'
        self.origin = json.dumps({'run_id': self.run_id, 'writer': 'development_host'}).encode()
        self.digest = hashlib.sha256(self.origin).hexdigest()
        (self.package / 'origin.json').write_bytes(self.origin)
        for target, value in (('cluster_eval.storage.JOB_DATA', self.root / 'job_data'),
                              ('cluster_eval.storage.PACKAGE_ROOT', self.package.parent),
                              ('cluster_eval.probe.ROOT', self.package)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.result = storage.local_result_path(self.run_id)
        self.environment = {'WORKING_PATH': str(self.package), 'EAI_CLUSTER_CONFIRMED': '1',
                            'EAI_STORAGE_MODE': 'job_local', 'EAI_RUN_ID': self.run_id,
                            'EAI_RESULT_DIR': str(self.result), 'EAI_ORIGIN_SHA256': self.digest}
        self.inventory = {'run_id': self.run_id, 'storage_mode': 'job_local',
                          'storage_origin_verified': True, 'jfs_origin_verified': False,
                          'single_5090_visible': True, 'inventory_passed': True,
                          'native_evaluation_ready': False}

    def test_explicit_local_storage_does_not_require_jfs(self):
        with patch.dict(os.environ, self.environment, clear=True), \
                patch('os.path.ismount', return_value=False):
            result = probe.check_context()
        self.assertEqual(result, self.result)
        self.assertEqual((result / 'origin.json').read_bytes(), self.origin)
        self.assertEqual(result.stat().st_mode & 0o777, 0o700)
        self.assertEqual((result / 'origin.json').stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            storage.create_local_results(self.package, self.result, self.run_id, self.digest)

    def test_local_storage_rejects_wrong_origin_and_development_package(self):
        for package, digest in ((self.package, 'wrong'), (self.root, self.digest)):
            with self.subTest(package=package, digest=digest), self.assertRaises(ValueError):
                storage.create_local_results(package, self.result, self.run_id, digest)
        self.assertFalse(self.result.exists())

    def test_local_storage_rejects_traversal_and_symlinks(self):
        for run_id in ('../other', '/tmp/other', '', 'unsafe name'):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                storage.local_result_path(run_id)
        elsewhere = self.root / 'elsewhere'
        elsewhere.mkdir()
        (self.root / 'job_data').symlink_to(elsewhere)
        with self.assertRaises(ValueError):
            storage.create_local_results(self.package, self.result, self.run_id, self.digest)
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_unknown_storage_mode_and_missing_confirmation_fail_closed(self):
        for updates in ({'EAI_STORAGE_MODE': 'fallback'}, {'EAI_CLUSTER_CONFIRMED': '0'}):
            with patch.dict(os.environ, {**self.environment, **updates}, clear=True), \
                    self.subTest(updates=updates), self.assertRaises(ValueError):
                probe.check_context()
        self.assertFalse(self.result.exists())

    def test_followup_downloads_and_installs_stay_in_verified_job_workspace(self):
        storage.create_local_results(self.package, self.result, self.run_id, self.digest)
        target = self.result.parent / 'runtime/dataset'
        with patch.dict(os.environ, self.environment, clear=True):
            self.assertEqual(storage.validate_local_destination(target, self.package), target)
            for forbidden in (self.root / 'other-job', self.result, self.result.parent,
                              self.result / 'inventory.json'):
                with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                    storage.validate_local_destination(forbidden, self.package)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            storage.validate_local_destination(target, self.package)

    def test_followup_scope_rejects_a_replaced_origin_marker(self):
        storage.create_local_results(self.package, self.result, self.run_id, self.digest)
        (self.result / 'origin.json').write_text('{}')
        with patch.dict(os.environ, self.environment, clear=True), self.assertRaises(ValueError):
            storage.validate_local_destination(self.result.parent / 'runtime', self.package)

    def test_export_round_trip_and_repeated_identical_log_lines(self):
        line = storage.export_inventory(self.inventory, self.digest)
        recovered = storage.parse_inventory_export('timestamp ' + line + '\n' + line,
                                                    self.run_id, self.digest)
        self.assertEqual(recovered['inventory'], self.inventory)
        self.assertIsNone(storage.parse_inventory_export('no exported data', self.run_id, self.digest))

    def test_export_rejects_wrong_identity_checksum_and_conflicting_records(self):
        line = storage.export_inventory(self.inventory, self.digest)
        for run_id, digest in (('other-run', self.digest), (self.run_id, 'wrong')):
            with self.subTest(run_id=run_id, digest=digest), self.assertRaises(ValueError):
                storage.parse_inventory_export(line, run_id, digest)
        envelope = json.loads(line[len(storage.EXPORT_PREFIX):])
        envelope['inventory']['inventory_passed'] = False
        with self.assertRaisesRegex(ValueError, 'checksum'):
            storage.parse_inventory_export(storage.EXPORT_PREFIX + json.dumps(envelope),
                                           self.run_id, self.digest)
        changed = storage.export_inventory({**self.inventory, 'inventory_passed': False}, self.digest)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            storage.parse_inventory_export(line + '\n' + changed, self.run_id, self.digest)

    def test_export_is_bounded_and_cannot_claim_a_different_storage_mode(self):
        with self.assertRaises(ValueError):
            storage.export_inventory({**self.inventory, 'large': 'x' * storage.MAX_EXPORT_BYTES}, self.digest)
        line = storage.export_inventory({**self.inventory, 'storage_mode': 'jfs'}, self.digest)
        with self.assertRaises(ValueError):
            storage.parse_inventory_export(line, self.run_id, self.digest)

    def test_persistence_is_private_idempotent_and_never_overwrites(self):
        path = self.root / 'recovered.json'
        storage.persist_export(path, self.inventory)
        storage.persist_export(path, self.inventory)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(ValueError):
            storage.persist_export(path, {'replacement': True})
        self.assertEqual(json.loads(path.read_text()), self.inventory)
        self.assertEqual(list(self.root.glob('.eai-export-*')), [])

    def test_local_probe_exports_inventory_without_real_gpu_or_network_in_test(self):
        output = io.StringIO()
        query = {'returncode': 0, 'stdout': '0, NVIDIA GeForce RTX 5090, 570.169, 32607', 'stderr': ''}
        with patch.dict(os.environ, self.environment, clear=True), \
                patch('cluster_eval.probe.run_command', return_value=query), \
                patch('cluster_eval.probe.library_status', return_value={'loadable': True}), \
                patch('cluster_eval.probe.metadata.version', return_value='test'), \
                patch('cluster_eval.probe.network_inventory', return_value={'test': {'status': 200}}), \
                patch('cluster_eval.probe.resource_inventory', return_value={}), \
                patch('cluster_eval.probe.inspect_image_runtime', return_value={}), \
                contextlib.redirect_stdout(output):
            probe.main()
        envelope = storage.parse_inventory_export(output.getvalue(), self.run_id, self.digest)
        self.assertTrue(envelope['inventory']['inventory_passed'])
        self.assertFalse(envelope['inventory']['jfs_origin_verified'])
        self.assertFalse(envelope['inventory']['native_evaluation_ready'])
        self.assertIsNone(envelope['inventory']['jfs_free_bytes'])
        self.assertEqual(json.loads((self.result / 'inventory.json').read_text()), envelope['inventory'])

    def test_job_environment_allows_only_explicit_local_mode_without_credentials(self):
        environment = {key: 'test' for key in ENV_KEYS}
        environment['EAI_STORAGE_MODE'] = 'job_local'
        self.assertEqual(job_environment({'env': environment}, 'secret')['EAI_STORAGE_MODE'], 'job_local')
        environment['AIDI_TOKEN'] = 'secret'
        with self.assertRaises(ValueError):
            job_environment({'env': environment}, 'secret')

    def test_status_recovers_from_the_receipted_job_and_reuses_verified_export(self):
        run = self.root / 'runs/cluster_eval' / self.run_id
        run.mkdir(parents=True)
        destination = self.root / 'recovered'
        destination.mkdir()
        (destination / 'origin.json').write_bytes(self.origin)
        receipt = {'job_id': 'own-job', 'job_name': 'eai_test', 'queue': 'reviewed-queue',
                   'cluster': 'bcloud-bj-zone1', 'accepted_at': '2026-10-30T00:00:00Z',
                   'storage_mode': 'job_local', 'origin_sha256': self.digest,
                   'result_dir': str(destination)}
        (run / 'receipt.json').write_text(json.dumps(receipt))
        manager = Mock()
        manager.get_job.return_value = SimpleNamespace(job_name='eai_test', queue_name='reviewed-queue',
                                                       job_id='own-job', status='Succeeded')
        manager.get_job_logs.return_value = SimpleNamespace(content=storage.export_inventory(self.inventory, self.digest))
        with patch('cluster_eval.status.ROOT', self.root), patch.object(Path, 'is_relative_to', return_value=True):
            report = collect_status(run, manager)
            self.assertTrue(report['inventory_export_verified'])
            self.assertTrue(report['inventory_exists'])
            self.assertEqual(report['inventory'], self.inventory)
            manager.get_job_logs.assert_called_once_with('own-job', cluster='bcloud-bj-zone1',
                queue='reviewed-queue', mode='filter', expr=storage.EXPORT_PREFIX.strip(), cache='off')
            manager.get_job_logs.reset_mock()
            self.assertTrue(collect_status(run, manager)['inventory_export_verified'])
            manager.get_job_logs.assert_not_called()
            manager.get_job.return_value.job_name = 'another-job'
            with self.assertRaises(ValueError):
                collect_status(run, manager)

    def test_supervisor_requires_verified_export_for_job_local_success(self):
        report = {'job_id': 'test', 'queue_name': 'reviewed-queue', 'status': 'Succeeded',
                  'finish_time': '2026-10-30 12:00:00', 'inventory': self.inventory}
        self.assertIsNone(summarize([report])['first_verified_inventory'])
        report['inventory_export_verified'] = True
        self.assertIsNotNone(summarize([report])['first_verified_inventory'])
        report['status'] = 'Failed'
        self.assertIsNone(summarize([report])['first_verified_inventory'])


if __name__ == '__main__':
    unittest.main()
