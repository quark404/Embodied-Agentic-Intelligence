import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cluster_eval.native_artifacts import CAMERAS, FILES, validate_native_report
from cluster_eval.native_assets import collect_manifest
from cluster_eval.native_job import (command_step, main, native_environment, prepare_video_binary,
                                     qwen_environment, qwen_install_plan)
from cluster_eval.prepare import ROOT, prepare, validate_template
from cluster_eval.status import collect_status
from cluster_eval.storage import export_record, parse_record
from cluster_eval.submit_preflight import job_environment
from cluster_eval.supervise import summarize
from scripts.download_qwen35 import MODEL, REVISION
from test_cluster_render import png, render_report


def native_report(run_id='native-test', *, success=True):
    native_id = run_id + '_q_only_general_pickup_s0'
    episode = native_id + '-general_pickup-seed0-layout2-trial0'
    camera_files = {f'cameras/{camera}.png': png(index) for index, camera in enumerate(CAMERAS)}
    summary = {'task_id': 'general_pickup', 'condition': 'q_only', 'seed': 0, 'layout_id': 2,
               'native_result_index': 0, 'native_terminal': True, 'native_success': success,
               'episode_id': episode, 'status': 'success' if success else 'decision_budget_exhausted',
               'executed_physical_steps': 1, 'camera_artifacts': {
                   camera: {'path': f'cameras/{camera}.png',
                            'sha256': hashlib.sha256(camera_files[f'cameras/{camera}.png']).hexdigest()}
                   for camera in CAMERAS}}
    objects = {
        '_result.json': {'details': {'0': {'success': success, 'score': float(success), 'layout_id': 2}}},
        'summary.json': summary,
        'experiment.json': {'condition': 'q_only', 'episode_id': episode,
                            'models': {'qwen': {'model': MODEL}}},
        'schedule.json': [{'native_run_id': native_id, 'condition': 'q_only', 'task_id': 'general_pickup',
                           'seed': 0, 'eval_num': 1, 'action_mode': 'ee_delta'}],
        'model-download.json': {'model': MODEL, 'revision': REVISION, 'weight_bytes': 55 * 1024 ** 3},
        'model-check.json': {'model': MODEL, 'passed': True, 'checks': [
            {'name': name, 'passed': True} for name in
            ('text_json', 'three_images', 'native_function_call', 'harness_planner_actor')]},
    }
    files = {name: json.dumps(value).encode() for name, value in objects.items()}
    trace = [{'episode_id': episode, 'event': 'model_call', 'ok': True, 'model': MODEL, 'role': role}
             for role in ('planner', 'actor')]
    trace.append({'episode_id': episode, 'event': 'tool', 'receipt': {'executed_steps': 1}})
    files['trace.jsonl'] = ('\n'.join(json.dumps(event) for event in trace) + '\n').encode()
    files.update(camera_files)
    encoded = {name: {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                      'base64': base64.b64encode(content).decode()} for name, content in files.items()}
    return {'run_id': run_id, 'storage_mode': 'job_local', 'storage_origin_verified': True,
            'benchmark': 'RoboDojo', 'task_id': 'general_pickup', 'episode_complete': True,
            'native_evaluation_ready': True, 'native_success': success, 'files': encoded}


def replace_file(report, name, value):
    content = value if isinstance(value, bytes) else json.dumps(value).encode()
    report['files'][name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                             'base64': base64.b64encode(content).decode()}


class NativeClusterTests(unittest.TestCase):
    def test_native_profile_is_explicit_three_gpu_and_keeps_other_profiles_restricted(self):
        config = json.loads((ROOT / 'configs/aidi_5090_native.example.json').read_text())
        validate_template(config, profile='native')
        for profile in ('inventory', 'render'):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_template(config, profile=profile)
        for key, value in (('gpu_per_worker', 1), ('gpu_per_worker', 4), ('priority', 2),
                           ('wall_time', 241), ('input_bucket', 'shared-data'), ('max_retries', 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_template({**config, key: value}, profile='native')
        with self.assertRaises(ValueError):
            prepare(run_id='no-write', image='image', result_root=Path('/tmp'), profile='native')

    def test_native_environment_never_forwards_token_or_accepts_another_gpu_count(self):
        config = {'python_executable': 'cluster_eval/native_job.py', 'env': {
            'EAI_CLUSTER_CONFIRMED': '0', 'EAI_RUN_ID': 'native-test', 'EAI_RESULT_DIR': '/private',
            'EAI_ORIGIN_SHA256': 'digest', 'PYTHONDONTWRITEBYTECODE': '1',
            'EAI_STORAGE_MODE': 'job_local', 'EAI_EXPECTED_GPUS': '3'}}
        self.assertEqual(job_environment(config, 'private-token')['EAI_EXPECTED_GPUS'], '3')
        config['env']['EAI_EXPECTED_GPUS'] = '4'
        with self.assertRaises(ValueError):
            job_environment(config, 'private-token')
        config['env']['EAI_EXPECTED_GPUS'] = '3'
        config['env']['AIDI_TOKEN'] = 'private-token'
        with self.assertRaises(ValueError):
            job_environment(config, 'private-token')

    def test_native_entry_refuses_the_development_host_without_any_process(self):
        with patch.dict(os.environ, {}, clear=True), patch('cluster_eval.native_job.subprocess.Popen') as start:
            with self.assertRaises(ValueError):
                main()
            start.assert_not_called()

    def test_qwen_and_simulation_use_disjoint_gpus_and_private_paths(self):
        runtime = Path('/job_data/eai_test/runtime')
        qwen = qwen_environment(runtime)
        sim = native_environment(runtime, runtime / 'config.json', {'native_run_id': 'test-episode'})
        self.assertEqual(qwen['CUDA_VISIBLE_DEVICES'], '1,2')
        self.assertEqual(sim['CUDA_VISIBLE_DEVICES'], '0')
        self.assertNotEqual(qwen['TORCH_EXTENSIONS_DIR'], sim['TORCH_EXTENSIONS_DIR'])
        self.assertEqual(sim['ROBOICL_RESULTS_ROOT'], str(runtime / 'native-results'))
        self.assertEqual(sim['ROBOICL_DATA_ROOT'], str(runtime / 'assets'))
        self.assertEqual(sim['ROBODOJO_MAX_BASH_RETRIES'], '0')
        self.assertEqual(sim['ROBODOJO_FATAL_RESTART_COUNT'], '3')
        for stage, command in qwen_install_plan(runtime):
            self.assertTrue(Path(command[0]).is_relative_to(runtime), stage)
        self.assertTrue(qwen_install_plan(runtime)[1][1][-1].endswith('qwen35-serving-cu129.txt'))

    def test_native_processes_preserve_internal_bypass_and_add_loopback(self):
        runtime = Path('/job_data/eai_test/runtime')
        with patch.dict(os.environ, {'no_proxy': '.hobot.cc', 'NO_PROXY': '.horizon.cc'}, clear=True):
            qwen = qwen_environment(runtime)
            sim = native_environment(runtime, runtime / 'config.json', {'native_run_id': 'test-episode'})
        for environment in (qwen, sim):
            self.assertEqual(environment['no_proxy'], environment['NO_PROXY'])
            self.assertEqual(set(environment['NO_PROXY'].split(',')),
                             {'.hobot.cc', '.horizon.cc', 'localhost', '127.0.0.1'})

    def test_stage_timeout_stops_only_its_created_session(self):
        process = Mock(pid=998002)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('native', 1), 0]
        report = {'stages': []}
        with patch('cluster_eval.native_job.subprocess.Popen', return_value=process) as start, \
                patch('cluster_eval.native_job.os.killpg') as stop, \
                patch('cluster_eval.native_job.time.monotonic', return_value=10), \
                self.assertRaises(subprocess.TimeoutExpired):
            command_step('native-test', ['python'], {}, deadline=20, max_seconds=1, report=report)
        self.assertTrue(start.call_args.kwargs['start_new_session'])
        stop.assert_called_once_with(998002, signal.SIGTERM)
        self.assertEqual(report['stages'][0]['stage'], 'native-test')

    def test_asset_inventory_cannot_silently_omit_files(self):
        lock = {'revision': 'pinned', 'file_count': 2, 'total_bytes': 12}
        items = [SimpleNamespace(path='Assets'), SimpleNamespace(path='Assets/a', size=5),
                 SimpleNamespace(path='Assets/b', size=7)]
        self.assertEqual(collect_manifest(items, lock)['file_count'], 2)
        with self.assertRaises(ValueError):
            collect_manifest(items[:-1], lock)

    def test_video_tool_cannot_link_a_system_binary_into_the_private_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / 'runtime'
            runtime.mkdir()
            with patch('cluster_eval.native_job.validate_local_destination'), \
                    patch('cluster_eval.native_job.subprocess.check_output', return_value='/usr/bin/ffmpeg\n'), \
                    patch('cluster_eval.native_job.subprocess.run') as start, \
                    self.assertRaises(ValueError):
                prepare_video_binary(runtime)
            start.assert_not_called()
            self.assertFalse((runtime / 'bin').exists())

    def test_complete_native_audit_joins_original_score_harness_model_and_images(self):
        for success in (True, False):
            with self.subTest(success=success):
                report = native_report(success=success)
                files, verdict = validate_native_report(report)
                self.assertEqual(set(files), FILES)
                self.assertEqual(verdict['native_success'], success)
                self.assertTrue(verdict['episode_complete'])
                self.assertEqual(verdict['executed_physical_steps'], 1)

    def test_native_export_is_chunked_and_bound_to_its_own_job(self):
        report = native_report()
        report['padding'] = 'x' * 300000
        record = export_record(report, 'own-origin', record_type='native')
        self.assertTrue(all(len(line.encode()) < 4096 for line in record.splitlines()))
        self.assertEqual(parse_record(record, 'native-test', 'own-origin', record_type='native')['native'], report)
        with self.assertRaises(ValueError):
            parse_record(record, 'other-job', 'own-origin', record_type='native')
        with self.assertRaises(ValueError):
            parse_record('\n'.join(record.splitlines()[1:]), 'native-test', 'own-origin', record_type='native')

    def test_missing_evidence_or_relabelled_success_is_rejected(self):
        cases = []
        report = native_report()
        del report['files']['trace.jsonl']
        cases.append(report)
        report = native_report(success=False)
        report['native_success'] = True
        cases.append(report)
        report = native_report()
        replace_file(report, '_result.json', {'details': {}})
        cases.append(report)
        report = native_report()
        replace_file(report, 'model-download.json', {'model': MODEL, 'revision': 'wrong', 'weight_bytes': 55 * 1024 ** 3})
        cases.append(report)
        report = native_report()
        trace = base64.b64decode(report['files']['trace.jsonl']['base64']).replace(b'"actor"', b'"other"')
        replace_file(report, 'trace.jsonl', trace)
        cases.append(report)
        report = native_report()
        report['files']['../outside'] = report['files'].pop('summary.json')
        cases.append(report)
        for index, report in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_native_report(report)

    def test_incomplete_attempt_can_return_diagnostics_but_cannot_pass(self):
        complete = native_report()
        failed = {key: value for key, value in complete.items() if key != 'files'}
        failed.update(episode_complete=False, native_evaluation_ready=False,
                      native_success=False, error_type='RuntimeError',
                      diagnostic_files={'trace.jsonl': complete['files']['trace.jsonl']})
        files, verdict = validate_native_report(failed)
        self.assertEqual(set(files), {'trace.jsonl'})
        self.assertFalse(verdict['episode_complete'])
        self.assertFalse(verdict['native_success'])
        failed['native_evaluation_ready'] = True
        with self.assertRaises(ValueError):
            validate_native_report(failed)

    def test_status_recovers_native_files_without_printing_images_or_claiming_failed_jobs_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/cluster_eval/native-test'
            run.mkdir(parents=True)
            output = root / 'output'
            output.mkdir()
            origin = b'native-origin'
            digest = hashlib.sha256(origin).hexdigest()
            (output / 'origin.json').write_bytes(origin)
            receipt = {'job_id': 'own-job', 'job_name': 'eai_native', 'queue': 'own-queue',
                       'cluster': 'bcloud-bj-zone1', 'profile': 'native', 'storage_mode': 'job_local',
                       'expected_gpus': 3, 'origin_sha256': digest, 'result_dir': str(output),
                       'accepted_at': '2026-10-30T04:00:00Z'}
            (run / 'receipt.json').write_text(json.dumps(receipt))
            inventory = {'run_id': run.name, 'storage_mode': 'job_local', 'storage_origin_verified': True,
                         'expected_gpus': 3, 'gpu_topology_verified': True, 'inventory_passed': True}
            native = native_report()
            logs = '\n'.join(export_record(payload, digest, record_type=kind) for kind, payload in
                              (('inventory', inventory), ('render', render_report(run.name)), ('native', native)))
            manager = Mock()
            job = SimpleNamespace(job_id='own-job', job_name='eai_native', queue_name='own-queue',
                                  status='Succeeded', finish_time='2026-10-30 14:00:00')
            manager.get_job.return_value = job
            manager.get_job_logs.return_value = SimpleNamespace(content=logs)
            with patch('cluster_eval.status.ROOT', root), patch.object(Path, 'is_relative_to', return_value=True):
                report = collect_status(run, manager)
                self.assertTrue(report['native_evaluation_ready'])
                self.assertTrue(summarize([report])['jobs'][0]['native_verified'])
                self.assertEqual(set(report['native_artifact_files']), FILES)
                self.assertNotIn('base64', json.dumps(report))
                manager.get_job_logs.assert_called_once()
                job.status = 'Failed'
                self.assertFalse(collect_status(run, manager)['native_evaluation_ready'])


if __name__ == '__main__':
    unittest.main()
