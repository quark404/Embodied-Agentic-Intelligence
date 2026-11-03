import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

from cluster_eval.prepare import ROOT, profile_settings, validate_template
from cluster_eval.rendering import CAMERAS, RESOLUTION, validate_png, validate_render_report
from cluster_eval.runtime_job import main, render_process
from cluster_eval.status import collect_status
from cluster_eval.storage import (RECORD_PREFIXES, export_inventory, export_record,
                                  parse_inventory_export, parse_record)
from cluster_eval.supervise import summarize


def png(offset=0, *, pixel_data=None):
    def chunk(kind, content):
        return (struct.pack('>I', len(content)) + kind + content
                + struct.pack('>I', zlib.crc32(kind + content) & 0xffffffff))
    header = struct.pack('>IIBBBBB', RESOLUTION, RESOLUTION, 8, 2, 0, 0, 0)
    data = bytearray()
    for row in range(RESOLUTION):
        data.append(0)
        for column in range(RESOLUTION):
            data.extend(((column + offset) % 256, (row * 2) % 256, 70))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
            + chunk(b'IDAT', zlib.compress(bytes(data) if pixel_data is None else pixel_data))
            + chunk(b'IEND', b''))


def render_report(run_id='render-test'):
    frames = {}
    for offset, name in enumerate(CAMERAS):
        content = png(offset)
        frames[name] = {'width': RESOLUTION, 'height': RESOLUTION, 'channels': 3, 'stddev': 32.0,
                        'sha256': hashlib.sha256(content).hexdigest(),
                        'png_base64': base64.b64encode(content).decode()}
    return {'run_id': run_id, 'storage_mode': 'job_local', 'storage_origin_verified': True,
            'render_passed': True, 'native_evaluation_ready': False,
            'scene': 'procedural_rgb_cubes', 'frames': frames}


class RenderTests(unittest.TestCase):
    def test_render_profile_is_explicit_and_does_not_relax_inventory_limits(self):
        config = json.loads((ROOT / 'configs/aidi_5090_render.example.json').read_text())
        validate_template(config, profile='render')
        with self.assertRaises(ValueError):
            validate_template(config)
        for key, value in (('wall_time', 121), ('gpu_per_worker', 2), ('execute', True),
                           ('python_executable', '/tmp/arbitrary.py'), ('to_upload', ['.']),
                           ('input_bucket', 'robot_lab'), ('priority', 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_template({**config, key: value}, profile='render')
        with self.assertRaises(ValueError):
            profile_settings('../arbitrary')

    def test_three_small_rgb_pngs_are_verified_without_pillow_or_numpy(self):
        report = render_report()
        images = validate_render_report(report)
        self.assertEqual(set(images), set(CAMERAS))
        self.assertTrue(all(content.startswith(b'\x89PNG') for content in images.values()))
        record = export_record(report, 'origin', record_type='render')
        inventory = {'run_id': report['run_id'], 'storage_mode': 'job_local',
                     'storage_origin_verified': True}
        combined = export_inventory(inventory, 'origin') + '\n' + record
        self.assertEqual(parse_inventory_export(combined, report['run_id'], 'origin')['inventory'], inventory)
        self.assertEqual(parse_record(combined, report['run_id'], 'origin', record_type='render')['render'], report)

    def test_corrupt_crc_truncated_images_and_decompression_overrun_fail(self):
        image = png()
        for corrupted in (image[:-4], image[:-1] + bytes([image[-1] ^ 1]),
                          png(pixel_data=b'\0' * 1000000)):
            with self.subTest(length=len(corrupted)), self.assertRaises(ValueError):
                validate_png(corrupted)

    def test_chunked_export_is_bounded_and_requires_complete_committed_data(self):
        report = render_report()
        record = export_record(report, 'origin', record_type='render')
        lines = record.splitlines()
        self.assertGreater(len(lines), 2)
        self.assertTrue(all(len(line.encode()) < 4096 for line in lines))
        self.assertIsNone(parse_record('\n'.join(lines[:-1]), report['run_id'], 'origin', record_type='render'))
        with self.assertRaisesRegex(ValueError, 'missing chunks'):
            parse_record('\n'.join(lines[1:]), report['run_id'], 'origin', record_type='render')
        reordered = '\n'.join(reversed(lines)) + '\n' + lines[0]
        self.assertEqual(parse_record(reordered, report['run_id'], 'origin', record_type='render')['render'], report)
        corrupted = json.loads(lines[0].split(RECORD_PREFIXES['render'], 1)[1])
        content = bytearray(base64.b64decode(corrupted['data']))
        content[0] ^= 1
        corrupted['data'] = base64.b64encode(content).decode()
        lines[0] = RECORD_PREFIXES['render'] + json.dumps(corrupted)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            parse_record('\n'.join(lines), report['run_id'], 'origin', record_type='render')

    def test_render_cannot_claim_native_success_or_accept_incomplete_views(self):
        cases = []
        report = render_report()
        cases.append({**report, 'native_evaluation_ready': True})
        cases.append({**report, 'scene': 'RoboDojo'})
        cases.append({**report, 'frames': {'front': report['frames']['front']}})
        cases.append({**report, 'frames': {name: report['frames']['front'] for name in CAMERAS}})
        invalid = copy.deepcopy(report)
        invalid['frames']['front']['sha256'] = 'wrong'
        cases.append(invalid)
        invalid = copy.deepcopy(report)
        invalid['frames']['front']['stddev'] = 0
        cases.append(invalid)
        for case in cases:
            with self.subTest(scene=case['scene']), self.assertRaises(ValueError):
                validate_render_report(case)

    def test_failed_render_is_diagnostic_not_a_successful_artifact_set(self):
        report = render_report()
        report.update(render_passed=False, frames={}, error_type='RuntimeError')
        self.assertEqual(validate_render_report(report), {})
        del report['error_type']
        with self.assertRaises(ValueError):
            validate_render_report(report)

    def test_render_runtime_entry_refuses_local_execution(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('cluster_eval.runtime_job.execute_installation') as install, \
                self.assertRaises(ValueError):
            main()
        install.assert_not_called()

    def test_timeout_terminates_only_the_session_this_runner_created(self):
        process = Mock(pid=998001)
        process.wait.side_effect = [subprocess.TimeoutExpired('render', 1), 0]
        process.poll.return_value = None
        with patch('cluster_eval.runtime_job.subprocess.Popen', return_value=process) as launch, \
                patch('cluster_eval.runtime_job.os.killpg') as kill, \
                self.assertRaises(subprocess.TimeoutExpired):
            render_process(Path('/job_data/eai_test/runtime'), timeout=1)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(998001, signal.SIGTERM)

    def test_zero_exit_from_kit_does_not_hide_a_persisted_camera_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results').mkdir()
            (root / 'results/camera-failure.json').write_text(json.dumps({'error_type': 'ValueError'}))
            process = Mock()
            process.wait.return_value = 0
            with patch('cluster_eval.runtime_job.subprocess.Popen', return_value=process), \
                    self.assertRaisesRegex(RuntimeError, 'Isaac capture failed before shutdown'):
                render_process(root / 'runtime')

    def test_status_recovers_verified_images_without_printing_base64(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = 'render-test'
            run = root / 'runs/cluster_eval' / run_id
            run.mkdir(parents=True)
            output = root / 'output'
            output.mkdir()
            origin = b'origin-marker'
            digest = hashlib.sha256(origin).hexdigest()
            (output / 'origin.json').write_bytes(origin)
            receipt = {'job_id': 'own-job', 'job_name': 'eai_render', 'queue': 'own-queue',
                       'cluster': 'bcloud-bj-zone1', 'profile': 'render', 'storage_mode': 'job_local',
                       'origin_sha256': digest, 'result_dir': str(output),
                       'accepted_at': '2026-10-30T04:00:00Z'}
            (run / 'receipt.json').write_text(json.dumps(receipt))
            inventory = {'run_id': run_id, 'storage_mode': 'job_local', 'storage_origin_verified': True,
                         'single_5090_visible': True, 'inventory_passed': True}
            rendered = render_report(run_id)
            manager = Mock()
            manager.get_job.return_value = SimpleNamespace(job_id='own-job', job_name='eai_render',
                queue_name='own-queue', status='Succeeded', finish_time='2026-10-30 13:00:00')
            manager.get_job_logs.return_value = SimpleNamespace(content=(
                export_inventory(inventory, digest) + '\n' + export_record(rendered, digest, record_type='render')))
            with patch('cluster_eval.status.ROOT', root), patch.object(Path, 'is_relative_to', return_value=True):
                report = collect_status(run, manager)
            self.assertTrue(report['render_export_verified'])
            self.assertTrue(report['inventory_export_verified'])
            manager.get_job_logs.assert_called_once()
            self.assertNotIn('png_base64', json.dumps(report))
            self.assertFalse(report['native_evaluation_ready'])
            self.assertTrue(summarize([report])['jobs'][0]['render_verified'])
            for name, content in validate_render_report(rendered).items():
                self.assertEqual((output / f'render_{name}.png').read_bytes(), content)
            report['inventory_export_verified'] = False
            self.assertFalse(summarize([report])['jobs'][0]['render_verified'])


if __name__ == '__main__':
    unittest.main()
