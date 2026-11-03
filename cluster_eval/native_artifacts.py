"""Recover a bounded native episode audit; procedural renders cannot pass it."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import struct

from cluster_eval.rendering import validate_png
from scripts.download_qwen35 import MODEL, REVISION


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
FILES = frozenset(('_result.json', 'summary.json', 'trace.jsonl', 'experiment.json',
                   'schedule.json', 'model-check.json', 'model-download.json')
                  + tuple(f'cameras/{camera}.png' for camera in CAMERAS))
MAX_FILE_BYTES = 4 * 1024 ** 2
MAX_TOTAL_BYTES = 10 * 1024 ** 2


def encode_files(paths: dict[str, Path], *, complete: bool = True) -> dict:
    if not set(paths).issubset(FILES) or complete and set(paths) != FILES:
        raise ValueError('A complete native audit requires the fixed artifact set')
    files = {}
    total = 0
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
            raise ValueError(f'Missing, linked or oversized native artifact: {name}')
        content = path.read_bytes()
        total += len(content)
        if total > MAX_TOTAL_BYTES:
            raise ValueError('Native audit exceeds the bounded log export budget')
        files[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                       'base64': base64.b64encode(content).decode()}
    return files


def decode_files(encoded: dict, *, complete: bool = True) -> dict[str, bytes]:
    if (not isinstance(encoded, dict) or not set(encoded).issubset(FILES)
            or complete and set(encoded) != FILES):
        raise ValueError('Native result, harness trace, endpoint evidence and camera images are all required')
    files = {}
    total = 0
    for name, record in encoded.items():
        if (not isinstance(record, dict) or type(record.get('bytes')) is not int
                or not 0 < record['bytes'] <= MAX_FILE_BYTES
                or not isinstance(record.get('base64'), str)
                or len(record['base64']) > 4 * ((MAX_FILE_BYTES + 2) // 3)):
            raise ValueError('Native file encoding exceeds the reviewed bounds')
        content = base64.b64decode(record['base64'], validate=True)
        total += len(content)
        if (len(content) != record['bytes'] or total > MAX_TOTAL_BYTES
                or hashlib.sha256(content).hexdigest() != record.get('sha256')):
            raise ValueError('Native artifact size or checksum mismatch')
        files[name] = content
    return files


def validate_native_report(report: dict) -> tuple[dict[str, bytes], dict]:
    if report.get('benchmark') != 'RoboDojo' or report.get('task_id') != 'general_pickup':
        raise ValueError('Only the reviewed native RoboDojo task can pass this gate')
    if report.get('episode_complete') is not True:
        if (report.get('native_evaluation_ready') is not False or report.get('native_success') is not False
                or not isinstance(report.get('error_type'), str) or not report['error_type']):
            raise ValueError('Incomplete native jobs require an explicit failure, never a success claim')
        return decode_files(report.get('diagnostic_files', {}), complete=False), {
            'episode_complete': False, 'native_success': False}
    encoded = report.get('files', {})
    files = decode_files(encoded)
    result = json.loads(files['_result.json'])
    summary = json.loads(files['summary.json'])
    experiment = json.loads(files['experiment.json'])
    schedule = json.loads(files['schedule.json'])
    model = json.loads(files['model-download.json'])
    endpoint = json.loads(files['model-check.json'])
    details = result.get('details', {})
    if not isinstance(details, dict) or set(details) != {'0'}:
        raise ValueError('Exactly one original native episode must be scored')
    detail = details['0']
    if (type(detail.get('success')) is not bool or type(detail.get('layout_id')) is not int
            or type(detail.get('score')) not in (int, float) or not math.isfinite(detail['score'])):
        raise ValueError('Invalid original native score')
    if (not isinstance(schedule, list) or len(schedule) != 1
            or schedule[0].get('condition') != 'q_only' or schedule[0].get('task_id') != 'general_pickup'
            or schedule[0].get('seed') != 0 or schedule[0].get('eval_num') != 1
            or schedule[0].get('action_mode') != 'ee_delta'
            or schedule[0].get('native_run_id') != report['run_id'] + '_q_only_general_pickup_s0'):
        raise ValueError('Native task identity differs from the frozen smoke schedule')
    if (summary.get('task_id') != 'general_pickup' or summary.get('condition') != 'q_only'
            or summary.get('seed') != 0 or summary.get('layout_id') != detail['layout_id']
            or summary.get('native_result_index') != 0 or summary.get('native_terminal') is not True
            or summary.get('native_success') is not detail['success']
            or not str(summary.get('episode_id', '')).startswith(schedule[0]['native_run_id'] + '-')
            or summary.get('status') in (None, 'error')
            or type(summary.get('executed_physical_steps')) is not int
            or summary['executed_physical_steps'] < 1):
        raise ValueError('Harness evidence does not match the native episode or shows no physical actions')
    if (model.get('model') != MODEL or model.get('revision') != REVISION
            or type(model.get('weight_bytes')) is not int or model['weight_bytes'] < 50 * 1024 ** 3
            or experiment.get('models', {}).get('qwen', {}).get('model') != MODEL
            or experiment.get('episode_id') != summary['episode_id']
            or experiment.get('condition') != 'q_only'):
        raise ValueError('Missing pinned BF16 Qwen model provenance')
    checks = endpoint.get('checks', [])
    expected_checks = {'text_json', 'three_images', 'native_function_call', 'harness_planner_actor'}
    if (endpoint.get('model') != MODEL or endpoint.get('passed') is not True or not isinstance(checks, list)
            or not expected_checks.issubset({item.get('name') for item in checks if item.get('passed') is True})
            or any(item.get('passed') is not True for item in checks)):
        raise ValueError('Qwen endpoint checks are missing or failed')
    trace = [json.loads(line) for line in files['trace.jsonl'].decode().splitlines() if line.strip()]
    if not trace or any(event.get('episode_id') != summary['episode_id'] for event in trace):
        raise ValueError('Trace belongs to another episode')
    calls = [event for event in trace if event.get('event') == 'model_call' and event.get('ok') is True]
    if (not {'planner', 'actor'}.issubset({event.get('role') for event in calls})
            or any(event.get('model') != MODEL for event in trace if event.get('event') == 'model_call')
            or any(event.get('event') == 'advisor_call' for event in trace)):
        raise ValueError('The actual harness must call Qwen as planner and actor, not a closed-source substitute')
    steps = sum(event.get('receipt', {}).get('executed_steps', 0) or 0
                for event in trace if event.get('event') == 'tool')
    if steps != summary['executed_physical_steps']:
        raise ValueError('Physical execution count disagrees with the harness trace')
    for camera in CAMERAS:
        name = f'cameras/{camera}.png'
        content = files[name]
        if len(content) < 33:
            raise ValueError('Truncated native camera frame')
        width, height = struct.unpack('>II', content[16:24])
        validate_png(content, width=width, height=height, max_bytes=MAX_FILE_BYTES)
        camera_meta = summary.get('camera_artifacts', {}).get(camera, {})
        if camera_meta.get('path') != name or camera_meta.get('sha256') != encoded[name]['sha256']:
            raise ValueError('Recovered camera image differs from the harness observation')
    if len({files[f'cameras/{camera}.png'] for camera in CAMERAS}) < 2:
        raise ValueError('Native camera views are all identical')
    verdict = {'episode_complete': True, 'native_success': detail['success'],
               'score': detail['score'], 'layout_id': detail['layout_id'],
               'executed_physical_steps': steps, 'qwen_calls': len(calls)}
    if (report.get('native_success') is not verdict['native_success']
            or report.get('native_evaluation_ready') is not True):
        raise ValueError('Native verdict disagrees with original evidence')
    return files, verdict
