"""Frozen single-GPU installation and rendering job; no Qwen or robot actions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from cluster_eval import probe
from cluster_eval.rendering import validate_render_report
from cluster_eval.runtime_install import execute_installation, runtime_environment
from cluster_eval.storage import export_record, local_result_path, persist_export


def render_process(runtime: Path, *, timeout: int = 900, python: Path | None = None) -> None:
    process = subprocess.Popen([str(python or runtime / 'sim/bin/python'), '-B', '-m', 'cluster_eval.isaac_render'],
        cwd=ROOT, env=runtime_environment(runtime), start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        raise
    if returncode != 0:
        raise RuntimeError(f'Isaac render subprocess exited with code {returncode}')
    failure = runtime.parent / 'results/camera-failure.json'
    if failure.is_file():
        detail = json.loads(failure.read_text())
        raise RuntimeError(f'Isaac capture failed before shutdown: {detail.get("error_type")}')


def main() -> None:
    probe.main()
    results = local_result_path(os.environ['EAI_RUN_ID'])
    runtime = results.parent / 'runtime'
    report = {'run_id': os.environ['EAI_RUN_ID'], 'storage_mode': 'job_local',
              'storage_origin_verified': True, 'render_passed': False,
              'scene': 'procedural_rgb_cubes', 'native_evaluation_ready': False}
    try:
        execute_installation(runtime, max_minutes=90)
        render_process(runtime)
        captured = json.loads((results / 'camera-capture.json').read_text())
        if (captured.get('run_id') != os.environ['EAI_RUN_ID']
                or captured.get('storage_mode') != 'job_local'
                or captured.get('storage_origin_verified') is not True):
            raise ValueError('Camera capture belongs to a different runtime')
        validate_render_report(captured)
        report.update(captured)
    except Exception as error:
        report.update(render_passed=False, error_type=type(error).__name__, frames={})
    finally:
        failure = results / 'camera-failure.json'
        if failure.is_file():
            report['camera_failure'] = json.loads(failure.read_text())
        installation = runtime / 'installation-result.json'
        if installation.is_file():
            report['installation'] = json.loads(installation.read_text())
        report['packages'] = {}
        for name in ('sim', 'bridge'):
            manifest = runtime / f'{name}-packages.json'
            if manifest.is_file():
                report['packages'][name] = json.loads(manifest.read_text())
        persist_export(results / 'render.json', report)
        print(export_record(report, os.environ['EAI_ORIGIN_SHA256'], record_type='render'), flush=True)
    if report['render_passed'] is not True:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
