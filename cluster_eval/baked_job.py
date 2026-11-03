"""Render from an existing image with private caches and read-only asset checks."""

from __future__ import annotations

import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from cluster_eval import probe
from cluster_eval.image_inventory import SIM_PYTHON, SOURCE_ROOT
from cluster_eval.rendering import validate_render_report
from cluster_eval.runtime_install import RUNTIME_PATHS
from cluster_eval.runtime_job import render_process
from cluster_eval.storage import export_record, local_result_path, persist_export, validate_local_destination


ASSET_DATASET = Path('/bucket/input/robot_lab/assets/RoboDojo')


def inspect_assets(dataset: Path = ASSET_DATASET, *, full_scan: bool = True) -> dict:
    lock = json.loads((ROOT / 'configs/robodojo_versions.json').read_text())['assets']
    assets = dataset / 'Assets'
    if not assets.is_dir() and (dataset / 'Object').is_dir() and (dataset / 'Robots').is_dir():
        assets = dataset
    report = {'dataset': str(dataset), 'assets_exists': assets.is_dir(),
              'assets_path': str(assets),
              'expected_revision': lock['revision'], 'content_identity_verified': False,
              'count_and_size_match': False, 'files': 0, 'bytes': 0}
    if not assets.is_dir():
        if dataset.is_dir():
            with os.scandir(dataset) as entries:
                report['top_level_names'] = sorted(entry.name for entry in islice(entries, 40))
        return report
    if not full_scan:
        report.update(files=None, bytes=None, full_scan_performed=False,
                      object_directory=(assets / 'Object').is_dir(), robot_directory=(assets / 'Robots').is_dir())
        return report
    report['full_scan_performed'] = True
    for directory, subdirectories, files in os.walk(assets, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in subdirectories):
            raise ValueError('Asset inspection refuses symlinks across the read-only dataset boundary')
        for name in files:
            metadata = (Path(directory) / name).lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError('Asset inventory requires regular files without symlinks')
            report['files'] += 1
            report['bytes'] += metadata.st_size
            if report['files'] % 1000 == 0:
                print(json.dumps({'event': 'asset_inventory_progress', 'files': report['files'],
                                  'bytes': report['bytes']}), flush=True)
            if report['files'] > lock['file_count'] + 100:
                report['scan_stopped_early'] = True
                return report
    report['count_and_size_match'] = report['files'] == lock['file_count'] and report['bytes'] == lock['total_bytes']
    return report


def prepare_runtime(runtime: Path, inventory: dict) -> None:
    validate_local_destination(runtime, ROOT)
    packages = inventory.get('baked_image', {}).get('runtime', {}).get('packages', {})
    if (inventory.get('inventory_passed') is not True or inventory.get('single_5090_visible') is not True
            or packages.get('isaacsim') != '5.1.0.0' or packages.get('torch') != '2.7.0+cu128'
            or not SIM_PYTHON.is_file()):
        raise ValueError('Use the actually inspected one-GPU image runtime, not an unverified environment')
    runtime.mkdir(mode=0o700)
    for suffix in RUNTIME_PATHS.values():
        (runtime / suffix).mkdir(mode=0o700, parents=True, exist_ok=True)


def main() -> None:
    probe.main(check_network=False)
    results = local_result_path(os.environ['EAI_RUN_ID'])
    runtime = results.parent / 'runtime'
    inventory = json.loads((results / 'inventory.json').read_text())
    report = {'run_id': os.environ['EAI_RUN_ID'], 'storage_mode': 'job_local',
              'storage_origin_verified': True, 'render_passed': False,
              'scene': 'procedural_rgb_cubes', 'native_evaluation_ready': False,
              'runtime_source': 'prebuilt_image', 'installers_executed': False}
    try:
        prepare_runtime(runtime, inventory)
        print(json.dumps({'event': 'baked_stage', 'stage': 'read_only_asset_inventory'}), flush=True)
        try:
            report['assets'] = inspect_assets(full_scan=False)
        except OSError as error:
            report['assets'] = {'error_type': type(error).__name__, 'content_identity_verified': False}
        relative = 'task/RoboDojo/tasks/general_pickup.py'
        source = SOURCE_ROOT / relative
        report['general_pickup_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        print(json.dumps({'event': 'baked_stage', 'stage': 'isaac_three_camera_render', 'assets': report['assets']}), flush=True)
        render_process(runtime, timeout=900, python=SIM_PYTHON)
        captured = json.loads((results / 'camera-capture.json').read_text())
        if (captured.get('run_id') != os.environ['EAI_RUN_ID'] or captured.get('storage_mode') != 'job_local'
                or captured.get('storage_origin_verified') is not True):
            raise ValueError('Capture belongs to another job')
        validate_render_report(captured)
        report.update(captured)
    except Exception as error:
        report.update(render_passed=False, error_type=type(error).__name__, frames={})
    finally:
        failure = results / 'camera-failure.json'
        if failure.is_file():
            report['camera_failure'] = json.loads(failure.read_text())
        persist_export(results / 'render.json', report)
        print(export_record(report, os.environ['EAI_ORIGIN_SHA256'], record_type='render'), flush=True)
    if not report['render_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
