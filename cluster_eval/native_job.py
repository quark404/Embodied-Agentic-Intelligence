"""Three allocated 5090s: private setup, Qwen BF16 and one native harness episode."""

from __future__ import annotations

from datetime import datetime, timezone
import base64
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from cluster_eval import probe
from cluster_eval import baked_runtime
from cluster_eval.native_artifacts import CAMERAS, encode_files, validate_native_report
from cluster_eval.rendering import validate_render_report
from cluster_eval.runtime_install import execute_installation, runtime_environment
from cluster_eval.runtime_job import render_process
from cluster_eval.storage import export_record, local_result_path, persist_export, validate_local_destination
from embodied_agentic.schedule import make_schedule, native_result_dir
from scripts.download_qwen35 import MODEL, REVISION


def stop_owned_group(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait(timeout=20)


def command_step(stage: str, command: list[str], environment: dict, *, deadline: float,
                 max_seconds: int, report: dict, cwd: Path = ROOT) -> None:
    report['phase'] = stage
    timeout = min(max_seconds, int(deadline - time.monotonic()))
    if timeout < 1:
        raise TimeoutError('Native job execution budget exhausted before the next stage')
    print(json.dumps({'event': 'native_stage', 'stage': stage,
                      'started_at': datetime.now(timezone.utc).isoformat()}), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=environment, start_new_session=True)
    result = {'stage': stage, 'returncode': None}
    try:
        result['returncode'] = process.wait(timeout=timeout)
        if result['returncode'] != 0:
            raise RuntimeError(f'Native stage failed: {stage}, exit={result["returncode"]}')
    finally:
        stop_owned_group(process)
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report['stages'].append(result)


def qwen_environment(runtime: Path) -> dict:
    environment = (baked_runtime.baked_environment(runtime)
                   if os.environ.get('EAI_RUNTIME_SOURCE') == 'prebuilt_image' else runtime_environment(runtime))
    environment.update(CUDA_VISIBLE_DEVICES='1,2', QWEN_GPU='1,2',
        QWEN_ENV=str(runtime / 'qwen'), QWEN_MAX_MODEL_LEN='16384', QWEN_MAX_NUM_SEQS='1',
        HF_HUB_CACHE=str(runtime / 'model-cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VLLM_CACHE_ROOT=str(runtime / 'cache/qwen_vllm'),
        TORCH_EXTENSIONS_DIR=str(runtime / 'cache/qwen_torch_extensions'),
        CUDA_CACHE_PATH=str(runtime / 'cache/qwen_cuda'))
    return environment


def qwen_install_plan(runtime: Path, *, prebuilt: bool = False,
                      preinstalled: bool = False) -> list[tuple[str, list[str]]]:
    if preinstalled and not prebuilt:
        raise ValueError('Image environment reuse requires the verified prebuilt-image path')
    uv = str(runtime / 'bootstrap/bin/uv')
    python = str(runtime / 'qwen/bin/python')
    selected_python = str(baked_runtime.SIM_PYTHON) if prebuilt else '3.11.14'
    cuda_version = '12.8' if prebuilt else '12.9'
    requirements = 'qwen35-serving-cu128.txt' if prebuilt else 'qwen35-serving-cu129.txt'
    index = ['--index-url', baked_runtime.INTERNAL_INDEX] if prebuilt else []
    plan = [
        ('qwen_venv', [uv, 'venv', '--python', selected_python, '--seed', str(runtime / 'qwen')]),
        ('qwen_dependencies', [uv, 'pip', 'install', '--python', python,
                               *index, '-r', str(ROOT / 'requirements' / requirements)]),
        ('qwen_dependency_check', [python, '-m', 'pip', 'check']),
        ('qwen_cuda_check', [python, '-c',
            f'import torch; assert torch.version.cuda == "{cuda_version}"; '
            'assert torch.cuda.device_count() == 2; '
            'assert all(torch.cuda.get_device_capability(index) == (12,0) for index in range(2)); '
            'print([torch.ones(8,device=f"cuda:{index}").sum().item() for index in range(2)])']),
    ]
    return plan[2:] if preinstalled else plan


def choose_port() -> int:
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def wait_model(process: subprocess.Popen, port: int, *, deadline: float) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Owned Qwen service exited before readiness')
        try:
            with urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as response:
                models = json.load(response)
            if any(item.get('id') == MODEL for item in models.get('data', [])):
                return
        except (OSError, ValueError):
            pass
        time.sleep(5)
    raise TimeoutError('Qwen service did not become ready inside its startup budget')


def native_environment(runtime: Path, config: Path, row: dict) -> dict:
    environment = runtime_environment(runtime)
    environment.update(EMBODIED_EXPERIMENT_CONFIG=str(config), EMBODIED_CONDITION='q_only',
        EMBODIED_ROBODOJO_CONTROL_MODE='ee_delta', EVAL_ENV_TYPE='sim',
        ROBODOJO_RUN_ID=row['native_run_id'], EVAL_NUM='1', ROBODOJO_MAX_BASH_RETRIES='0',
        ROBODOJO_FATAL_RESTART_COUNT='3', ROBOICL_DATA_ROOT=str(runtime / 'assets'),
        ROBOICL_RESULTS_ROOT=str(runtime / 'native-results'),
        PYTHONPATH=str(ROOT))
    if os.environ.get('EAI_RUNTIME_SOURCE') == 'prebuilt_image':
        environment['ROBODOJO_KIT_ARGS'] = f'--portable --portable-root {runtime / "kit-native"}'
        environment['ROBODOJO_ENV_CONFIG_DIR'] = str(runtime / 'RoboDojo/env_cfg')
    environment['PATH'] = str(runtime / 'bin') + os.pathsep + environment.get('PATH', '')
    if os.getuid() == 0:
        environment['OMNI_KIT_ALLOW_ROOT'] = '1'
    return environment


def prepare_video_binary(runtime: Path, *, prebuilt: bool = False) -> None:
    validate_local_destination(runtime, ROOT)
    python = str(runtime / 'sim/bin/python')
    target = Path(subprocess.check_output([python, '-c',
        'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],
        env=runtime_environment(runtime), text=True, timeout=30).strip()).resolve()
    allowed = target.is_relative_to(runtime / 'sim') or (
        prebuilt and target.is_relative_to(baked_runtime.SIM_PYTHON.parent.parent))
    if not target.is_file() or not allowed or not os.access(target, os.X_OK):
        raise ValueError('Use only the FFmpeg executable bundled inside this private simulator environment')
    tools = runtime / 'bin'
    tools.mkdir(mode=0o700)
    (tools / 'ffmpeg').symlink_to(target)
    subprocess.run([str(tools / 'ffmpeg'), '-version'], check=True, timeout=15,
                   env=runtime_environment(runtime), stdout=subprocess.DEVNULL)


def collect_episode(runtime: Path, results: Path, row: dict, *, complete: bool = True) -> dict:
    native = native_result_dir(runtime / 'native-results', row)
    summaries = list((native / 'embodied_agentic').glob('*/summary.json'))
    if len(summaries) > 1 or complete and len(summaries) != 1:
        raise ValueError('Expected exactly one native harness episode summary')
    paths = {'_result.json': native / '_result.json', 'schedule.json': runtime / 'schedule.json',
             'model-check.json': results / 'model-check.json',
             'model-download.json': results / 'model-download.json'}
    if summaries:
        episode = summaries[0].parent
        paths.update({name: episode / name for name in ('summary.json', 'trace.jsonl', 'experiment.json')})
        paths.update({f'cameras/{camera}.png': episode / f'cameras/{camera}.png' for camera in CAMERAS})
    if not complete:
        paths = {name: path for name, path in paths.items() if path.is_file()}
    return encode_files(paths, complete=complete)


def main() -> None:
    if os.environ.get('EAI_EXPECTED_GPUS') != '3' or os.environ.get('EAI_STORAGE_MODE') != 'job_local':
        raise ValueError('Native entry requires the explicit three-GPU job-local profile')
    prebuilt = os.environ.get('EAI_RUNTIME_SOURCE') == 'prebuilt_image'
    probe.main(check_network=not prebuilt)
    results = local_result_path(os.environ['EAI_RUN_ID'])
    runtime = validate_local_destination(results.parent / 'runtime', ROOT)
    inventory = json.loads((results / 'inventory.json').read_text())
    if (inventory.get('gpu_topology_verified') is not True
            or sorted(device['index'] for device in inventory['gpus']) != [0, 1, 2]):
        raise ValueError('The reviewed native topology requires exactly three visible indices 0,1,2')
    report = {'run_id': os.environ['EAI_RUN_ID'], 'storage_mode': 'job_local',
              'storage_origin_verified': True, 'benchmark': 'RoboDojo', 'task_id': 'general_pickup',
              'model': MODEL, 'model_revision': REVISION, 'episode_complete': False,
              'native_evaluation_ready': False, 'native_success': False, 'stages': []}
    deadline = time.monotonic() + 220 * 60
    service = None
    rows = None
    try:
        report['phase'] = 'simulation_installation'
        if shutil.disk_usage(runtime.parent).free < 240 * 1024 ** 3:
            raise ValueError('Native job needs 240 GiB free for both runtimes, weights and assets')
        if prebuilt:
            report['baked_runtime'] = baked_runtime.setup(runtime, inventory)
        else:
            execute_installation(runtime, max_minutes=90)
        report['phase'] = 'video_binary'
        prepare_video_binary(runtime, prebuilt=prebuilt)
        command_step('bridge_handshake', [str(runtime / 'bridge/bin/python'), '-B', '-m',
            'scripts.check_xpolicylab_bridge', '--robodojo-root', str(runtime / 'RoboDojo'),
            '--server-python', str(runtime / 'bridge/bin/python'),
            '--client-python', str(runtime / 'sim/bin/python'), '--output', str(runtime / 'bridge-check')],
            runtime_environment(runtime), deadline=deadline, max_seconds=300, report=report)
        report['phase'] = 'isaac_rgb_rendering'
        render_process(runtime)
        captured = json.loads((results / 'camera-capture.json').read_text())
        if captured.get('run_id') != report['run_id']:
            raise ValueError('Procedural capture identity mismatch')
        validate_render_report(captured)
        if captured.get('render_passed') is not True:
            raise ValueError('Cannot load Qwen before this job passes actual rendering')
        persist_export(results / 'render.json', captured)
        print(export_record(captured, os.environ['EAI_ORIGIN_SHA256'], record_type='render'), flush=True)
        qwen_env = qwen_environment(runtime)
        preinstalled = report.get('baked_runtime', {}).get('preinstalled_runtime') is not None
        for stage, command in qwen_install_plan(runtime, prebuilt=prebuilt, preinstalled=preinstalled):
            command_step(stage, command, qwen_env, deadline=deadline, max_seconds=2700, report=report)
        packages = subprocess.check_output([str(runtime / 'qwen/bin/python'), '-m', 'pip', 'freeze'],
                                           env=qwen_env, text=True, timeout=60)
        report['qwen_packages'] = packages.splitlines()
        report['sim_packages'] = json.loads((runtime / 'sim-packages.json').read_text())
        report['bridge_packages'] = json.loads((runtime / 'bridge-packages.json').read_text())
        download_env = {**qwen_env, 'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0'}
        if not prebuilt:
            command_step('assets', [str(runtime / 'qwen/bin/python'), '-B', '-m', 'cluster_eval.native_assets',
                '--destination', str(runtime / 'assets'), '--manifest', str(runtime / 'asset-manifest.json')],
                download_env, deadline=deadline, max_seconds=3600, report=report)
        command_step('model_weights', [str(runtime / 'qwen/bin/python'), '-B', '-m', 'scripts.download_qwen35',
            '--cache-dir', str(runtime / 'model-cache'), '--storage-mode', 'job_local',
            '--source', 'modelscope' if prebuilt else 'huggingface'],
            download_env, deadline=deadline, max_seconds=3600, report=report)
        port = choose_port()
        report['phase'] = 'model_startup'
        qwen_env['QWEN_PORT'] = str(port)
        service = subprocess.Popen(['bash', str(ROOT / 'scripts/serve_qwen35.sh')],
            cwd=ROOT, env=qwen_env, start_new_session=True)
        wait_model(service, port, deadline=min(deadline, time.monotonic() + 1200))
        command_step('qwen_endpoint', [str(runtime / 'qwen/bin/python'), '-B', '-m', 'scripts.check_qwen35',
            '--base-url', f'http://127.0.0.1:{port}/v1', '--output', str(results / 'model-check.json')],
            qwen_env, deadline=deadline, max_seconds=900, report=report)
        config = json.loads((ROOT / 'configs/robodojo_qwen35_27b.local.example.json').read_text())
        config['qwen'].update(base_url=f'http://127.0.0.1:{port}/v1',
                              max_output_tokens=2048, timeout_seconds=180)
        config['harness']['max_seconds'] = 2400
        config_path = runtime / 'native-config.json'
        persist_export(config_path, config)
        rows = make_schedule(robodojo_root=runtime / 'RoboDojo', experiment_config=config_path,
            task_manifest=ROOT / 'configs/robodojo_smoke.example.json',
            policy_env=str(runtime / 'bridge'), sim_env=str(runtime / 'sim'),
            run_id=report['run_id'], action_mode='ee_delta', conditions=('q_only',))
        persist_export(runtime / 'schedule.json', rows)
        command_step('native_episode', rows[0]['command'], native_environment(runtime, config_path, rows[0]),
            deadline=deadline, max_seconds=3600, report=report, cwd=runtime / 'RoboDojo')
        report['phase'] = 'native_artifact_verification'
        report['files'] = collect_episode(runtime, results, rows[0])
        original = json.loads(base64.b64decode(report['files']['_result.json']['base64'], validate=True))
        report.update(episode_complete=True, native_evaluation_ready=True,
                      native_success=original['details']['0']['success'])
        _, verdict = validate_native_report(report)
        report['verdict'] = verdict
        report['phase'] = 'native_episode_complete'
    except Exception as error:
        report.update(episode_complete=False, native_evaluation_ready=False,
                      native_success=False, error_type=type(error).__name__)
        report.pop('files', None)
        if type(error) is ValueError:
            report['validation_error'] = str(error)[:512]
        if rows:
            try:
                report['diagnostic_files'] = collect_episode(runtime, results, rows[0], complete=False)
            except Exception as diagnostic_error:
                report['diagnostic_error_type'] = type(diagnostic_error).__name__
    finally:
        if service is not None:
            stop_owned_group(service)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        persist_export(results / 'native.json', report)
        print(export_record(report, os.environ['EAI_ORIGIN_SHA256'], record_type='native'), flush=True)
    if not report['episode_complete'] or not report['native_success']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
