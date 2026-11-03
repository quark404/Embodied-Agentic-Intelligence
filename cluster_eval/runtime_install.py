"""Plan or execute isolated RoboDojo installation inside a verified cluster job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cluster_eval.storage import (
    BOOTSTRAP_WHEEL_URL, PUBLIC_DEPENDENCY_URLS, local_result_path, persist_export, validate_local_destination,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = {'HOME': 'home', 'XDG_CACHE_HOME': 'cache/xdg', 'UV_CACHE_DIR': 'cache/uv',
                 'UV_PYTHON_INSTALL_DIR': 'python', 'PIP_CACHE_DIR': 'cache/pip',
                 'TMPDIR': 'tmp', 'HF_HOME': 'cache/huggingface', 'TORCH_HOME': 'cache/torch',
                 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions', 'CUDA_CACHE_PATH': 'cache/cuda'}


def runtime_environment(runtime: Path) -> dict:
    environment = dict(os.environ)
    bypass = list(dict.fromkeys(
        host.strip() for name in ('no_proxy', 'NO_PROXY')
        for host in environment.get(name, '').split(',') if host.strip()))
    bypass = list(dict.fromkeys([*bypass, 'localhost', '127.0.0.1']))
    environment.update(no_proxy=','.join(bypass), NO_PROXY=','.join(bypass))
    environment.update({variable: str(runtime / suffix) for variable, suffix in RUNTIME_PATHS.items()})
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       PIP_DISABLE_PIP_VERSION_CHECK='1', UV_NO_PROGRESS='1',
                       UV_PYTHON_PREFERENCE='only-managed', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                       OMNI_KIT_ACCEPT_EULA='YES', CUDA_VISIBLE_DEVICES='0',
                       TORCH_CUDA_ARCH_LIST='12.0')
    return environment


def installation_plan(runtime: Path, *, package: Path = ROOT, python: str = sys.executable) -> list[dict]:
    runtime = runtime.absolute()
    bootstrap = runtime / 'bootstrap'
    uv = str(bootstrap / 'bin/uv')
    sim_python = str(runtime / 'sim/bin/python')
    bridge_python = str(runtime / 'bridge/bin/python')
    constraints = str(package / 'requirements/robodojo-runtime-constraints.txt')
    source = runtime / 'RoboDojo'
    plan = [
        ('bootstrap_venv', [python, '-m', 'venv', str(bootstrap)]),
        ('isolated_uv', [str(bootstrap / 'bin/python'), '-m', 'pip', 'install',
                         '--index-url', 'https://pypi.org/simple', 'uv==0.9.3']),
        ('managed_python', [uv, 'python', 'install', '3.11.14', '--no-bin']),
        ('simulation_venv', [uv, 'venv', '--python', '3.11.14', '--seed', str(runtime / 'sim')]),
        ('bridge_venv', [uv, 'venv', '--python', '3.11.14', '--seed', str(runtime / 'bridge')]),
        ('simulation_torch', [uv, 'pip', 'install', '--python', sim_python,
                               'torch==2.7.0', 'torchvision==0.22.0', 'torchaudio==2.7.0',
                               '--index-url', 'https://download.pytorch.org/whl/cu128']),
        ('simulation_dependencies', [uv, 'pip', 'install', '--python', sim_python, '-c', constraints,
                                      '-r', str(package / 'requirements/robodojo-sim.txt'),
                                      '--extra-index-url', 'https://pypi.nvidia.com']),
        ('bridge_dependencies', [uv, 'pip', 'install', '--python', bridge_python,
                                  '-r', str(package / 'requirements/robodojo-bridge.txt')]),
        ('pinned_sources', [python, '-B', '-m', 'scripts.prepare_robodojo_sources', str(source),
                            '--sim-submodules', '--policy-root', str(package)]),
    ]
    for name in ('isaaclab', 'isaaclab_assets', 'isaaclab_tasks'):
        plan.append((name, [uv, 'pip', 'install', '--python', sim_python, '--no-build-isolation',
                            '-c', constraints, '-e', str(source / 'third_party/IsaacLab/source' / name)]))
    plan.extend([
        ('curobo', [uv, 'pip', 'install', '--python', sim_python, '--no-build-isolation',
                    '-c', constraints, '-e', str(source / 'third_party/curobo') + '[cu12]']),
        ('cuda_backend_check', [sim_python, '-c',
                                'from cuda.core import Device, LaunchConfig, Program, ProgramOptions; '
                                'Device(0).set_current(); '
                                'import torch; '
                                'assert torch.cuda.is_available(); '
                                'assert torch.cuda.get_device_capability(0) == (12, 0); '
                                'print(torch.ones(8, device="cuda").sum().item())']),
        ('simulation_dependency_check', [sim_python, '-m', 'pip', 'check']),
        ('bridge_dependency_check', [bridge_python, '-m', 'pip', 'check']),
    ])
    return [{'stage': name, 'argv': argv} for name, argv in plan]


def execute_installation(runtime: Path, *, max_minutes: int = 90) -> dict:
    runtime = validate_local_destination(runtime, ROOT)
    inventory = json.loads((local_result_path(os.environ['EAI_RUN_ID']) / 'inventory.json').read_text())
    if (inventory.get('run_id') != os.environ['EAI_RUN_ID']
            or inventory.get('storage_mode') != 'job_local'
            or inventory.get('inventory_passed') is not True
            or not (inventory.get('single_5090_visible') is True
                    or os.environ.get('EAI_EXPECTED_GPUS') == '3'
                    and inventory.get('expected_gpus') == 3
                    and inventory.get('gpu_topology_verified') is True)):
        raise ValueError('This job must pass its own GPU and storage preflight before installation')
    network = inventory.get('network', {})
    if any(network.get(url, {}).get('reachable') is not True
           and not isinstance(network.get(url, {}).get('status'), int)
           for url in PUBLIC_DEPENDENCY_URLS):
        raise ValueError('Public dependency endpoints are unreachable for this online-install profile; '
                         'use a verified prebuilt image/cache or a reachable resource source')
    if network.get(BOOTSTRAP_WHEEL_URL, {}).get('status') != 200:
        raise ValueError('Bootstrap wheel download endpoint failed; an accessible package index alone is insufficient')
    if type(max_minutes) is not int or not 1 <= max_minutes <= 90:
        raise ValueError('Installation budget must be 1..90 minutes')
    if runtime.exists() or runtime.is_symlink():
        raise FileExistsError('Refusing to modify any existing runtime environment')
    if shutil.disk_usage(runtime.parent).free < 80 * 1024 ** 3:
        raise ValueError('Installation requires at least 80 GiB free in this job storage')
    runtime.mkdir(mode=0o700)
    environment = runtime_environment(runtime)
    for suffix in RUNTIME_PATHS.values():
        directory = runtime / suffix
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    plan = installation_plan(runtime)
    persist_export(runtime / 'installation-plan.json', {'stages': plan})
    deadline = time.monotonic() + max_minutes * 60
    report = {'run_id': os.environ['EAI_RUN_ID'], 'runtime': str(runtime), 'stages': [],
              'installation_complete': False, 'native_evaluation_ready': False}
    previous_umask = os.umask(0o077)
    try:
        for step in plan:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Installation time budget exhausted')
            if shutil.disk_usage(runtime).free < 16 * 1024 ** 3:
                raise ValueError('Stopping installation before job storage is exhausted')
            print(json.dumps({'event': 'runtime_install_stage', 'stage': step['stage']}), flush=True)
            started = time.monotonic()
            result = subprocess.run(step['argv'], cwd=ROOT, env=environment, timeout=remaining, check=False)
            report['stages'].append({'stage': step['stage'], 'returncode': result.returncode,
                                      'elapsed_seconds': round(time.monotonic() - started, 3)})
            if result.returncode != 0:
                raise RuntimeError('An isolated installation stage failed; no automatic retry')
        for name in ('sim', 'bridge'):
            frozen = subprocess.check_output([str(runtime / name / 'bin/python'), '-m', 'pip', 'freeze'],
                                             env=environment, cwd=ROOT, text=True, timeout=60)
            persist_export(runtime / f'{name}-packages.json', {'pip_freeze': frozen.splitlines()})
        report['installation_complete'] = True
        return report
    except Exception as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        persist_export(runtime / 'installation-result.json', report)
        print('EAI_RUNTIME_SETUP_V1 ' + json.dumps(report), flush=True)
        os.umask(previous_umask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-minutes', type=int, default=90)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'executed': False, 'stages': installation_plan(args.runtime_root)}, indent=2))
        return
    execute_installation(args.runtime_root, max_minutes=args.max_minutes)


if __name__ == '__main__':
    main()
