"""Private overlays over a verified image; never install into its base environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from cluster_eval.baked_job import ASSET_DATASET, inspect_assets
from cluster_eval.image_inventory import SIM_PYTHON, SOURCE_ROOT
from cluster_eval.prebuilt_runtime import attach
from cluster_eval.resource_probe import INTERNAL_INDEX
from cluster_eval.runtime_install import RUNTIME_PATHS, runtime_environment
from cluster_eval.storage import persist_export, validate_local_destination


ROOT = Path(__file__).resolve().parents[1]


def baked_environment(runtime: Path) -> dict:
    environment = runtime_environment(runtime)
    environment.pop('UV_PYTHON_PREFERENCE', None)
    for variable in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_ROOT', 'PIP_USER',
                     'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS'):
        environment.pop(variable, None)
    environment.update(PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL=INTERNAL_INDEX,
                       PIP_REQUIRE_VIRTUALENV='true', UV_DEFAULT_INDEX=INTERNAL_INDEX, UV_NO_CONFIG='1')
    metadata = json.loads((ROOT / 'configs/robodojo-baked-source-lock.json').read_text())['curobo_build_metadata']
    environment.update(CUROBO_USE_PYBIND='0',
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=metadata['local_packaging_version'])
    return environment


def installation_plan(runtime: Path, *, preinstalled: bool = False) -> list[tuple[str, list[str]]]:
    uv = str(runtime / 'bootstrap/bin/uv')
    bridge_python = str(runtime / 'bridge/bin/python')
    sim_python = str(runtime / 'sim/bin/python')
    plan = [
        ('bootstrap_venv', [str(SIM_PYTHON), '-m', 'venv', str(runtime / 'bootstrap')]),
        ('isolated_uv_internal', [str(runtime / 'bootstrap/bin/python'), '-m', 'pip', 'install',
             '--index-url', INTERNAL_INDEX, '--retries', '0', '--timeout', '30', 'uv==0.9.3']),
        ('simulation_overlay', [str(SIM_PYTHON), '-m', 'venv', '--system-site-packages', str(runtime / 'sim')]),
        ('private_curobo_dependencies', [sim_python, '-m', 'pip', 'install', '--no-input',
             '--index-url', INTERNAL_INDEX, '--constraint', str(runtime / 'inherited-constraints.txt'),
             '-e', str(runtime / 'RoboDojo/third_party/curobo')]),
        ('bridge_venv', [str(SIM_PYTHON), '-m', 'venv', str(runtime / 'bridge')]),
        ('bridge_dependencies', [uv, 'pip', 'install', '--python', bridge_python, '--index-url', INTERNAL_INDEX,
                                 '-r', str(ROOT / 'requirements/robodojo-bridge.txt')]),
        ('bridge_dependency_check', [bridge_python, '-m', 'pip', 'check']),
        ('simulation_cuda_check', [sim_python, '-B', '-c',
             'import torch, curobo; '
             'from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg; '
             'from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg; '
             'from curobo.motion_planner import MotionPlanner; '
             'from cuda.core import Device, Program, ProgramOptions; '
             'assert torch.__version__ == "2.7.0+cu128"; '
             'assert torch.cuda.device_count() == 1; '
             'assert torch.cuda.get_device_capability(0) == (12,0); '
             'Device(0).set_current(); '
             'program=Program(\'extern "C" __global__ void eai_probe() {}\', '
             'code_type="c++", options=ProgramOptions(arch="sm_120")); '
             'module=program.compile("cubin", name_expressions=("eai_probe",)); '
             'assert module.get_kernel("eai_probe") is not None; '
             'print(torch.ones(8,device="cuda").sum().item())']),
    ]
    if preinstalled:
        return [(stage, command) for stage, command in plan
                if stage in ('bridge_dependency_check', 'simulation_cuda_check')]
    return plan


def constraint_lines(packages: dict) -> str:
    lines = []
    for name, version in sorted(packages.items()):
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
                or not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.!+_-]*', version)):
            raise ValueError('Inherited package metadata is not a safe exact constraint')
        normalized = re.sub(r'[-_.]+', '-', name).lower()
        if normalized != 'nvidia-curobo':
            lines.append(f'{normalized}=={version}')
    return '\n'.join(lines) + '\n'


def snapshot_dependencies(runtime: Path, environment: dict) -> None:
    script = ('from importlib import metadata; import json; '
              'names=sorted({entry.metadata["Name"] for entry in metadata.distributions()}); '
              'print(json.dumps({name:metadata.version(name) for name in names}))')
    output = subprocess.check_output([str(runtime / 'sim/bin/python'), '-I', '-B', '-c', script],
                                     env=environment, text=True, timeout=120)
    packages = json.loads(output)
    (runtime / 'inherited-constraints.txt').write_text(constraint_lines(packages))
    persist_export(runtime / 'inherited-packages.json', packages)


def validate_curobo_metadata(source: Path) -> dict:
    metadata = json.loads((ROOT / 'configs/robodojo-baked-source-lock.json').read_text())['curobo_build_metadata']
    actual = {}
    for relative, expected in metadata['files'].items():
        path = source / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('CuRobo build metadata must remain inside the private source')
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[relative] != expected:
            raise ValueError(f'Unreviewed CuRobo build metadata: {relative}')
    return {**metadata, 'files': actual, 'build_metadata_verified': True}


def validate_core_sources(source: Path) -> dict:
    lock = json.loads((ROOT / 'configs/robodojo-baked-source-lock.json').read_text())
    actual = {}
    for relative, expected in lock['files'].items():
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing regular benchmark grading file: {relative}')
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[relative] != expected:
            raise ValueError(f'Benchmark task/reward definition differs from the reference: {relative}')
    return {'reference_commit': lock['reference_commit'], 'core_file_sha256': actual,
            'core_files_verified': True, 'whole_source_revision_verified': False}


def review_launcher(launcher: Path) -> dict:
    if not launcher.is_file() or launcher.is_symlink():
        raise ValueError('Native launcher must be a regular file')
    content = launcher.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    reviewed = json.loads((ROOT / 'configs/robodojo-baked-source-lock.json').read_text())['reviewed_launchers']
    if digest not in reviewed:
        raise ValueError('Native launcher differs from both reviewed source versions')
    forwarding = reviewed[digest]['forwards_kit_args']
    expected = 'KIT_ARGS="${ROBODOJO_KIT_ARGS:-}"' if forwarding else 'KIT_ARGS=""'
    assignments = [line for line in content.decode().splitlines() if line.startswith('KIT_ARGS=')]
    if assignments != [expected]:
        raise ValueError('Reviewed native Kit assignment is inconsistent with its source lock')
    return {'sha256': digest, 'forwards_kit_args': forwarding}


def configure_private_launcher(source: Path, expected: dict) -> dict:
    launcher = source / 'scripts/eval_policy.sh'
    if launcher.is_symlink() or not launcher.resolve().is_relative_to(source.resolve()):
        raise ValueError('Native launcher must stay inside the private source copy')
    actual = review_launcher(launcher)
    if actual != expected:
        raise ValueError('Native launcher changed during source copy')
    if not actual['forwards_kit_args']:
        launcher.write_text(launcher.read_text().replace('KIT_ARGS=""', 'KIT_ARGS="${ROBODOJO_KIT_ARGS:-}"'))
    return {**actual, 'modified': not actual['forwards_kit_args'],
            'configured_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest()}


def copy_sources(runtime: Path) -> dict:
    evidence = validate_core_sources(SOURCE_ROOT)
    launcher_review = review_launcher(SOURCE_ROOT / 'scripts/eval_policy.sh')
    print(json.dumps({'event': 'baked_runtime_stage', 'stage': 'copy_verified_sources',
                      'core_files_verified': True, 'launcher': launcher_review}), flush=True)
    source = runtime / 'RoboDojo'
    shutil.copytree(SOURCE_ROOT, source, symlinks=True,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    policy = source / 'XPolicyLab/policy/Embodied_Agentic'
    if not policy.parent.resolve().is_relative_to(source.resolve()):
        raise ValueError('Copied policy directory must not redirect writes into the base image')
    policy.parent.mkdir(parents=True, exist_ok=True)
    if policy.exists() or policy.is_symlink():
        raise FileExistsError('Refusing to overwrite an existing policy in the copied source')
    policy.symlink_to(ROOT / 'integrations/xpolicylab/Embodied_Agentic', target_is_directory=True)
    launcher_evidence = configure_private_launcher(source, launcher_review)
    evidence.update(source=str(source), source_origin='prebuilt_image',
                    launcher=launcher_evidence,
                    private_launcher_change=('Forward ROBODOJO_KIT_ARGS for private Kit storage only'
                        if launcher_evidence['modified'] else 'Already forwards ROBODOJO_KIT_ARGS; unchanged'))
    persist_export(runtime / 'baked-source.json', evidence)
    return evidence


def setup(runtime: Path, inventory: dict) -> dict:
    validate_local_destination(runtime, ROOT)
    packages = inventory.get('baked_image', {}).get('runtime', {}).get('packages', {})
    if (inventory.get('run_id') != os.environ.get('EAI_RUN_ID')
            or inventory.get('inventory_passed') is not True or inventory.get('expected_gpus') != 3
            or inventory.get('gpu_topology_verified') is not True
            or packages.get('isaacsim') != '5.1.0.0' or packages.get('torch') != '2.7.0+cu128'):
        raise ValueError('Require this job\'s three-GPU and prebuilt-runtime evidence')
    runtime.mkdir(mode=0o700)
    for suffix in RUNTIME_PATHS.values():
        (runtime / suffix).mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = baked_environment(runtime)
    report = {'runtime_source': 'prebuilt_image', 'base_environment_modified': False, 'stages': []}
    report['preinstalled_runtime'] = attach(runtime, environment)
    report['source'] = copy_sources(runtime)
    report['curobo_build_metadata'] = validate_curobo_metadata(runtime / 'RoboDojo/third_party/curobo')
    assets = inspect_assets(ASSET_DATASET, full_scan=False)
    if not assets.get('object_directory') or not assets.get('robot_directory'):
        raise ValueError('Verified read-only asset root must contain Object and Robots')
    private_assets = runtime / 'assets'
    private_assets.mkdir(mode=0o700)
    (private_assets / 'Assets').symlink_to(Path(assets['assets_path']), target_is_directory=True)
    report['assets'] = assets
    persist_export(runtime / 'shared-assets.json', assets)
    deadline = time.monotonic() + 1800
    for stage, arguments in installation_plan(runtime, preinstalled=report['preinstalled_runtime'] is not None):
        print(json.dumps({'event': 'baked_runtime_stage', 'stage': stage}), flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Private image-overlay preparation budget exhausted')
        completed = subprocess.run(arguments, cwd=ROOT, env=environment, timeout=remaining, check=False)
        report['stages'].append({'stage': stage, 'returncode': completed.returncode})
        if completed.returncode:
            persist_export(runtime / 'baked-setup-failure.json', report)
            raise RuntimeError(f'Private image-overlay setup failed at {stage}')
        if stage == 'simulation_overlay':
            snapshot_dependencies(runtime, environment)
    for name in ('sim', 'bridge'):
        frozen = subprocess.check_output([str(runtime / name / 'bin/python'), '-m', 'pip', 'freeze'],
                                         cwd=ROOT, env=environment, text=True, timeout=60)
        persist_export(runtime / f'{name}-packages.json', {'pip_freeze': frozen.splitlines()})
    persist_export(runtime / 'baked-setup.json', report)
    return report
