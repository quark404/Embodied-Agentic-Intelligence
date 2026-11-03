"""Read metadata from a baked image and approved read-only data mounts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


SIM_PYTHON = Path('/opt/conda/envs/RoboDojo/bin/python')
SOURCE_ROOT = Path('/opt/robodojo')
STARVLA_SIM_PYTHON = Path('/opt/robodojo-env/bin/python')
STARVLA_PROXY_PYTHON = Path('/opt/robodojo-proxy-env/bin/python')
STARVLA_SOURCE_ROOT = Path('/workspace/RoboDojo')
EAI_SIM_PYTHON = Path('/opt/eai/sim/bin/python')
EAI_QWEN_PYTHON = Path('/opt/eai/qwen/bin/python')
EAI_SOURCE_ROOT = Path('/opt/robodojo')
PACKAGE_NAMES = ('isaacsim', 'isaacsim-kernel', 'isaaclab', 'torch', 'torchvision', 'curobo',
                 'numpy', 'websockets', 'cuda-core', 'huggingface-hub', 'vllm', 'transformers')
SCRIPT = (
    'from importlib import metadata; import json,sys; '
    'installed={entry.metadata["Name"].lower().replace("_","-"):entry.version '
    'for entry in metadata.distributions()}; '
    'print(json.dumps({"python":sys.version.split()[0],"prefix":sys.prefix,'
    '"packages":{name:installed.get(name) for name in ' + repr(PACKAGE_NAMES) + '}}))'
)


def layout_candidates() -> tuple[tuple[str, Path, Path], ...]:
    return (
        ('legacy', SIM_PYTHON, SOURCE_ROOT),
        ('starvla-v4.6-sim', STARVLA_SIM_PYTHON, STARVLA_SOURCE_ROOT),
        ('starvla-v4.6-proxy', STARVLA_PROXY_PYTHON, STARVLA_SOURCE_ROOT),
        ('eai-sim', EAI_SIM_PYTHON, EAI_SOURCE_ROOT),
        ('eai-qwen', EAI_QWEN_PYTHON, EAI_SOURCE_ROOT),
    )


def inspect_layout(name: str, python: Path, source: Path) -> dict:
    layout = {'name': name, 'python': str(python), 'source': str(source),
              'python_exists': python.is_file(), 'source_exists': source.is_dir()}
    if python.is_file():
        try:
            completed = subprocess.run([str(python), '-I', '-B', '-c', SCRIPT],
                                       capture_output=True, text=True, timeout=30)
            layout['interpreter_exit_code'] = completed.returncode
            if completed.returncode == 0:
                layout['runtime'] = json.loads(completed.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            layout['runtime_error_type'] = type(error).__name__
    if source.is_dir():
        try:
            completed = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True, timeout=10)
            revision = completed.stdout.strip()
            layout['source_git_revision'] = revision if re.fullmatch('[0-9a-f]{40}', revision) else None
        except (OSError, subprocess.TimeoutExpired) as error:
            layout['source_error_type'] = type(error).__name__
        layout['source_files'] = {file_name: hashlib.sha256((source / file_name).read_bytes()).hexdigest()
                                  for file_name in ('README.md', 'pyproject.toml', 'task/general_pickup.py')
                                  if (source / file_name).is_file()
                                  and (source / file_name).stat().st_size < 1024 ** 2}
    return layout


def inspect_image_runtime() -> dict:
    from cluster_eval.probe import check_execution_context

    check_execution_context()
    layouts = [inspect_layout(name, python, source) for name, python, source in layout_candidates()]
    result = {'sim_python_exists': any(item['python_exists'] for item in layouts),
              'source_exists': any(item['source_exists'] for item in layouts),
              'layouts': layouts, 'modified_image': False, 'native_evaluation_ready': False}
    matched = next((item for item in layouts if item.get('runtime') is not None), None)
    if matched:
        result.update(runtime=matched['runtime'], interpreter_exit_code=matched['interpreter_exit_code'])
    sourced = next((item for item in layouts if item['source_exists']), None)
    if sourced:
        result.update(source_git_revision=sourced.get('source_git_revision'),
                      source_files=sourced.get('source_files', {}))
    result['input_mounts'] = {}
    for bucket in ('robot_lab', 'robot_lab2'):
        root = Path('/bucket/input') / bucket
        found = []
        failures = []
        for suffix in ('', 'datasets', 'models', 'pretrained_models', 'benchmarks', 'benchmark_data',
                       'assets', 'eval', 'simulation', 'holobrain', 'robodojo_compare_e2e'):
            directory = root / suffix
            if not directory.is_dir():
                continue
            try:
                with os.scandir(directory) as entries:
                    for count, entry in enumerate(entries):
                        if count >= 512:
                            break
                        if re.search(r'robodojo|roboicl|qwen.?3.?5|assets|benchmark|eval.?data|pretrained', entry.name, re.IGNORECASE):
                            found.append(str(Path(suffix) / entry.name))
            except OSError as error:
                failures.append({'relative_path': suffix, 'error_type': type(error).__name__})
        result['input_mounts'][bucket] = {'exists': root.is_dir(), 'resource_candidates': sorted(found),
                                          'scan_errors': failures}
    return result
