"""Validate and reuse image-owned environments without installing into them."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from scripts.render_dockerfiles import PREFIX, ROOT, manifest


ENVIRONMENTS = ('bootstrap', 'sim', 'bridge', 'qwen')


def expected_packages(root: Path = ROOT) -> dict:
    lock = json.loads((root / 'configs/robodojo-baked-source-lock.json').read_text())
    packages = {
        'bootstrap': {'uv': '0.9.3'},
        'sim': {'isaacsim': '5.1.0.0', 'torch': '2.7.0+cu128', 'websockets': '12.0',
                'isaaclab': '0.54.3', 'cuda-core': '1.1.0', 'starlette': '0.45.3',
                'omegaconf': '2.3.1', 'hydra-core': '1.3.4',
                'nvidia-curobo': lock['curobo_build_metadata']['local_packaging_version']},
    }
    for name, requirements in (('bridge', 'robodojo-bridge.txt'), ('qwen', 'qwen35-serving-cu128.txt')):
        packages[name] = dict(line.split('==', 1) for line in
                              (root / 'requirements' / requirements).read_text().splitlines())
    return packages


def verify(prefix: Path, environment: dict, *, root: Path = ROOT) -> dict:
    if str(prefix) != PREFIX or prefix.is_symlink() or prefix.resolve() != prefix:
        raise ValueError('Preinstalled runtime must use the fixed /opt/eai image prefix')
    actual = json.loads((prefix / 'runtime.json').read_text())
    if actual != manifest(root):
        raise ValueError('Preinstalled image manifest differs from current dependency/source locks')
    checks = {}
    for name, expected in expected_packages(root).items():
        virtualenv = prefix / name
        if virtualenv.is_symlink() or virtualenv.resolve() != virtualenv:
            raise ValueError('Image environments must not redirect to another directory')
        script = ('from importlib import metadata; from pathlib import Path; import json, sys; '
                  'assert sys.version_info[:2] == (3, 11); '
                  f'assert Path(sys.prefix).resolve() == Path({str(virtualenv)!r}); '
                  f'names={list(expected)!r}; '
                  'print(json.dumps({name:metadata.version(name) for name in names}))')
        output = subprocess.check_output([str(virtualenv / 'bin/python'), '-I', '-B', '-c', script],
                                         env=environment, text=True, timeout=60)
        versions = json.loads(output)
        if set(versions) != set(expected) or any(
                not isinstance(versions[name], str) or
                (versions[name] if '+' in pin else versions[name].split('+')[0]) != pin
                for name, pin in expected.items()):
            raise ValueError(f'Preinstalled {name} package versions differ from the image contract')
        checks[name] = versions
    return {'prefix': str(prefix), 'manifest': actual, 'packages': checks,
            'native_evaluation_ready': False}


def attach(runtime: Path, environment: dict) -> dict | None:
    configured = os.environ.get('EAI_PREBUILT_RUNTIME')
    if configured is None:
        return None
    if configured != PREFIX:
        raise ValueError('EAI_PREBUILT_RUNTIME must be exactly /opt/eai')
    prefix = Path(configured)
    for name in ENVIRONMENTS:
        destination = runtime / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Refusing to overwrite an existing job environment')
    report = verify(prefix, environment)
    for name in ENVIRONMENTS:
        (runtime / name).symlink_to(prefix / name, target_is_directory=True)
    return report
