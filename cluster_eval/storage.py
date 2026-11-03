"""Explicit job-local storage and bounded, verifiable inventory export."""

from __future__ import annotations

import hashlib
import base64
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import time
import urllib.request
import urllib.error
from urllib.parse import urlsplit


JOB_DATA = Path('/job_data')
PLATFORM_DATA_ROOT = Path('/running_root')
PACKAGE_ROOT = Path('/running_package')
EXPORT_PREFIX = 'EAI_INVENTORY_V1 '
RECORD_PREFIXES = {'inventory': EXPORT_PREFIX, 'render': 'EAI_RENDER_V1 ', 'native': 'EAI_NATIVE_V1 '}
MAX_EXPORT_BYTES = 131072
MAX_RENDER_BYTES = 262144
MAX_NATIVE_BYTES = 16 * 1024 ** 2
CHUNK_BYTES = 2000
BOOTSTRAP_WHEEL_URL = (
    'https://files.pythonhosted.org/packages/82/45/'
    '488417c6c0127c00bcdfac3556ae2ea0597df8245fe5f9bcfda35ebdbe85/'
    'uv-0.9.3-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
)
PUBLIC_DEPENDENCY_URLS = (
    'https://pypi.org/simple/', 'https://pypi.nvidia.com/',
    'https://huggingface.co/', 'https://github.com/', BOOTSTRAP_WHEEL_URL,
)


def chunk_render_record(encoded: bytes, run_id: str, origin_digest: str, *, record_type: str = 'render') -> str:
    count = (len(encoded) + CHUNK_BYTES - 1) // CHUNK_BYTES
    common = {'format': 'chunked-json-v1', 'run_id': run_id, 'origin_sha256': origin_digest,
              'count': count, 'bytes': len(encoded), 'record_sha256': hashlib.sha256(encoded).hexdigest()}
    lines = []
    for index in range(count):
        content = encoded[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES]
        entry = {**common, 'kind': 'chunk', 'index': index, 'data': base64.b64encode(content).decode()}
        lines.append(RECORD_PREFIXES[record_type] + json.dumps(entry, separators=(',', ':')))
    lines.append(RECORD_PREFIXES[record_type] + json.dumps({**common, 'kind': 'commit'}, separators=(',', ':')))
    return '\n'.join(lines)


def assemble_render_record(log: str, run_id: str, origin_digest: str, *, record_type: str = 'render') -> str:
    chunks = {}
    metadata = None
    committed = False
    direct = []
    prefix = RECORD_PREFIXES[record_type]
    limit = MAX_NATIVE_BYTES if record_type == 'native' else MAX_RENDER_BYTES
    for line in log.splitlines():
        if prefix not in line:
            continue
        encoded = line.split(prefix, 1)[1]
        if len(encoded.encode()) > limit:
            raise ValueError('Oversized render record')
        entry = json.loads(encoded)
        if entry.get('format') != 'chunked-json-v1':
            direct.append(prefix + encoded)
            continue
        current = {key: entry.get(key) for key in ('run_id', 'origin_sha256', 'count', 'bytes', 'record_sha256')}
        if (current['run_id'] != run_id or current['origin_sha256'] != origin_digest
                or type(current['bytes']) is not int or not 0 < current['bytes'] <= limit
                or type(current['count']) is not int
                or current['count'] != (current['bytes'] + CHUNK_BYTES - 1) // CHUNK_BYTES):
            raise ValueError('Invalid render chunk identity or bounds')
        if metadata is not None and metadata != current:
            raise ValueError('Conflicting render transfer metadata')
        metadata = current
        if entry.get('kind') == 'commit':
            committed = True
        elif entry.get('kind') == 'chunk':
            index = entry.get('index')
            if (type(index) is not int or not 0 <= index < current['count']
                    or not isinstance(entry.get('data'), str) or len(entry['data']) > 2700):
                raise ValueError('Invalid render chunk index or size')
            content = base64.b64decode(entry['data'], validate=True)
            expected_length = min(CHUNK_BYTES, current['bytes'] - index * CHUNK_BYTES)
            if len(content) != expected_length or (index in chunks and chunks[index] != content):
                raise ValueError('Conflicting or malformed render chunk')
            chunks[index] = content
        else:
            raise ValueError('Invalid render transfer record')
    if committed:
        if set(chunks) != set(range(metadata['count'])):
            raise ValueError('Committed render transfer has missing chunks')
        content = b''.join(chunks[index] for index in range(metadata['count']))
        if hashlib.sha256(content).hexdigest() != metadata['record_sha256']:
            raise ValueError('Render transfer checksum mismatch')
        direct.append(prefix + content.decode())
    return '\n'.join(direct)


def local_result_path(run_id: str) -> Path:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', run_id):
        raise ValueError('Invalid job-local run identity')
    return JOB_DATA / f'eai_{run_id}' / 'results'


def validate_origin(content: bytes, run_id: str, expected_digest: str) -> None:
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ValueError('Frozen origin digest mismatch')
    origin = json.loads(content)
    if origin.get('run_id') != run_id or origin.get('writer') != 'development_host':
        raise ValueError('Frozen origin identity mismatch')


def canonical_job_data_root() -> Path:
    canonical = JOB_DATA.resolve()
    if canonical == JOB_DATA:
        return canonical
    if (PLATFORM_DATA_ROOT.resolve() != PLATFORM_DATA_ROOT
            or not os.path.ismount(PLATFORM_DATA_ROOT)
            or not canonical.is_relative_to(PLATFORM_DATA_ROOT) or not canonical.is_dir()):
        raise ValueError('Job data alias must resolve inside the real platform /running_root mount')
    return canonical


def create_local_results(package_root: Path, result: Path, run_id: str, digest: str) -> Path:
    package_root = package_root.resolve()
    if not package_root.is_relative_to(PACKAGE_ROOT):
        raise ValueError('Job-local storage is only allowed in an uploaded cluster package')
    logical = local_result_path(run_id)
    canonical_root = canonical_job_data_root()
    expected = canonical_root / logical.relative_to(JOB_DATA)
    if result != logical or result.resolve() != expected:
        raise ValueError('Job-local output must be the exact isolated run directory')
    origin_path = package_root / 'origin.json'
    if origin_path.is_symlink() or not origin_path.is_file():
        raise ValueError('Frozen package origin must be a regular file')
    origin = origin_path.read_bytes()
    validate_origin(origin, run_id, digest)
    canonical_root.mkdir(mode=0o755, exist_ok=True)
    expected.parent.mkdir(mode=0o700)
    expected.mkdir(mode=0o700)
    descriptor = os.open(expected / 'origin.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as output:
        output.write(origin)
    return expected


def validate_local_destination(destination: Path, package_root: Path) -> Path:
    package_root = package_root.resolve()
    if (os.environ.get('EAI_CLUSTER_CONFIRMED') != '1'
            or os.environ.get('EAI_STORAGE_MODE') != 'job_local'
            or Path(os.environ.get('WORKING_PATH', '')).resolve() != package_root
            or not package_root.is_relative_to(PACKAGE_ROOT)):
        raise ValueError('Downloads and installs require an explicitly confirmed cluster job')
    run_id = os.environ.get('EAI_RUN_ID', '')
    logical_result = local_result_path(run_id)
    canonical_root = canonical_job_data_root()
    result = canonical_root / logical_result.relative_to(JOB_DATA)
    digest = os.environ.get('EAI_ORIGIN_SHA256', '')
    if logical_result.resolve() != result or result.parent.stat().st_uid != os.getuid():
        raise ValueError('Job-local workspace identity changed')
    validate_origin((result / 'origin.json').read_bytes(), run_id, digest)
    validate_origin((package_root / 'origin.json').read_bytes(), run_id, digest)
    target = destination.absolute()
    if target.is_relative_to(JOB_DATA):
        target = canonical_root / target.relative_to(JOB_DATA)
    if (target != target.resolve() or not target.is_relative_to(result.parent)
            or target == result.parent or target.is_relative_to(result)):
        raise ValueError('Workload writes must stay in this job workspace, outside exported results')
    return target


def export_record(payload: dict, origin_digest: str, *, record_type: str) -> str:
    if record_type not in RECORD_PREFIXES:
        raise ValueError('Unreviewed export record type')
    content = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    envelope = {'run_id': payload['run_id'], 'origin_sha256': origin_digest,
                f'{record_type}_sha256': hashlib.sha256(content.encode()).hexdigest(),
                record_type: payload}
    encoded = json.dumps(envelope, separators=(',', ':'), allow_nan=False)
    limit = MAX_NATIVE_BYTES if record_type == 'native' else MAX_RENDER_BYTES if record_type == 'render' else MAX_EXPORT_BYTES
    if len(encoded.encode()) > limit:
        raise ValueError('Inventory export exceeds the log transport limit')
    if record_type in ('render', 'native'):
        return chunk_render_record(encoded.encode(), payload['run_id'], origin_digest, record_type=record_type)
    return RECORD_PREFIXES[record_type] + encoded


def export_inventory(inventory: dict, origin_digest: str) -> str:
    return export_record(inventory, origin_digest, record_type='inventory')


def parse_record(log: str, run_id: str, origin_digest: str, *, record_type: str) -> dict | None:
    if record_type not in RECORD_PREFIXES:
        raise ValueError('Unreviewed export record type')
    if record_type in ('render', 'native'):
        log = assemble_render_record(log, run_id, origin_digest, record_type=record_type)
    prefix = RECORD_PREFIXES[record_type]
    found = None
    for line in log.splitlines():
        if prefix not in line:
            continue
        encoded = line.split(prefix, 1)[1]
        limit = MAX_NATIVE_BYTES if record_type == 'native' else MAX_RENDER_BYTES if record_type == 'render' else MAX_EXPORT_BYTES
        if len(encoded.encode()) > limit:
            raise ValueError('Oversized inventory export')
        envelope = json.loads(encoded)
        inventory = envelope.get(record_type)
        if (envelope.get('run_id') != run_id or envelope.get('origin_sha256') != origin_digest
                or not isinstance(inventory, dict) or inventory.get('run_id') != run_id
                or inventory.get('storage_mode') != 'job_local'
                or inventory.get('storage_origin_verified') is not True):
            raise ValueError('Inventory export identity or storage mode mismatch')
        canonical = json.dumps(inventory, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != envelope.get(f'{record_type}_sha256'):
            raise ValueError('Inventory export checksum mismatch')
        if found is not None and found != envelope:
            raise ValueError('Conflicting inventory exports for one job')
        found = envelope
    return found


def parse_inventory_export(log: str, run_id: str, origin_digest: str) -> dict | None:
    return parse_record(log, run_id, origin_digest, record_type='inventory')


def persist_bytes(path: Path, encoded: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix='.eai-export-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(encoded)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != encoded:
                raise ValueError('Refusing to replace a different recovered artifact')
    finally:
        Path(temporary).unlink()


def persist_export(path: Path, envelope: dict) -> None:
    encoded = (json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    persist_bytes(path, encoded)


def network_error(error: Exception) -> dict:
    reason = getattr(error, 'reason', error)
    result = {'error_type': type(error).__name__, 'reason_type': type(reason).__name__, 'reachable': False}
    error_number = getattr(reason, 'errno', None)
    if isinstance(error_number, int):
        result['errno'] = error_number
    tunnel_error = re.search(r'Tunnel connection failed: (\d{3})\b', str(reason))
    if tunnel_error:
        result['proxy_tunnel_status'] = int(tunnel_error.group(1))
    return result


def probe_network_url(url: str, *, direct: bool = False) -> dict:
    started = time.monotonic()
    try:
        request = urllib.request.Request(url, method='HEAD')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else None
        open_url = opener.open if opener is not None else urllib.request.urlopen
        with open_url(request, timeout=10) as response:
            result = {'status': response.status, 'reachable': True}
    except urllib.error.HTTPError as error:
        result = {'status': error.code, 'reachable': True, 'error_type': type(error).__name__}
    except Exception as error:
        result = network_error(error)
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result


def network_inventory() -> dict:
    results = {url: probe_network_url(url) for url in PUBLIC_DEPENDENCY_URLS}
    diagnostics = {'direct_internal_pypi': probe_network_url('https://pypi.hobot.cc/simple/', direct=True)}
    proxy = urllib.request.getproxies().get('https')
    diagnostics['effective_https_proxy_present'] = bool(proxy)
    if proxy:
        try:
            parsed = urlsplit(proxy)
            diagnostics['effective_proxy_credential_free'] = parsed.username is None and parsed.password is None
            if diagnostics['effective_proxy_credential_free'] and parsed.hostname and parsed.port:
                diagnostics['effective_proxy_host'] = parsed.hostname
                diagnostics['effective_proxy_port'] = parsed.port
                started = time.monotonic()
                try:
                    with socket.create_connection((parsed.hostname, parsed.port), timeout=5):
                        diagnostics['proxy_tcp'] = {'reachable': True}
                except Exception as error:
                    diagnostics['proxy_tcp'] = network_error(error)
                diagnostics['proxy_tcp']['elapsed_seconds'] = round(time.monotonic() - started, 3)
            diagnostics['direct_public_pypi'] = probe_network_url('https://pypi.org/simple/', direct=True)
        except ValueError:
            diagnostics['proxy_url_valid'] = False
    results['_diagnostics'] = diagnostics
    return results
