"""Download identical pinned Qwen files through ModelScope in job-local storage."""

from __future__ import annotations

from concurrent.futures import as_completed, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler

from cluster_eval.storage import persist_export, validate_local_destination


ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 ** 2


def validate_lock(lock: dict, model: str, revision: str) -> list[dict]:
    if (lock.get('model') != model or lock.get('hf_revision') != revision
            or lock.get('mirror') != 'https://modelscope.cn'):
        raise ValueError('Mirror must preserve the reviewed original model identity')
    files = lock.get('files', [])
    names = set()
    for item in files:
        name = item.get('path', '')
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
                or name in names or type(item.get('size')) is not int or item['size'] < 1
                or not re.fullmatch('[0-9a-f]{40}', item.get('mirror_revision', ''))
                or not re.fullmatch('[0-9a-f]{64}', item.get('sha256', ''))):
            raise ValueError('Invalid pinned mirror entry')
        names.add(name)
        blob = item.get('hf_git_blob_sha1')
        lfs = item.get('hf_lfs_sha256')
        if not ((blob is None and lfs == item['sha256'])
                or (lfs is None and isinstance(blob, str) and re.fullmatch('[0-9a-f]{40}', blob))):
            raise ValueError('Every mirror file needs an independent original Hugging Face digest')
    if (not {'config.json', 'model.safetensors.index.json', 'tokenizer_config.json'}.issubset(names)
            or not any(name.endswith('.safetensors') for name in names)
            or sum(item['size'] for item in files) != lock.get('total_bytes')):
        raise ValueError('Incomplete mirror inventory')
    return files


def download_file(item: dict, snapshot: Path, model: str) -> dict:
    destination = snapshot / item['path']
    partial = snapshot / (item['path'] + '.partial')
    url = ('https://modelscope.cn/models/' + model + '/resolve/'
           + item['mirror_revision'] + '/' + quote(item['path'], safe=''))
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f'blob {item["size"]}\0'.encode())
    count = 0
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=120) as response:
        if response.status != 200:
            raise ValueError('Mirror did not return a complete file')
        declared = response.headers.get('Content-Length')
        if declared is not None and int(declared) != item['size']:
            raise ValueError('Mirror response length differs from the original file')
        descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, 'wb') as output:
            while True:
                content = response.read(min(CHUNK_BYTES, item['size'] - count + 1))
                if not content:
                    break
                count += len(content)
                if count > item['size']:
                    raise ValueError('Mirror returned more bytes than the pinned file')
                sha256.update(content)
                git_blob.update(content)
                output.write(content)
            output.flush()
            os.fsync(output.fileno())
    if count != item['size'] or sha256.hexdigest() != item['sha256']:
        raise ValueError('Mirror payload SHA-256 or size mismatch')
    if item.get('hf_git_blob_sha1') and git_blob.hexdigest() != item['hf_git_blob_sha1']:
        raise ValueError('Mirror payload differs from the pinned original Git blob')
    os.link(partial, destination)
    partial.unlink()
    print(json.dumps({'event': 'model_mirror_file_verified', 'file': item['path'], 'bytes': count}), flush=True)
    return {'path': item['path'], 'size': count, 'sha256': sha256.hexdigest(), 'original_digest_verified': True}


def download_model(cache: Path, *, model: str, revision: str) -> dict:
    validate_local_destination(cache, ROOT)
    lock = json.loads((ROOT / 'configs/qwen35-modelscope-lock.json').read_text())
    files = validate_lock(lock, model, revision)
    if cache.exists() or cache.is_symlink():
        raise FileExistsError('Use a new private model cache; never overwrite another download')
    if shutil.disk_usage(cache.parent).free < lock['total_bytes'] + 16 * 1024 ** 3:
        raise ValueError('Insufficient space for verified weights and the private storage reserve')
    cache.mkdir(mode=0o700)
    snapshot = cache / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
    snapshot.mkdir(mode=0o700, parents=True)
    persist_export(cache / 'mirror-origin.json', lock)
    checks = [download_file(item, snapshot, model) for item in files if not item['path'].endswith('.safetensors')]
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    expected = {item['path'] for item in files if item['path'].endswith('.safetensors')}
    if set(index['weight_map'].values()) != expected:
        raise ValueError('Verified model index does not match the reviewed weight shards')
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(download_file, item, snapshot, model) for item in files if item['path'] in expected]
        try:
            for future in as_completed(futures):
                checks.append(future.result())
        except Exception:
            for future in futures:
                future.cancel()
            raise
    report = {'model': model, 'revision': revision, 'snapshot': str(snapshot),
              'weight_files': len(expected),
              'weight_bytes': sum(item['size'] for item in files if item['path'] in expected),
              'mirror': 'modelscope', 'original_digests_verified': True,
              'files': sorted(checks, key=lambda item: item['path'])}
    persist_export(cache / 'mirror-verified.json', report)
    return report
