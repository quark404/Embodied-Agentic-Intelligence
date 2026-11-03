"""Bounded, direct resource checks inside an allocated job; never install anything."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatchcase
import hashlib
from html.parser import HTMLParser
import time
import urllib.error
import urllib.request
from urllib.parse import unquote, urljoin, urlsplit

from cluster_eval.storage import network_error


INDEX_LIMIT = 4 * 1024 ** 2
SAMPLE_LIMIT = 4096
TIMEOUT_SECONDS = 8
INTERNAL_INDEX = 'https://pypi.hobot.cc/simple/'
INDEX_TARGETS = (
    ('internal_uv', INTERNAL_INDEX + 'uv/', 'uv-0.9.3-*-manylinux*x86_64.whl'),
    ('internal_isaacsim', INTERNAL_INDEX + 'isaacsim/', 'isaacsim-5.1.0*-cp311-*-*x86_64.whl'),
    ('internal_isaacsim_kernel', INTERNAL_INDEX + 'isaacsim-kernel/', 'isaacsim_kernel-5.1.0*-cp311-*-*x86_64.whl'),
    ('nvidia_isaacsim_kernel', 'https://pypi.nvidia.com/isaacsim-kernel/', 'isaacsim_kernel-5.1.0*-cp311-*-*x86_64.whl'),
    ('internal_torch_cu128', INTERNAL_INDEX + 'torch/', 'torch-2.7.0+cu128-cp311-cp311-*x86_64.whl'),
)
FILE_TARGETS = (
    ('robodojo_source', 'https://codeload.github.com/Mosi-AI/RoboDojo/tar.gz/00ef04cd245bfa76e6f1e0bd8383c82f7964f942'),
    ('python_release_metadata', 'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest'),
    ('qwen_pinned_index', 'https://huggingface.co/Qwen/Qwen3.5-27B/resolve/fc05daec18b0a78c049392ed2e771dde82bdf654/model.safetensors.index.json'),
    ('robodojo_pinned_assets_metadata', 'https://huggingface.co/api/datasets/RoboDojo-Benchmark/RoboDojo/revision/0e0739f55524af015344a2b31e791cb3486d589d'),
    ('modelscope_qwen_config_candidate', 'https://modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/config.json'),
    ('pytorch_cu128_index', 'https://download.pytorch.org/whl/cu128/torch/'),
)


class WheelLinks(HTMLParser):
    def __init__(self, base: str, pattern: str):
        super().__init__()
        self.base = base
        self.pattern = pattern
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        href = attributes.get('href')
        if tag != 'a' or not href or 'data-yanked' in attributes:
            return
        parsed = urlsplit(urljoin(self.base, href))
        filename = unquote(parsed.path.rsplit('/', 1)[-1])
        if (parsed.scheme == 'https' and parsed.hostname
                and parsed.username is None and parsed.password is None
                and fnmatchcase(filename, self.pattern)):
            self.links.append(parsed._replace(fragment='').geturl())


def direct_open(url: str, *, sample: bool):
    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'EAI-resource-preflight/1'}
    if sample:
        headers['Range'] = f'bytes=0-{SAMPLE_LIMIT - 1}'
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=TIMEOUT_SECONDS)


def file_sample(url: str, *, wheel: bool = False) -> dict:
    started = time.monotonic()
    try:
        with direct_open(url, sample=True) as response:
            content = response.read(SAMPLE_LIMIT)
            result = {'status': response.status, 'final_host': urlsplit(response.geturl()).hostname,
                      'sample_bytes': len(content), 'sample_sha256': hashlib.sha256(content).hexdigest(),
                      'sample_readable': response.status in (200, 206) and bool(content),
                      'full_download_verified': False}
            if wheel:
                result['wheel_signature_valid'] = content.startswith(b'PK\x03\x04')
                result['sample_readable'] &= result['wheel_signature_valid']
    except urllib.error.HTTPError as error:
        result = {'status': error.code, 'sample_readable': False}
        error.close()
    except Exception as error:
        result = {**network_error(error), 'sample_readable': False}
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result


def index_sample(url: str, pattern: str) -> dict:
    started = time.monotonic()
    try:
        with direct_open(url, sample=False) as response:
            content = response.read(INDEX_LIMIT + 1)
            result = {'index_status': response.status, 'index_host': urlsplit(response.geturl()).hostname,
                      'index_bytes': min(len(content), INDEX_LIMIT), 'index_truncated': len(content) > INDEX_LIMIT}
            parser = WheelLinks(response.geturl(), pattern)
            parser.feed(content[:INDEX_LIMIT].decode('utf-8'))
        result['matching_wheels'] = len(parser.links)
        if parser.links:
            target = sorted(parser.links)[0]
            result['wheel_filename'] = unquote(urlsplit(target).path.rsplit('/', 1)[-1])
            result['wheel_sample'] = file_sample(target, wheel=True)
    except urllib.error.HTTPError as error:
        result = {'index_status': error.code, 'error_type': type(error).__name__}
        error.close()
    except Exception as error:
        result = network_error(error)
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result


def resource_inventory() -> dict:
    from cluster_eval.probe import check_execution_context

    check_execution_context()
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {name: executor.submit(index_sample, url, pattern) for name, url, pattern in INDEX_TARGETS}
        pending.update({name: executor.submit(file_sample, url) for name, url in FILE_TARGETS})
        checks = {name: future.result() for name, future in pending.items()}
    return {'direct_only': True, 'installs_performed': False, 'full_resource_set_verified': False,
            'sample_limit_bytes': SAMPLE_LIMIT, 'checks': checks}
