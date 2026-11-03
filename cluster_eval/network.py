"""Explicit credential-free outbound proxy settings for isolated workloads."""

from __future__ import annotations

from urllib.parse import urlsplit


PROXY_KEYS = frozenset(('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY'))
BYPASS = 'localhost,127.0.0.1,.hobot.cc,.horizon.cc,.oss-cn-beijing-internal.aliyuncs.com'


def proxy_environment(value: str) -> dict[str, str]:
    parsed = urlsplit(value)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.port is None
            or not 1 <= parsed.port <= 65535 or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ('', '/')
            or parsed.hostname in ('localhost', '::1') or parsed.hostname.startswith('127.')
            or any(character.isspace() for character in value)):
        raise ValueError('Use an explicit non-loopback HTTP(S) proxy host and port without credentials or query data')
    proxy = value.rstrip('/')
    return {'http_proxy': proxy, 'https_proxy': proxy, 'HTTP_PROXY': proxy, 'HTTPS_PROXY': proxy,
            'no_proxy': BYPASS, 'NO_PROXY': BYPASS}


def validate_proxy_environment(environment: dict) -> None:
    present = PROXY_KEYS.intersection(environment)
    if present and (present != PROXY_KEYS
                    or {name: environment[name] for name in PROXY_KEYS}
                    != proxy_environment(environment['https_proxy'])):
        raise ValueError('Workload proxy environment differs from the explicit reviewed proxy settings')
