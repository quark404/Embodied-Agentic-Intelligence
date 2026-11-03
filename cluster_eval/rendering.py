"""Validate small procedural camera artifacts without simulator dependencies."""

from __future__ import annotations

import base64
import hashlib
import math
import struct
import zlib


CAMERAS = ('front', 'left', 'right')
RESOLUTION = 96


def validate_png(content: bytes, *, width: int = RESOLUTION, height: int = RESOLUTION,
                 max_bytes: int = 32768) -> None:
    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= 2048 or not 1 <= height <= 2048):
        raise ValueError('PNG dimensions exceed the reviewed bound')
    if len(content) > max_bytes or content[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Invalid or oversized render PNG')
    offset = 8
    chunks = []
    image_data = bytearray()
    while offset < len(content):
        if offset + 12 > len(content):
            raise ValueError('Truncated PNG chunk')
        size = struct.unpack('>I', content[offset:offset + 4])[0]
        end = offset + 12 + size
        if end > len(content):
            raise ValueError('Truncated PNG data')
        kind = content[offset + 4:offset + 8]
        data = content[offset + 8:end - 4]
        checksum = struct.unpack('>I', content[end - 4:end])[0]
        if zlib.crc32(kind + data) & 0xffffffff != checksum:
            raise ValueError('PNG CRC mismatch')
        if not chunks:
            expected = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
            if kind != b'IHDR' or data != expected:
                raise ValueError('Render PNG must match the requested dimensions and RGB8 format')
        elif kind == b'IHDR':
            raise ValueError('Duplicate PNG header')
        if kind == b'IDAT':
            image_data.extend(data)
        if kind == b'IEND' and (size != 0 or end != len(content)):
            raise ValueError('Invalid PNG end marker')
        chunks.append(kind)
        offset = end
    if not chunks or chunks[-1] != b'IEND' or not image_data:
        raise ValueError('Incomplete PNG image')
    expected_length = height * (1 + width * 3)
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(image_data, expected_length + 1)
    if (len(decoded) != expected_length or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError('PNG pixel payload length mismatch')
    if any(decoded[row * (1 + width * 3)] > 4 for row in range(height)):
        raise ValueError('Invalid PNG row filter')


def validate_render_report(report: dict) -> dict[str, bytes]:
    if (report.get('native_evaluation_ready') is not False
            or report.get('scene') != 'procedural_rgb_cubes'
            or type(report.get('render_passed')) is not bool):
        raise ValueError('Render report must not claim a native RoboDojo result')
    if not report['render_passed']:
        if not isinstance(report.get('error_type'), str) or not report['error_type']:
            raise ValueError('Failed render needs an explicit error type')
        return {}
    frames = report.get('frames')
    if not isinstance(frames, dict) or set(frames) != set(CAMERAS):
        raise ValueError('All three procedural camera views are required')
    images = {}
    for name in CAMERAS:
        frame = frames[name]
        if (not isinstance(frame, dict) or frame.get('width') != RESOLUTION
                or frame.get('height') != RESOLUTION or frame.get('channels') != 3
                or type(frame.get('stddev')) not in (float, int)
                or not math.isfinite(frame['stddev']) or frame['stddev'] <= 1):
            raise ValueError('Empty, uniform or invalid rendered frame')
        content = base64.b64decode(frame['png_base64'], validate=True)
        validate_png(content)
        if hashlib.sha256(content).hexdigest() != frame.get('sha256'):
            raise ValueError('Rendered image checksum mismatch')
        images[name] = content
    if len(set(images.values())) < 2:
        raise ValueError('Camera views unexpectedly contain identical images')
    return images
