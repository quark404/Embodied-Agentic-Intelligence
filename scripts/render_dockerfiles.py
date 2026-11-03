"""Minimal build-manifest interface used by the prebuilt runtime checker.

The full web/Dockerfile generation flow was intentionally removed from this
snapshot. This module keeps the small constants and content-bound manifest that
``cluster_eval.prebuilt_runtime`` consumes when an image company-owned
environment is reused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "/opt/eai"


MANIFEST_FILES = (
    "configs/robodojo-baked-source-lock.json",
    "configs/robodojo-image-sources.json",
    "requirements/robodojo-bridge.txt",
    "requirements/robodojo-sim.txt",
    "requirements/robodojo-eval.txt",
    "requirements/qwen35-serving-cu128.txt",
)


def manifest(root: Path = ROOT) -> dict:
    """Return a deterministic digest over the current image dependency locks."""

    files = {}
    for relative in MANIFEST_FILES:
        path = root / relative
        if path.is_file():
            files[relative] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
    return {"version": 1, "prefix": PREFIX, "files": json.loads(json.dumps(files, sort_keys=True))}
