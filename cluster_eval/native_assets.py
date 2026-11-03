"""Fetch a fixed asset inventory and its payload only within a confirmed job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from cluster_eval.storage import persist_export, validate_local_destination
from scripts.download_robodojo_assets import validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def collect_manifest(items, lock: dict) -> dict:
    files = [{'path': item.path, 'size': item.size}
             for item in items if hasattr(item, 'size') and item.size is not None]
    manifest = {'revision': lock['revision'], 'complete': True,
                'file_count': len(files), 'total_bytes': sum(item['size'] for item in files),
                'files': sorted(files, key=lambda item: item['path'])}
    validate_manifest(manifest, lock)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    validate_local_destination(args.destination, ROOT)
    validate_local_destination(args.manifest, ROOT)
    if args.destination.exists() or args.manifest.exists():
        raise FileExistsError('Native asset inputs must use a new private destination')
    from huggingface_hub import HfApi

    lock = json.loads((ROOT / 'configs/robodojo_versions.json').read_text())['assets']
    items = HfApi(token=False).list_repo_tree(lock['repo_id'], repo_type='dataset',
        revision=lock['revision'], path_in_repo='Assets', recursive=True)
    persist_export(args.manifest, collect_manifest(items, lock))
    subprocess.run([sys.executable, '-B', '-m', 'scripts.download_robodojo_assets',
        '--destination', str(args.destination), '--manifest', str(args.manifest),
        '--storage-mode', 'job_local'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
