"""Download one pinned BF16 model into the shared-per-user HF cache."""

import argparse
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download


MODEL = "Qwen/Qwen3.5-27B"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get(
        "HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub"))))
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = snapshot_download(MODEL, revision=REVISION,
            cache_dir=args.cache_dir, local_files_only=True)
        index = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())
        complete = all((Path(snapshot) / name).is_file()
                       for name in set(index["weight_map"].values()))
    except (OSError, ValueError, KeyError):
        complete = False
    if not complete and shutil.disk_usage(args.cache_dir).free < 75 * 1024 ** 3:
        raise SystemExit("Need at least 75 GiB free for weights and a safety reserve.")
    snapshot = snapshot_download(MODEL, revision=REVISION, cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE"])
    print(json.dumps({"model": MODEL, "revision": REVISION, "snapshot": snapshot}, indent=2))


if __name__ == "__main__":
    main()
