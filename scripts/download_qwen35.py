"""Download one pinned BF16 model into the shared-per-user HF cache."""

import argparse
import json
import os
from pathlib import Path
import shutil

MODEL = "Qwen/Qwen3.5-27B"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get(
        "HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub"))))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--storage-mode", choices=("legacy", "job_local"), default="legacy")
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    args = parser.parse_args()
    if args.source == 'modelscope':
        if args.storage_mode != 'job_local':
            raise ValueError('The verified mirror route is restricted to allocated cluster job storage')
        from cluster_eval.model_mirror import download_model
        from cluster_eval.storage import local_result_path, persist_export

        report = download_model(args.cache_dir, model=MODEL, revision=REVISION)
        persist_export(local_result_path(os.environ['EAI_RUN_ID']) / 'model-download.json', report)
        print(json.dumps({key: value for key, value in report.items() if key != 'files'}, indent=2))
        return
    if args.storage_mode == "job_local":
        from cluster_eval.storage import validate_local_destination
        validate_local_destination(args.cache_dir, Path(__file__).resolve().parents[1])
    from huggingface_hub import snapshot_download

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = snapshot_download(MODEL, revision=REVISION,
            cache_dir=args.cache_dir, local_files_only=True, token=False)
        index = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())
        complete = all((Path(snapshot) / name).is_file()
                       for name in set(index["weight_map"].values()))
    except (OSError, ValueError, KeyError):
        complete = False
    if not complete and shutil.disk_usage(args.cache_dir).free < 75 * 1024 ** 3:
        raise SystemExit("Need at least 75 GiB free for weights and a safety reserve.")
    snapshot = snapshot_download(MODEL, revision=REVISION, cache_dir=args.cache_dir,
        max_workers=args.max_workers, token=False,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE"])
    index = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())
    weights = [Path(snapshot) / name for name in set(index["weight_map"].values())]
    if not weights or any(not path.is_file() or not path.resolve().is_relative_to(args.cache_dir.resolve())
                          for path in weights):
        raise RuntimeError("Pinned model weights are missing or outside the private cache")
    report = {"model": MODEL, "revision": REVISION, "snapshot": snapshot,
              "weight_files": len(weights), "weight_bytes": sum(path.stat().st_size for path in weights)}
    if args.storage_mode == "job_local":
        from cluster_eval.storage import local_result_path, persist_export
        persist_export(local_result_path(os.environ["EAI_RUN_ID"]) / 'model-download.json', report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
