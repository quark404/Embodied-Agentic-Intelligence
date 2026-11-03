"""Download pinned assets into an explicit private JFS or cluster-job directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil


ROOT = Path(__file__).resolve().parents[1]


def validate_manifest(manifest: dict, lock: dict) -> None:
    if (manifest.get("revision") != lock["revision"] or manifest.get("complete") is not True
            or manifest.get("file_count") != lock["file_count"]
            or manifest.get("total_bytes") != lock["total_bytes"]):
        raise ValueError("Asset inventory differs from the reviewed version lock")
    paths = set()
    total = 0
    for item in manifest["files"]:
        path = PurePosixPath(item["path"])
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                or path.parts[0] != "Assets" or item["path"] in paths
                or type(item["size"]) is not int or item["size"] < 0):
            raise ValueError("Invalid asset entry")
        paths.add(item["path"])
        total += item["size"]
    if len(paths) != lock["file_count"] or total != lock["total_bytes"]:
        raise ValueError("Asset inventory count or size mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--storage-mode", choices=("jfs", "job_local"), default="jfs")
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/robodojo_versions.json").read_text())["assets"]
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, lock)
    destination = args.destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a new private asset directory; existing data is never replaced")
    if args.storage_mode == "job_local":
        from cluster_eval.storage import validate_local_destination
        validate_local_destination(destination, ROOT)
    elif not destination.resolve().is_relative_to("/jfs-public") or not os.path.ismount("/jfs-public"):
        raise ValueError("Assets must stay on a real JFS mount, not home")
    if shutil.disk_usage(destination.parent).free < 2 * lock["total_bytes"]:
        raise ValueError("Insufficient headroom for the pinned asset payload")
    os.umask(0o077)
    destination.mkdir(mode=0o700)
    os.environ.update(HF_HOME=str(destination / ".cache/huggingface"),
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_XET="1", HF_HUB_DOWNLOAD_TIMEOUT="120")
    from huggingface_hub import snapshot_download

    (destination / "download-origin.json").write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(), "revision": lock["revision"],
        "repo_id": lock["repo_id"], "allow_patterns": lock["allow_patterns"],
        "expected_bytes": lock["total_bytes"], "native_evaluation_ready": False}, indent=2))
    snapshot_download(repo_id=lock["repo_id"], repo_type="dataset", revision=lock["revision"],
                      allow_patterns=lock["allow_patterns"], local_dir=destination,
                      cache_dir=destination / ".cache/hub", token=False, max_workers=4)
    for item in manifest["files"]:
        path = destination / item["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
            raise RuntimeError(f"Asset missing or size mismatch: {item['path']}")
    report = {"completed_at": datetime.now(timezone.utc).isoformat(),
              "revision": lock["revision"], "file_count": len(manifest["files"]),
              "total_bytes": lock["total_bytes"], "file_sizes_verified": True,
              "native_evaluation_ready": False}
    (destination / "download-complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
