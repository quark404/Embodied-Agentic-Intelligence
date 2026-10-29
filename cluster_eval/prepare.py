"""Build an offline, single-GPU inventory job in new private directories only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets


ROOT = Path(__file__).resolve().parents[1]
QUEUE = "project-5090-robot-lab-acloud-langfang"


def validate_template(config: dict) -> None:
    required = {"execute": False, "clear_workspace": False,
                "project_id": "horizon-labs", "queue_name": QUEUE,
                "num_workers": 1, "gpu_per_worker": 1, "max_retries": 0,
                "priority": 1, "cpu_per_worker": 8, "cpu_mem_ratio": 8,
                "python_launcher": "python3", "python_executable": "cluster_eval/probe.py",
                "input_bucket": None, "output_bucket": None,
                "follow_symlinks": False, "to_upload": ["cluster_eval"]}
    for key, expected in required.items():
        if key not in config or type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"Unsafe inventory template: {key}")
    if type(config.get("wall_time")) is not int or not 1 <= config["wall_time"] <= 15:
        raise ValueError("Inventory wall_time must be 1..15 minutes")
    if config.get("env", {}).get("EAI_CLUSTER_CONFIRMED") != "0":
        raise ValueError("Offline template must retain EAI_CLUSTER_CONFIRMED=0")
    if "job_password" in config:
        raise ValueError("Never store a job password in a repository template")


def write_private(path: Path, content: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(content, output, ensure_ascii=False, indent=2)
        output.write("\n")


def prepare(*, run_id: str, image: str, result_root: Path) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must be a safe, unique name of at most 64 characters")
    if not image or image.startswith("REPLACE_") or any(character.isspace() for character in image):
        raise ValueError("Provide an approved OCI image reference")
    result_root = result_root.resolve()
    if not result_root.is_relative_to(Path("/jfs-public")) or not os.path.ismount("/jfs-public"):
        raise ValueError("A real /jfs-public mount is required; no home or bucket fallback")
    template = ROOT / "configs/aidi_5090_preflight.example.json"
    config = json.loads(template.read_text())
    validate_template(config)
    run = ROOT / "runs/cluster_eval" / run_id
    result_dir = result_root / run_id
    if run.exists() or run.is_symlink() or result_dir.exists() or result_dir.is_symlink():
        raise FileExistsError("Refusing to reuse or clear an existing staging/result path")
    from robo_orchard_jobs.job_submit.aidi.job_config import JobSubmitParamForAIDI

    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir(mode=0o700)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(mode=0o700)
    config.update(job_name=f"eai_{run_id}", docker_image=image,
                  workspace_folder=str(run / "workspace"), job_password=secrets.token_urlsafe(32))
    config["env"].update(EAI_RUN_ID=run_id, EAI_RESULT_DIR=str(result_dir))
    origin = {"run_id": run_id, "writer": "development_host",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "nonce": secrets.token_hex(16)}
    origin_bytes = (json.dumps(origin, sort_keys=True) + "\n").encode()
    descriptor = os.open(result_dir / "origin.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(origin_bytes)
    config["env"]["EAI_ORIGIN_SHA256"] = hashlib.sha256(origin_bytes).hexdigest()
    params = JobSubmitParamForAIDI(**config)
    previous_umask = os.umask(0o077)
    previous_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        prepared = params.prepare_workspace()
        mounts = params.as_aidi_job_params().OPTIONAL.DATA_SPACE
        if mounts.INPUT is not None or mounts.OUTPUT is not None:
            raise RuntimeError("Inventory job must not mount any dataset bucket")
        write_private(run / "submit.private.json", config)
        manifest = {"run_id": run_id, "queue": QUEUE, "project_id": "horizon-labs",
                    "submitted": False, "execute": False, "inventory_only": True,
                    "result_dir": str(result_dir), "workspace": prepared.workspace_folder,
                    "config_sha256": hashlib.sha256((run / "submit.private.json").read_bytes()).hexdigest(),
                    "artifact_sha256": {str(path.relative_to(run / "workspace")):
                                        hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in sorted((run / "workspace").rglob("*")) if path.is_file()},
                    "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted((ROOT / "cluster_eval").rglob("*")) if path.is_file()
                                      and "__pycache__" not in path.parts},
                    "pending": ["Explicit submission authorization", "Approved image and platform graphics runtime",
                                "Remote JFS verification", "Isaac RGB rendering", "Qwen connectivity", "Native episode"]}
        write_private(run / "manifest.json", manifest)
        return manifest
    finally:
        os.chdir(previous_directory)
        os.umask(previous_umask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(run_id=args.run_id, image=args.image,
                             result_root=args.result_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
