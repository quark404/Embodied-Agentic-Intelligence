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

from cluster_eval.storage import local_result_path
from cluster_eval.network import proxy_environment


ROOT = Path(__file__).resolve().parents[1]
QUEUE = "project-5090-robot-lab-acloud-langfang"
QUEUE_CLUSTERS = {
    QUEUE: "acloud-langfang",
    "project-5090-robot-lab-debug-bcloud-bj": "bcloud-bj-zone1",
    "project-5090-horizon-labs-aidc": "hcloud",
}
PROFILES = {
    "inventory": {"template": "aidi_5090_preflight.example.json", "max_minutes": 15,
                  "entry": "cluster_eval/probe.py", "upload": ["cluster_eval"]},
    "image_inventory": {"template": "aidi_5090_image_inventory.example.json", "max_minutes": 15,
                        "entry": "cluster_eval/image_probe.py", "upload": ["cluster_eval"],
                        "input_buckets": "robot_lab,robot_lab2"},
    "baked_render": {"template": "aidi_5090_baked_render.example.json", "max_minutes": 30,
                     "entry": "cluster_eval/baked_job.py", "upload": ["cluster_eval", "configs"],
                     "input_buckets": "robot_lab,robot_lab2"},
    "baked_native": {"template": "aidi_5090_baked_native.example.json", "max_minutes": 240,
                     "gpus": 3, "entry": "cluster_eval/native_job.py", "runtime_source": "prebuilt_image",
                     "input_buckets": "robot_lab,robot_lab2",
                     "upload": ["cluster_eval", "requirements", "scripts", "configs", "embodied_agentic", "integrations"]},
    "render": {"template": "aidi_5090_render.example.json", "max_minutes": 120,
               "entry": "cluster_eval/runtime_job.py",
               "upload": ["cluster_eval", "requirements", "scripts", "configs",
                          "embodied_agentic", "integrations"]},
    "native": {"template": "aidi_5090_native.example.json", "max_minutes": 240,
               "gpus": 3, "entry": "cluster_eval/native_job.py",
               "upload": ["cluster_eval", "requirements", "scripts", "configs",
                          "embodied_agentic", "integrations"]},
}


def profile_settings(profile: str) -> dict:
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("Only explicitly reviewed job profiles are allowed")
    return PROFILES[profile]


def queue_cluster(queue: str) -> str:
    if not isinstance(queue, str) or queue not in QUEUE_CLUSTERS:
        raise ValueError("Only explicitly reviewed inventory queues are allowed")
    return QUEUE_CLUSTERS[queue]


def validate_template(config: dict, *, profile: str = "inventory") -> None:
    settings = profile_settings(profile)
    queue_cluster(config.get("queue_name"))
    required = {"execute": False, "clear_workspace": False,
                "project_id": "horizon-labs",
                "num_workers": 1, "gpu_per_worker": settings.get("gpus", 1), "max_retries": 0,
                "priority": 1, "cpu_per_worker": 8, "cpu_mem_ratio": 8,
                "python_launcher": "python3", "python_executable": settings["entry"],
                "input_bucket": settings.get("input_buckets"), "output_bucket": None,
                "follow_symlinks": False, "to_upload": settings["upload"]}
    for key, expected in required.items():
        if key not in config or type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"Unsafe inventory template: {key}")
    if type(config.get("wall_time")) is not int or not 1 <= config["wall_time"] <= settings["max_minutes"]:
        raise ValueError(f"{profile} wall_time must be 1..{settings['max_minutes']} minutes")
    if config.get("env", {}).get("EAI_CLUSTER_CONFIRMED") != "0":
        raise ValueError("Offline template must retain EAI_CLUSTER_CONFIRMED=0")
    if config.get('env', {}).get('EAI_RUNTIME_SOURCE') != settings.get('runtime_source'):
        raise ValueError('Runtime source must match the explicitly selected profile')
    if "job_password" in config:
        raise ValueError("Never store a job password in a repository template")


def write_private(path: Path, content: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(content, output, ensure_ascii=False, indent=2)
        output.write("\n")


def prepare(*, run_id: str, image: str, result_root: Path, queue: str = QUEUE,
            storage_mode: str = "jfs", profile: str = "inventory", egress_proxy: str | None = None) -> dict:
    cluster = queue_cluster(queue)
    settings = profile_settings(profile)
    network_environment = proxy_environment(egress_proxy) if egress_proxy is not None else {}
    if storage_mode not in ("jfs", "job_local"):
        raise ValueError("Storage mode must be explicit: jfs or job_local")
    if profile in ("render", "native", "baked_render", "baked_native") and storage_mode != "job_local":
        raise ValueError("Render installation uses only the isolated job_local runtime")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must be a safe, unique name of at most 64 characters")
    if not image or image.startswith("REPLACE_") or any(character.isspace() for character in image):
        raise ValueError("Provide an approved OCI image reference")
    result_root = result_root.resolve()
    if not result_root.is_relative_to(Path("/jfs-public")) or not os.path.ismount("/jfs-public"):
        raise ValueError("A real /jfs-public mount is required; no home or bucket fallback")
    template = ROOT / "configs" / settings["template"]
    config = json.loads(template.read_text())
    validate_template(config, profile=profile)
    config["queue_name"] = queue
    validate_template(config, profile=profile)
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
    if storage_mode == "job_local":
        config["env"].update(EAI_STORAGE_MODE="job_local", EAI_RESULT_DIR=str(local_result_path(run_id)))
    if settings.get('gpus') == 3:
        config["env"]["EAI_EXPECTED_GPUS"] = "3"
    config['env'].update(network_environment)
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
        if storage_mode == "job_local":
            descriptor = os.open(run / "workspace/origin.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(origin_bytes)
        mounts = params.as_aidi_job_params().OPTIONAL.DATA_SPACE
        if mounts.INPUT != settings.get('input_buckets') or mounts.OUTPUT is not None:
            raise RuntimeError("Only the profile's explicitly reviewed read-only buckets may be mounted")
        write_private(run / "submit.private.json", config)
        manifest = {"run_id": run_id, "queue": queue, "cluster": cluster, "project_id": "horizon-labs",
                    "submitted": False, "execute": False, "inventory_only": profile in ("inventory", "image_inventory"),
                    "profile": profile,
                    "egress_proxy_configured": bool(network_environment),
                    "storage_mode": storage_mode,
                    "result_dir": str(result_dir), "workspace": prepared.workspace_folder,
                    "config_sha256": hashlib.sha256((run / "submit.private.json").read_bytes()).hexdigest(),
                    "artifact_sha256": {str(path.relative_to(run / "workspace")):
                                        hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in sorted((run / "workspace").rglob("*")) if path.is_file()},
                    "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted((ROOT / "cluster_eval").rglob("*")) if path.is_file()
                                      and "__pycache__" not in path.parts},
                    "pending": ["Explicit submission authorization", "Approved image and platform graphics runtime",
                                "Remote storage and export verification", "Isaac RGB rendering", "Qwen connectivity", "Native episode"]}
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
    parser.add_argument("--queue", choices=QUEUE_CLUSTERS, default=QUEUE)
    parser.add_argument("--storage-mode", choices=("jfs", "job_local"), default="jfs")
    parser.add_argument("--profile", choices=PROFILES, default="inventory")
    parser.add_argument('--egress-proxy', help='Explicit approved HTTP(S) proxy without credentials')
    args = parser.parse_args()
    print(json.dumps(prepare(run_id=args.run_id, image=args.image,
                             result_root=args.result_root, queue=args.queue,
                             storage_mode=args.storage_mode, profile=args.profile,
                             egress_proxy=args.egress_proxy), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
