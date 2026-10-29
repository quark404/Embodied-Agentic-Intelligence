"""Submit one frozen low-priority inventory job, with no credential in Job.env."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

from cluster_eval.credentials import CONFIG_PATH, load_token
from cluster_eval.prepare import QUEUE, ROOT, validate_template, write_private


ENV_KEYS = {"EAI_CLUSTER_CONFIRMED", "EAI_RUN_ID", "EAI_RESULT_DIR", "EAI_ORIGIN_SHA256",
            "PYTHONDONTWRITEBYTECODE"}


def verify_artifacts(workspace: Path, expected: dict) -> None:
    if not expected:
        raise ValueError("Prepared artifact hashes are required")
    actual = {}
    for path in workspace.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in the frozen upload workspace")
        if path.is_file():
            actual[str(path.relative_to(workspace))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Prepared artifacts changed; do not regenerate or submit this snapshot")


def job_environment(config: dict, token: str) -> dict:
    environment = dict(config["env"])
    if set(environment) != ENV_KEYS:
        raise ValueError("Inventory job environment differs from the fixed allowlist")
    if any(token in str(value) for value in environment.values()):
        raise ValueError("Credentials must not enter the workload environment")
    environment["EAI_CLUSTER_CONFIRMED"] = "1"
    return environment


def load_prepared(run_dir: Path) -> tuple[dict, dict]:
    if run_dir.parent != ROOT / "runs/cluster_eval":
        raise ValueError("Use a run directory prepared by this repository")
    config_path = run_dir / "submit.private.json"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Private configuration changed after preparation")
    config = json.loads(config_path.read_text())
    preview = {key: value for key, value in config.items() if key != "job_password"}
    validate_template(preview)
    template = json.loads((ROOT / "configs/aidi_5090_preflight.example.json").read_text())
    if config["cmd"] != template["cmd"] or set(preview) != set(template):
        raise ValueError("Only the reviewed inventory launch commands are allowed")
    workspace = run_dir / "workspace"
    if Path(config["workspace_folder"]).resolve() != workspace:
        raise ValueError("Workspace path mismatch")
    if config["job_name"] != f"eai_{manifest['run_id']}":
        raise ValueError("Job identity mismatch")
    if config["env"]["EAI_RUN_ID"] != manifest["run_id"]:
        raise ValueError("Runtime identity mismatch")
    result_dir = Path(config["env"]["EAI_RESULT_DIR"]).resolve()
    if not result_dir.is_relative_to(Path("/jfs-public")) or not os.path.ismount("/jfs-public"):
        raise ValueError("The prepared JFS result directory is unavailable")
    origin = (result_dir / "origin.json").read_bytes()
    if hashlib.sha256(origin).hexdigest() != config["env"]["EAI_ORIGIN_SHA256"]:
        raise ValueError("Original JFS marker changed")
    verify_artifacts(workspace, manifest["artifact_sha256"])
    return config, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config, manifest = load_prepared(run_dir)
    if not args.execute:
        print(json.dumps({"frozen_artifacts_verified": True, "submitted": False,
                          "run_id": manifest["run_id"], "gpus": 1, "minutes": config["wall_time"]}))
        return
    if (run_dir / "submit-attempt.json").exists() or (run_dir / "receipt.json").exists():
        raise FileExistsError("An attempt already exists; inspect the platform instead of retrying")
    for name, expected in (("aidisdk", "0.20.23"),
                           ("robo_orchard_jobs", "0.1.0.dev20261026204009")):
        if version(name) != expected:
            raise RuntimeError(f"Unreviewed SDK version for {name}")
    token = load_token()
    environment = job_environment(config, token)
    from aidisdk import AIDIClient
    from aidisdk.infra.session import Session
    from aidisdk.compute.job_abstract import RunningResourceConfig, StartUpConfig
    from aidisdk.compute.package_abstract import CodePackageConfig, LocalPackageItem
    from robo_orchard_jobs.job_manager.aidi.manager import AIDIJobManager
    from robo_orchard_jobs.job_submit.aidi.job_config import AIDIJobParams

    Session.config_file = str(CONFIG_PATH)
    manager = AIDIJobManager(token=token)
    existing = manager.list_jobs(cluster="acloud-langfang", queue=QUEUE, ongoing=True,
                                 history=True, current_user_only=True, recent_days=7, limit=200)
    terminal = {"succeeded", "failed", "stopped", "completed", "cancelled", "canceled"}
    conflicts = [item.job_id for item in existing if item.job_name == config["job_name"] or
                 ((item.job_name or "").startswith("eai_") and (item.status or "").lower() not in terminal)]
    if conflicts:
        raise RuntimeError("Existing project evaluation job found; do not duplicate or stop it")
    for key in ("no_proxy", "NO_PROXY"):
        values = [value for value in os.environ.get(key, "").split(",") if value]
        suffix = ".oss-cn-beijing-internal.aliyuncs.com"
        if suffix not in values:
            values.append(suffix)
        os.environ[key] = ",".join(values)
    native = AIDIJobParams.from_str((run_dir / "workspace/job_config.yaml").read_text(), format="yaml")
    client = AIDIClient(token=token)
    package = LocalPackageItem(lpath=str(run_dir / "workspace"),
                              encrypt_passwd=config["job_password"], follow_softlink=False)
    job = client.single_job.new_job(
        job_name=config["job_name"], job_type="train", project_id=config["project_id"],
        queue_name=QUEUE, priority=config["priority"],
        running_resource=RunningResourceConfig(docker_image=config["docker_image"], instance=1,
            gpu=1, cpu=config["cpu_per_worker"], cpu_mem_ratio=config["cpu_mem_ratio"],
            walltime=config["wall_time"]),
        startup=StartUpConfig(command=native.REQUIRED.RUN_SCRIPTS, startup_dir=package),
        code_package=CodePackageConfig(raw_package=package.set_as_startup_dir()),
        mount=[], env=dict(environment), max_retries=0)
    job.env = environment
    if token in json.dumps(job.env):
        raise RuntimeError("SDK injected a credential into the job environment")
    verify_artifacts(run_dir / "workspace", manifest["artifact_sha256"])
    write_private(run_dir / "submit-attempt.json", {
        "started_at": datetime.now(timezone.utc).isoformat(), "job_name": config["job_name"],
        "queue": QUEUE, "credential_forwarded": False, "automatic_resubmission": False})
    try:
        handle = client.single_job.submit_job(job)
    except Exception as error:
        write_private(run_dir / "submission-error.json", {"error_type": type(error).__name__,
                      "status": "unknown_until_platform_query", "retry_allowed": False})
        raise SystemExit("Submission did not return a receipt; inspect platform before any retry") from None
    receipt = {"accepted_at": datetime.now(timezone.utc).isoformat(), "job_id": handle.job_id,
               "job_name": handle.job_name, "queue": QUEUE, "credential_forwarded": False,
               "expected_gpus": 1, "wall_time_minutes": config["wall_time"], "priority": "low",
               "result_dir": manifest["result_dir"], "native_evaluation_ready": False}
    write_private(run_dir / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
