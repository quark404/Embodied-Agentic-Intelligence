"""Read back one receipted project job without displaying raw SDK metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from cluster_eval.credentials import load_token
from cluster_eval.prepare import ROOT, write_private
from cluster_eval.native_artifacts import validate_native_report
from cluster_eval.rendering import validate_render_report
from cluster_eval.storage import RECORD_PREFIXES, parse_record, persist_bytes, persist_export


def collect_status(run_dir: Path, manager) -> dict:
    run_dir = run_dir.resolve()
    if run_dir.parent != ROOT / "runs/cluster_eval":
        raise ValueError("Use a run directory prepared by this repository")
    receipt = json.loads((run_dir / "receipt.json").read_text())
    job = manager.get_job(receipt["job_id"])
    if job.job_name != receipt["job_name"] or job.queue_name != receipt["queue"]:
        raise ValueError("Platform job does not match this project's receipt")
    now = datetime.now(timezone.utc)
    fields = ("job_id", "job_name", "cluster", "status", "queue_name", "project_id",
              "priority", "created_at", "start_time", "finish_time", "resource_summary")
    report = {key: getattr(job, key, None) for key in fields}
    report.update(checked_at=now.isoformat(), accepted_at=receipt["accepted_at"],
                  run_id=run_dir.name, native_evaluation_ready=False)
    result_dir = Path(receipt["result_dir"]).resolve()
    if not result_dir.is_relative_to(Path("/jfs-public")):
        raise ValueError("Unexpected result location")
    result = result_dir / "inventory.json"
    if receipt.get("storage_mode") == "job_local":
        if hashlib.sha256((result_dir / "origin.json").read_bytes()).hexdigest() != receipt["origin_sha256"]:
            raise ValueError("Local submission origin changed before artifact retrieval")
        record_types = {"native": ("inventory", "render", "native"),
                        "baked_native": ("inventory", "render", "native"),
                        "baked_render": ("inventory", "render"),
                        "render": ("inventory", "render")}.get(receipt.get("profile"), ("inventory",))
        fetched_log = None
        for record_type in record_types:
            exported = result_dir / f"{record_type}-export.json"
            envelope = None
            prefix = RECORD_PREFIXES[record_type]
            try:
                if exported.is_file():
                    text = prefix + json.dumps(json.loads(exported.read_text()))
                elif str(report.get("status", "")).lower() not in ("queuing", "pending", "unknown"):
                    if fetched_log is None:
                        logs = manager.get_job_logs(receipt["job_id"], cluster=receipt["cluster"],
                            queue=receipt["queue"], mode="filter",
                            expr="EAI_" if len(record_types) > 1 else prefix.strip(), cache="off")
                        fetched_log = logs.content or ""
                    text = fetched_log
                else:
                    text = ""
                envelope = parse_record(text, run_dir.name, receipt["origin_sha256"], record_type=record_type)
                if envelope is not None:
                    payload = envelope[record_type]
                    if record_type == "render":
                        images = validate_render_report(payload)
                        for name, content in images.items():
                            persist_bytes(result_dir / f"render_{name}.png", content)
                        report["render_result"] = {**payload, "frames": {
                            name: {key: value for key, value in frame.items() if key != "png_base64"}
                            for name, frame in payload.get("frames", {}).items()}}
                    elif record_type == "native":
                        files, verdict = validate_native_report(payload)
                        artifact_root = result_dir / 'native-artifacts'
                        if artifact_root.resolve() != artifact_root:
                            raise ValueError('Native artifact destination must not be a symlink')
                        artifact_root.mkdir(mode=0o700, exist_ok=True)
                        for name, content in files.items():
                            destination = artifact_root / name
                            if destination.resolve() != destination:
                                raise ValueError('Native artifact parent identity changed')
                            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                            persist_bytes(destination, content)
                        report['native_result'] = {key: value for key, value in payload.items()
                                                   if key not in ('files', 'diagnostic_files')}
                        report['native_result']['verdict'] = verdict
                        report['native_artifact_files'] = sorted(files)
                    persist_export(exported, envelope)
                    persist_export(result_dir / f"{record_type}.json", payload)
            except Exception as error:
                envelope = None
                report[f"{record_type}_export_error_type"] = type(error).__name__
            report[f"{record_type}_export_verified"] = envelope is not None
        report["storage_mode"] = "job_local"
    report["inventory_exists"] = result.is_file()
    if result.is_file():
        inventory = json.loads(result.read_text())
        if inventory.get("run_id") != run_dir.name:
            raise ValueError("Inventory belongs to another run")
        report["inventory"] = inventory
    report['expected_gpus'] = receipt.get('expected_gpus', 1)
    report['native_evaluation_ready'] = (
        str(report.get('status', '')).lower() in ('succeeded', 'completed')
        and report.get('inventory_export_verified') is True
        and report.get('inventory', {}).get('inventory_passed') is True
        and report.get('inventory', {}).get('gpu_topology_verified') is True
        and report.get('inventory', {}).get('expected_gpus') == report['expected_gpus'] == 3
        and report.get('render_export_verified') is True
        and report.get('render_result', {}).get('render_passed') is True
        and report.get('native_export_verified') is True
        and report.get('native_result', {}).get('verdict', {}).get('native_success') is True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    from robo_orchard_jobs.job_manager.aidi.manager import AIDIJobManager

    run_dir = args.run_dir.resolve()
    report = collect_status(run_dir, AIDIJobManager(token=load_token()))
    now = datetime.now(timezone.utc)
    output = run_dir / f"status-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    write_private(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
