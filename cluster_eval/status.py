"""Read back one receipted project job without displaying raw SDK metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from cluster_eval.credentials import load_token
from cluster_eval.prepare import ROOT, write_private


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.parent != ROOT / "runs/cluster_eval":
        raise ValueError("Use a run directory prepared by this repository")
    receipt = json.loads((run_dir / "receipt.json").read_text())
    from robo_orchard_jobs.job_manager.aidi.manager import AIDIJobManager

    manager = AIDIJobManager(token=load_token())
    job = manager.get_job(receipt["job_id"])
    if job.job_name != receipt["job_name"] or job.queue_name != receipt["queue"]:
        raise ValueError("Platform job does not match this project's receipt")
    now = datetime.now(timezone.utc)
    fields = ("job_id", "job_name", "cluster", "status", "queue_name", "project_id",
              "priority", "created_at", "start_time", "finish_time", "resource_summary")
    report = {key: getattr(job, key, None) for key in fields}
    report.update(checked_at=now.isoformat(), native_evaluation_ready=False)
    result_dir = Path(receipt["result_dir"]).resolve()
    if not result_dir.is_relative_to(Path("/jfs-public")):
        raise ValueError("Unexpected result location")
    result = result_dir / "inventory.json"
    report["inventory_exists"] = result.is_file()
    if result.is_file():
        inventory = json.loads(result.read_text())
        if inventory.get("run_id") != run_dir.name:
            raise ValueError("Inventory belongs to another run")
        report["inventory"] = inventory
    output = run_dir / f"status-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    write_private(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
