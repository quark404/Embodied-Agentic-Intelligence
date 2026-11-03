"""Read-only supervision of explicitly receipted jobs; never submit or stop jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from cluster_eval.credentials import load_token
from cluster_eval.prepare import ROOT, queue_cluster, write_private
from cluster_eval.status import collect_status


TERMINAL = {"succeeded", "failed", "stopped", "completed", "cancelled", "canceled"}
SUCCESS = {"succeeded", "completed"}


def platform_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return parsed.astimezone(timezone.utc)


def summarize(reports: list[dict]) -> dict:
    jobs = []
    for report in reports:
        status = str(report.get("status") or "unknown").lower()
        inventory = report.get("inventory", {})
        accepted = platform_time(report.get("accepted_at"))
        started = platform_time(report.get("start_time"))
        finished = platform_time(report.get("finish_time"))
        storage_verified = (inventory.get("jfs_origin_verified") is True
                            if inventory.get("storage_mode", "jfs") == "jfs" else
                            inventory.get("storage_mode") == "job_local"
                            and inventory.get("storage_origin_verified") is True
                            and report.get("inventory_export_verified") is True)
        topology_verified = (inventory.get("single_5090_visible") is True
                             or report.get('expected_gpus') == 3
                             and inventory.get('expected_gpus') == 3
                             and inventory.get('gpu_topology_verified') is True)
        passed = (status in SUCCESS and inventory.get("inventory_passed") is True
                  and topology_verified
                  and storage_verified)
        jobs.append({"job_id": report["job_id"], "queue": report["queue_name"],
                     "status": report.get("status"), "terminal": status in TERMINAL,
                     "render_verified": (passed and report.get("render_export_verified") is True
                                         and report.get("render_result", {}).get("render_passed") is True),
                     "native_verified": (passed and report.get('native_evaluation_ready') is True),
                     "inventory_verified": passed, "finish_time": finished.isoformat() if finished else None,
                     "queue_seconds": (started - accepted).total_seconds() if started and accepted else None,
                     "runtime_seconds": (finished - started).total_seconds() if finished and started else None,
                     "accept_to_finish_seconds": (finished - accepted).total_seconds() if finished and accepted else None})
    completed = sorted((job for job in jobs if job["terminal"] and job["finish_time"]),
                       key=lambda job: job["finish_time"])
    verified = [job for job in completed if job["inventory_verified"]]
    return {"jobs": jobs, "first_terminal": completed[0] if completed else None,
            "first_verified_inventory": verified[0] if verified else None,
            "all_terminal": bool(jobs) and all(job["terminal"] for job in jobs),
            "native_evaluation_ready": any(job['native_verified'] for job in jobs),
            "comparison_note": "Submissions were staggered; first completion is not a controlled queue-speed comparison"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--max-polls", type=int, default=360)
    args = parser.parse_args()
    if args.interval < 60 or not 1 <= args.max_polls <= 1440:
        parser.error("Use at least 60 seconds between polls and 1..1440 polls")
    runs = [path.resolve() for path in args.run_dir]
    if len(runs) != len(set(runs)) or not 1 <= len(runs) <= 3:
        parser.error("Provide 1..3 distinct, explicitly receipted runs")
    output = args.output_dir.resolve()
    if output.parent != ROOT / "runs/cluster_eval":
        parser.error("Use a new supervision directory under runs/cluster_eval")
    for run in runs:
        if run.parent != ROOT / "runs/cluster_eval":
            parser.error("Only this repository's receipted runs may be supervised")
        queue_cluster(json.loads((run / "receipt.json").read_text())["queue"])
    output.mkdir(mode=0o700)
    write_private(output / "monitor.json", {"pid": os.getpid(), "runs": [str(run) for run in runs],
                  "interval_seconds": args.interval, "max_polls": args.max_polls,
                  "read_only": True, "started_at": datetime.now(timezone.utc).isoformat()})
    from robo_orchard_jobs.job_manager.aidi.manager import AIDIJobManager

    manager = AIDIJobManager(token=load_token())
    seen_events = set()
    for poll in range(args.max_polls):
        reports = []
        errors = []
        for run in runs:
            try:
                reports.append(collect_status(run, manager))
            except Exception as error:
                errors.append({"run_id": run.name, "error_type": type(error).__name__})
        summary = summarize(reports)
        if errors:
            summary["all_terminal"] = False
        now = datetime.now(timezone.utc)
        snapshot = {"checked_at": now.isoformat(), "poll": poll, "reports": reports,
                    "errors": errors, **summary}
        write_private(output / f"poll-{poll:04d}.json", snapshot)
        for event in ("first_terminal", "first_verified_inventory"):
            if summary[event] and event not in seen_events:
                write_private(output / f"{event}.json", {"detected_at": now.isoformat(), **summary[event]})
                seen_events.add(event)
        print(json.dumps({"checked_at": now.isoformat(), "poll": poll, "errors": errors, **summary}), flush=True)
        if summary["all_terminal"]:
            break
        if poll + 1 < args.max_polls:
            time.sleep(args.interval)
    write_private(output / "monitor-ended.json", {"ended_at": datetime.now(timezone.utc).isoformat(),
                  "reason": "all_terminal" if summary["all_terminal"] else "observation_budget_exhausted",
                  "jobs_terminated_by_monitor": False})


if __name__ == "__main__":
    main()
