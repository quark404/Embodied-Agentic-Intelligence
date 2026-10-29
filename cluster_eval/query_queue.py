"""Read only the requested AIDI queue; never submit or change authentication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from cluster_eval.credentials import CONFIG_PATH, load_token

QUEUE = "project-5090-robot-lab-acloud-langfang"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; existing audit records are not replaced")
    import yaml
    from robo_orchard_jobs.job_manager.aidi.manager import AIDIJobManager

    credential = os.environ.get("AIDI_TOKEN")
    if not credential and CONFIG_PATH.exists():
        credential = load_token()
    config = Path.home() / ".aidisdk/config.yaml"
    if not credential and config.is_file():
        credential = (yaml.safe_load(config.read_text()) or {}).get("token")
    if not credential:
        raise SystemExit("No existing AIDI credential; configure through your approved private channel")
    manager = AIDIJobManager(token=credential)
    try:
        queues = [item for item in manager.list_queues(all_visible=True) if item.name == QUEUE]
        fields = ("name", "cluster", "project_id", "tenant", "category", "type", "queue_version")
        status = [item.model_dump(exclude={"raw"})
                  for item in manager.list_queue_status(queue=QUEUE)]
        report = {"checked_at": datetime.now(timezone.utc).isoformat(),
                  "queue": [{key: getattr(item, key, None) for key in fields} for item in queues],
                  "status": status, "submitted": False,
                  "note": "Visibility and current capacity do not prove submission permission or immediate allocation"}
    except Exception as error:
        raise SystemExit(f"Queue query failed ({type(error).__name__}); credentials not displayed") from None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Matched {len(queues)} target queue(s); read-only report: {args.output}")


if __name__ == "__main__":
    main()
