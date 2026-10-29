"""Read-only AIDI manager CLI using the private project credential profile."""

from __future__ import annotations

import os
import sys

from cluster_eval.credentials import CONFIG_PATH, load_token


READ_ONLY_COMMANDS = {"list", "get", "logs", "queues", "queue_status", "version", "--help", "-h"}


def validate_arguments(arguments: list[str]) -> list[str]:
    if not arguments or arguments[0] != "manager":
        raise ValueError("Use manager for read-only queries; submit jobs through cluster_eval.submit_preflight")
    commands = arguments[1:] or ["--help"]
    if commands[0] not in READ_ONLY_COMMANDS:
        raise ValueError("This wrapper does not allow stop, delete, urgent, or submit operations")
    return commands


def main() -> None:
    try:
        arguments = validate_arguments(sys.argv[1:])
    except ValueError as error:
        raise SystemExit(str(error)) from None
    os.environ["AIDI_TOKEN"] = load_token()
    from aidisdk.infra.session import Session

    Session.config_file = str(CONFIG_PATH)
    from robo_orchard_jobs.job_manager.aidi.cli import aidi_job_manager_cli as entry

    sys.argv = ["project-aidi-manager", *arguments]
    entry()


if __name__ == "__main__":
    main()
