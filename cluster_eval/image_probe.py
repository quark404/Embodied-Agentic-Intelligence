"""Inspect an allocated prebuilt image without contacting any download endpoint."""

from pathlib import Path
import sys


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_eval.probe import main


if __name__ == '__main__':
    main(check_network=False)
