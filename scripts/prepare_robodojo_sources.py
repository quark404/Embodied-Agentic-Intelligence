"""Fetch pinned, source-only checkouts into a NEW directory; never install deps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "configs/robodojo_versions.json"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def checkout(root: Path, spec: dict, *, sparse: bool = False) -> None:
    root.mkdir(exist_ok=False)
    git(root, "init", "-q")
    git(root, "remote", "add", "origin", spec["url"])
    git(root, "fetch", "--depth=1", "--filter=blob:none", "origin", spec["revision"])
    if sparse:
        git(root, "sparse-checkout", "init", "--cone")
        git(root, "sparse-checkout", "set", "client_server", "utils")
    git(root, "checkout", "--detach", "FETCH_HEAD")
    if git(root, "rev-parse", "HEAD") != spec["revision"]:
        raise RuntimeError(f"Revision mismatch at {root}")


def prepare(destination: Path) -> dict:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to change an existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK_PATH.read_text())
    checkout(destination, lock["robodojo"])
    for path, key in (("XPolicyLab", "xpolicylab"),
                      ("third_party/IsaacLab", "isaaclab"),
                      ("third_party/curobo", "curobo")):
        recorded = git(destination, "ls-tree", "HEAD", path).split()[2]
        if recorded != lock[key]["revision"]:
            raise RuntimeError(f"Lock does not match RoboDojo gitlink: {path}")
    xpl_root = destination / "XPolicyLab"
    if xpl_root.is_dir() and not any(xpl_root.iterdir()):
        xpl_root.rmdir()
    checkout(xpl_root, lock["xpolicylab"], sparse=True)
    policy = xpl_root / "policy/Embodied_Agentic"
    policy.parent.mkdir(exist_ok=True)
    policy.symlink_to(PROJECT_ROOT / "integrations/xpolicylab/Embodied_Agentic",
                      target_is_directory=True)
    return {"robodojo_root": str(destination.resolve()),
            "source_only": True, "policy_link": str(policy),
            "robodojo_revision": lock["robodojo"]["revision"],
            "xpolicylab_revision": lock["xpolicylab"]["revision"],
            "simulator_installed": False, "assets_downloaded": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination.absolute()), indent=2))


if __name__ == "__main__":
    main()
