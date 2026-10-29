"""Private AIDI credentials, separate from the repository and shared SDK config."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import stat
import sys


CONFIG_PATH = Path.home() / ".config/embodied-agentic/aidi.json"


def save_token(token: str, path: Path = CONFIG_PATH) -> None:
    token = token.strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("Token must be nonempty and contain no whitespace")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
            or stat.S_IMODE(parent.st_mode) != 0o700):
        raise PermissionError("Credential directory must be owned by this user and mode 0700")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump({"token": token}, output)
        output.write("\n")


def load_token(path: Path = CONFIG_PATH) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r", encoding="utf-8") as source:
        info = os.fstat(source.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise PermissionError("Credential file must be owned by this user and mode 0600")
        token = json.load(source).get("token")
    if not isinstance(token, str) or not token or any(character.isspace() for character in token):
        raise ValueError("Invalid private credential format")
    return token


def main() -> None:
    if CONFIG_PATH.exists() or CONFIG_PATH.is_symlink():
        raise SystemExit("Private credential file already exists; refusing an implicit overwrite")
    if not sys.stdin.isatty():
        raise SystemExit("Use an interactive terminal so token input is not echoed")
    token = getpass.getpass("AIDI token (hidden input): ")
    save_token(token)
    print(f"Private credentials configured: {CONFIG_PATH}; token not displayed")


if __name__ == "__main__":
    main()
