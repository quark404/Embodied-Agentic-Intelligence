from __future__ import annotations

import json
import re
from pathlib import Path


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w-]+", text.lower()))


class MemoryBank:
    """Episode-level facts with transparent lexical retrieval and split provenance."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records = []
        if self.path and self.path.exists():
            self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def add(self, *, task_id: str, split: str, cabinet: str, code: str, success: bool):
        if split != "train" or not success:
            raise ValueError("Only successful training episodes can enter the memory bank")
        record = {"task_id": task_id, "split": split, "cabinet": cabinet, "code": code}
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    def retrieve(self, cabinet: str, k: int = 1) -> list[dict]:
        query = tokens(cabinet)
        ranked = sorted(
            (r for r in self.records if r["split"] == "train"),
            key=lambda r: (len(query & tokens(r["cabinet"])), r["cabinet"] == cabinet),
            reverse=True,
        )
        return [r for r in ranked if r["cabinet"] == cabinet][:k]
