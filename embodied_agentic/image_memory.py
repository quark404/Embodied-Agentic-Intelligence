"""Task-scoped recall of immutable images captured at subtask entry.

Retrieval is by task/subtask identity or exact metadata tags. No embeddings,
semantic search, or outcome labels are involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    task_id: str
    episode_id: str
    subtask_id: str
    instruction: str
    tags: tuple[str, ...]
    camera: str
    mime: str
    sha256: str
    filename: str


@dataclass(frozen=True)
class RecalledImage:
    record: ImageRecord
    data: bytes


class SubtaskImageMemory:
    """A run-local bank of initial images; existing snapshots are never replaced."""

    def __init__(self, root: str | Path, *, max_image_bytes: int = 10_000_000):
        self.root = Path(root)
        self.max_image_bytes = max_image_bytes
        self.manifest = self.root / "manifest.jsonl"
        self.records: list[ImageRecord] = []
        if self.manifest.exists():
            for line in self.manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    item["tags"] = tuple(item["tags"])
                    self.records.append(ImageRecord(**item))

    @staticmethod
    def _mime(data: bytes) -> str:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        raise ValueError("Initial image must be PNG or JPEG bytes")

    @staticmethod
    def _write_atomic(path: Path, data: bytes):
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp_name = stream.name
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temp_name, path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)

    def capture(self, *, task_id: str, episode_id: str, subtask_id: str, instruction: str,
                tags: tuple[str, ...], images: dict[str, bytes]) -> list[ImageRecord]:
        if not task_id or not episode_id or not subtask_id or not images:
            raise ValueError("A task, episode, subtask and initial image are required")
        previous = [r for r in self.records if r.task_id == task_id and
                    r.episode_id == episode_id and r.subtask_id == subtask_id]
        if previous:
            if any(r.instruction != instruction or r.tags != tuple(tags) for r in previous):
                raise ValueError("Reused subtask ID has different instructions or tags")
            if set(images) != {r.camera for r in previous}:
                raise ValueError("Cannot change the cameras of an existing subtask snapshot")
            # A replan may revisit the subtask; retain its original entry image.
            return previous
        prepared: list[tuple[ImageRecord, bytes]] = []
        for camera, data in sorted(images.items()):
            if not camera or not isinstance(data, bytes) or not 0 < len(data) <= self.max_image_bytes:
                raise ValueError("Invalid camera name or image size")
            mime = self._mime(data)
            digest = hashlib.sha256(data).hexdigest()
            filename = f"{hashlib.sha256((task_id + chr(0) + episode_id + chr(0) + subtask_id + chr(0) + camera).encode()).hexdigest()}-{digest[:16]}" + (".png" if mime == "image/png" else ".jpg")
            prepared.append((ImageRecord(task_id, episode_id, subtask_id, instruction, tuple(tags),
                                         camera, mime, digest, filename), data))
        self.root.mkdir(parents=True, exist_ok=True)
        for record, data in prepared:
            path = self.root / record.filename
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Image artifact collision")
            if not path.exists():
                self._write_atomic(path, data)
        all_records = self.records + [record for record, _ in prepared]
        manifest_bytes = "".join(json.dumps(asdict(record), ensure_ascii=False) + "\n"
                                 for record in all_records).encode("utf-8")
        self._write_atomic(self.manifest, manifest_bytes)
        self.records.extend(record for record, _ in prepared)
        return [record for record, _ in prepared]

    def recall(self, *, task_id: str, episode_id: str, subtask_id: str | None = None,
               tags: tuple[str, ...] = (), limit: int = 4) -> list[RecalledImage]:
        if not task_id or not episode_id or limit < 1 or (subtask_id is None and not tags):
            raise ValueError("Recall needs task/episode ID and subtask ID or tags")
        found = [r for r in self.records if r.task_id == task_id and r.episode_id == episode_id and
                 (subtask_id is None or r.subtask_id == subtask_id) and
                 (not tags or set(tags).issubset(r.tags))]
        result = []
        for record in found[:limit]:
            if Path(record.filename).name != record.filename:
                raise ValueError("Unsafe image artifact name")
            data = (self.root / record.filename).read_bytes()
            if hashlib.sha256(data).hexdigest() != record.sha256:
                raise ValueError("Stored image checksum mismatch")
            result.append(RecalledImage(record, data))
        return result
