"""Immutable registry for frozen model versions used in Mode D revalidation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelVersion:
    version_id: str
    artifact_hash: str
    parent_version: str | None
    training_data_ids: tuple[str, ...]
    excluded_test_ids: tuple[str, ...]
    ood_cutoff: float
    freeze_hash: str


class ModelVersionRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"versions": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def register(self, record: ModelVersion) -> str:
        overlap = set(record.training_data_ids) & set(record.excluded_test_ids)
        if overlap:
            raise ValueError(f"held-out test items cannot enter a model version: {sorted(overlap)}")
        if not record.version_id or not record.artifact_hash or not record.freeze_hash:
            raise ValueError("version id, artifact hash, and freeze hash are required")
        data = self._read()
        if record.version_id in data["versions"]:
            raise FileExistsError(f"model version already registered: {record.version_id}")
        canonical = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
        version_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        data["versions"][record.version_id] = {**asdict(record), "version_hash": version_hash}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return version_hash

    def get(self, version_id: str) -> dict:
        try:
            return self._read()["versions"][version_id]
        except KeyError as exc:
            raise KeyError(f"model version is not registered: {version_id}") from exc
