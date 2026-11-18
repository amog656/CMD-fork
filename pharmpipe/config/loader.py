"""TCC loading, resolved provenance, and frozen-config hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .schema import TargetConfig


@dataclass(frozen=True)
class LoadedConfig:
    config: TargetConfig
    config_hash: str
    resolved: dict[str, Any]


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _with_provenance(config: TargetConfig) -> dict[str, Any]:
    values = config.model_dump(mode="json", by_alias=True)
    defaults: dict[str, str] = {}

    def record(model, prefix=""):
        for name, field in type(model).model_fields.items():
            value = getattr(model, name)
            external_name = field.alias or name
            path = f"{prefix}.{external_name}" if prefix else external_name
            if isinstance(value, BaseModel):
                record(value, path)
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                for index, item in enumerate(value):
                    record(item, f"{path}[{index}]")
            else:
                supplied = name in model.model_fields_set
                source = "provided in TCC" if supplied else "schema default from Implementation_Specification_V13.3_rev2.md section 2.4"
                if path.startswith("n_minima.") and not supplied:
                    source += "; pilot placeholder unless G4 or EF1 minimum"
                defaults[path] = source

    record(config)
    return {"values": values, "provenance": defaults}


def load_tcc(path: str | Path) -> LoadedConfig:
    tcc_path = Path(path)
    if not tcc_path.is_file():
        raise FileNotFoundError(f"TCC file does not exist: {tcc_path}")
    with tcc_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("TCC must contain a YAML mapping")
    raw = dict(raw)
    base = tcc_path.resolve().parent

    def resolve_path(value):
        if value is None:
            return None
        candidate = Path(value)
        return str((base / candidate).resolve()) if not candidate.is_absolute() else str(candidate.resolve())

    raw_target = dict(raw.get("target", {}))
    if raw_target.get("structure_path"):
        raw_target["structure_path"] = resolve_path(raw_target["structure_path"])
    raw["target"] = raw_target
    raw_reference = dict(raw.get("reference", {}))
    for key in ("actives_path", "inactives_path", "decoys_path"):
        if raw_reference.get(key):
            raw_reference[key] = resolve_path(raw_reference[key])
    raw["reference"] = raw_reference
    raw_thresholds = dict(raw.get("thresholds", {}))
    for key in (
        "precomputed_ensemble_dir",
        "pharmflow_repository_dir",
        "pharmflow_checkpoint_path",
        "pharmflow_python_executable",
        "gcpg_repository_dir",
        "gcpg_model_path",
        "gcpg_tokenizer_path",
    ):
        if raw_thresholds.get(key):
            value = raw_thresholds[key]
            if key in ("pharmflow_python_executable", "gcpg_python_executable") and not Path(value).is_absolute() and len(Path(value).parts) == 1:
                continue
            raw_thresholds[key] = resolve_path(value)
    raw["thresholds"] = raw_thresholds
    raw["off_targets"] = [
        {**item, "structure_path": resolve_path(item.get("structure_path"))}
        for item in raw.get("off_targets", [])
    ]
    config = TargetConfig.model_validate(raw)
    resolved = _with_provenance(config)
    return LoadedConfig(config=config, config_hash=_canonical_hash(resolved["values"]), resolved=resolved)


def freeze(loaded: LoadedConfig, output_path: str | Path) -> str:
    """Write the immutable evaluation freeze record and return its hash."""
    frozen = {
        "config_hash": loaded.config_hash,
        "values": loaded.resolved["values"],
        "provenance": loaded.resolved["provenance"],
    }
    frozen_hash = _canonical_hash(frozen)
    record = {**frozen, "frozen_hash": frozen_hash}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"freeze record already exists: {destination}")
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen_hash


def require_frozen(path: str | Path, config_hash: str) -> dict[str, Any]:
    freeze_path = Path(path)
    if not freeze_path.is_file():
        raise RuntimeError("test evaluation is forbidden until freeze() has been recorded")
    record = json.loads(freeze_path.read_text(encoding="utf-8"))
    if record.get("config_hash") != config_hash:
        raise ValueError("freeze record config hash does not match the active TCC")
    return record
