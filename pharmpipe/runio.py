"""Immutable run folders, JSON manifests, and gate-log persistence."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy
import pydantic
import yaml

from . import __version__
from .config.loader import LoadedConfig
from .data.models import GateResult


def library_versions() -> dict[str, str]:
    versions = {"pharmpipe": __version__, "python": sys.version.split()[0]}
    for package in ("numpy", "pydantic", "PyYAML", "scipy", "scikit-learn", "rdkit"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    versions["numpy_runtime"] = numpy.__version__
    versions["pydantic_runtime"] = pydantic.__version__
    versions["pyyaml_runtime"] = yaml.__version__
    return versions


def create_run_folder(output_root: str | Path, loaded: LoadedConfig, tcc_path: str | Path, input_paths: list[str | Path] | None = None) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(loaded.config_hash.encode("utf-8"))
    for input_path in input_paths or []:
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"run input does not exist: {path}")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    run_hash = digest.hexdigest()
    run_dir = root / f"run_{run_hash[:16]}"
    if run_dir.exists():
        raise FileExistsError(f"run folder already exists; refusing to overwrite: {run_dir}")
    run_dir.mkdir()
    for name in ("config", "step0", "step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "gates", "report"):
        (run_dir / name).mkdir()
    shutil.copy2(tcc_path, run_dir / "config" / "tcc.yaml")
    (run_dir / "config" / "resolved_config.json").write_text(
        json.dumps({**loaded.resolved, "config_hash": loaded.config_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def prepare_stage_dir(stage_dir: str | Path) -> Path:
    """Prepare a new or pre-created empty stage directory without overwriting outputs."""
    directory = Path(stage_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"stage directory already contains outputs: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_manifest(
    stage_dir: str | Path,
    *,
    config_hash: str,
    seed: int,
    counts_in: dict[str, int] | None = None,
    counts_out: dict[str, int] | None = None,
    dropped: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_hash": config_hash,
        "seed": seed,
        "versions": library_versions(),
        "counts_in": counts_in or {},
        "counts_out": counts_out or {},
        "dropped_items": dropped or [],
        "extra": extra or {},
    }
    path = directory / "manifest.json"
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_gate_result(gate_log_path: str | Path, result: GateResult) -> None:
    path = Path(gate_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "gate_id": result.gate_id,
            "outcome": result.outcome.value,
            "consequence_type": result.consequence_type.value,
            "statistics": result.statistics,
            "sample_sizes": result.sample_sizes,
            "threshold": result.threshold,
            "config_hash": result.config_hash,
            "reason": result.reason,
        }, sort_keys=True) + "\n")

