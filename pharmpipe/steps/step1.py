"""Step 1: leakage-checked mock sampling or ingestion of a precomputed ensemble."""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..adapters.asamt import MockASAMTAdapter
from ..config.loader import LoadedConfig
from ..leakage import ExclusionSet, LigandRecord, assert_no_excluded_seed
from ..runio import prepare_stage_dir, write_manifest
from .structure_io import read_pdb_atoms, write_pdb_atoms


def _atom_signature(atoms: list[dict]) -> list[tuple[str, ...]]:
    """Identity and ordering required to safely pair coordinates across frames."""
    return [
        (atom["record"], atom["atom"], atom["resname"], atom["chain"], atom["resid"], atom["element"])
        for atom in atoms
    ]


def _load_precomputed_frames(directory: Path, expected_count: int) -> list[Path]:
    marker = directory / ".done"
    if not marker.is_file():
        raise ValueError(f"precomputed ensemble is missing its .done marker: {directory}")
    match = re.fullmatch(r"COMPLETED_(\d+)_FRAMES\s*", marker.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"precomputed ensemble has an invalid .done marker: {marker}")
    frames = sorted(directory.glob("conf_*.pdb"))
    declared_count = int(match.group(1))
    if declared_count != expected_count or len(frames) != expected_count:
        raise ValueError(
            f"precomputed ensemble must contain exactly {expected_count} frames "
            f"(marker says {declared_count}; found {len(frames)} PDB files)"
        )
    reference_signature = None
    for frame in frames:
        atoms = read_pdb_atoms(frame)
        if not atoms:
            raise ValueError(f"precomputed frame contains no readable atoms: {frame}")
        signature = _atom_signature(atoms)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"precomputed frames do not have matching atom identities/order: {frame}")
    return frames


@dataclass
class Step1Result:
    ensemble_table: Path
    conformer_paths: list[Path]


def run_step1(
    loaded: LoadedConfig, *, seeds: list[dict], held_out: list[LigandRecord], exclusion_set: ExclusionSet,
    n_samples_per_seed: int, temperature: float, output_dir: str | Path,
    adapter: MockASAMTAdapter | None = None, precomputed_ensemble_dir: str | Path | None = None,
) -> Step1Result:
    if n_samples_per_seed <= 0 or temperature <= 0:
        raise ValueError("n_samples_per_seed and temperature must be positive config values")
    # Both leakage operations are required before any sampler call.
    if set(x.ligand_id for x in held_out) - exclusion_set.excluded_ids:
        raise ValueError("exclusion set does not contain every held-out active")
    assert_no_excluded_seed(seeds, exclusion_set)
    if precomputed_ensemble_dir is not None and len(seeds) != 1:
        raise ValueError("a precomputed ensemble directory must correspond to exactly one seed")
    sampler = adapter or MockASAMTAdapter()
    out = Path(output_dir)
    prepare_stage_dir(out)
    rows, conformers = [], []
    seed_offset = 0
    stripped_ligand_atoms = 0
    for seed_record in seeds:
        seed_type = seed_record["seed_type"]
        if seed_type not in {"apo-seeded", "holo-seeded"}:
            raise ValueError("aSAMt seeds must be explicitly typed apo-seeded or holo-seeded")
        if seed_type == "holo-seeded" and (not seed_record.get("ligand_id") or not seed_record.get("ligand_resname")):
            raise ValueError("holo seeds require ligand_id and ligand_resname so leakage can be checked and the ligand removed")
        source_path = Path(seed_record["structure_path"])
        atoms = read_pdb_atoms(source_path)
        if seed_type == "holo-seeded":
            assert_no_excluded_seed([seed_record], exclusion_set)
            ligand_resname = str(seed_record["ligand_resname"])
            ligand_atoms = [atom for atom in atoms if atom["resname"] == ligand_resname and atom["record"] == "HETATM"]
            if not ligand_atoms:
                raise ValueError(f"configured holo ligand residue {ligand_resname!r} was not found in {source_path}")
            stripped_ligand_atoms += len(ligand_atoms)
            atoms = [atom for atom in atoms if atom not in ligand_atoms]
        else:
            atoms = [atom for atom in atoms if atom["record"] == "ATOM"]
        if not atoms:
            raise ValueError(f"seed structure has no readable PDB atoms: {source_path}")

        if precomputed_ensemble_dir is not None:
            frames = _load_precomputed_frames(Path(precomputed_ensemble_dir), n_samples_per_seed)
            for frame_path in frames:
                state_id = f"conf_{len(rows):04d}"
                path = out / f"{state_id}.pdb"
                shutil.copy2(frame_path, path)
                rows.append({
                    "state_id": state_id,
                    "seed_id": str(seed_record["seed_id"]),
                    "seed_type": seed_type,
                    "temperature": temperature,
                    "file_path": path.name,
                    "sampler_version": "aSAM-precomputed",
                })
                conformers.append(path)
            seed_offset += n_samples_per_seed
            continue

        base = np.asarray([atom["xyz"] for atom in atoms], dtype=float)
        samples = sampler.sample(base, n_samples_per_seed, temperature, loaded.config.seed + seed_offset)
        for sample_index, coordinates in enumerate(samples):
            state_id = f"conf_{len(rows):04d}"
            path = out / f"{state_id}.pdb"
            sampled_atoms = [dict(atom, xyz=coordinates[index].tolist()) for index, atom in enumerate(atoms)]
            write_pdb_atoms(path, sampled_atoms)
            rows.append({"state_id": state_id, "seed_id": str(seed_record["seed_id"]), "seed_type": seed_type, "temperature": temperature, "file_path": path.name, "sampler_version": sampler.version})
            conformers.append(path)
        seed_offset += n_samples_per_seed
    table = out / "ensemble_table.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state_id", "seed_id", "seed_type", "temperature", "file_path", "sampler_version"])
        writer.writeheader()
        writer.writerows(rows)
    write_manifest(
        out,
        config_hash=loaded.config_hash,
        seed=loaded.config.seed,
        counts_in={"seeds": len(seeds)},
        counts_out={"conformers": len(rows)},
        dropped=[],
        extra={
            "stripped_holo_ligand_atoms": stripped_ligand_atoms,
            "ensemble_source": "precomputed-aSAM" if precomputed_ensemble_dir is not None else sampler.version,
            "precomputed_ensemble_dir": str(Path(precomputed_ensemble_dir).resolve()) if precomputed_ensemble_dir is not None else None,
        },
    )
    return Step1Result(table, conformers)
