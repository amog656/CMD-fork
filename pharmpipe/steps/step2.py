"""Step 2: protein geometry, heavy-atom clash screening, and Q gate."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..config.loader import LoadedConfig
from ..data.models import GateResult
from ..gates.q import evaluate_q
from ..runio import prepare_stage_dir, write_manifest
from .structure_io import read_pdb_atoms


# Bond intervals are deliberately broad screening ranges. They detect broken
# coordinates; they are not a replacement for a full structure validator.
_BACKBONE_BONDS = {
    ("N", "CA"): (1.20, 1.70),
    ("CA", "C"): (1.20, 1.70),
    ("C", "O"): (1.00, 1.60),
    ("C", "N_NEXT"): (1.10, 1.80),
}
_VDW_RADII_A = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}


@dataclass
class Step2Result:
    gate: GateResult
    surviving_paths: list[Path]
    qc_table: Path


def _residue_key(atom: dict) -> tuple[str, str]:
    return atom["chain"], atom["resid"]


def _backbone_qc(atoms: list[dict]) -> tuple[bool, int, list[str]]:
    residues: dict[tuple[str, str], dict[str, dict]] = {}
    residue_order: dict[str, list[str]] = {}
    reasons: list[str] = []
    for atom in atoms:
        key = _residue_key(atom)
        by_name = residues.setdefault(key, {})
        if atom["atom"] in by_name:
            reasons.append("duplicate_atom_identity")
        by_name[atom["atom"]] = atom
        if key[1] not in residue_order.setdefault(key[0], []):
            residue_order[key[0]].append(key[1])

    def distance(a: dict, b: dict) -> float:
        return float(np.linalg.norm(np.asarray(a["xyz"], dtype=float) - np.asarray(b["xyz"], dtype=float)))

    for key, names in residues.items():
        required = ("N", "CA", "C")
        if any(name not in names for name in required):
            reasons.append(f"missing_backbone_atoms:{key[0]}:{key[1]}")
            continue
        for pair in (("N", "CA"), ("CA", "C")):
            low, high = _BACKBONE_BONDS[pair]
            measured = distance(names[pair[0]], names[pair[1]])
            if not low <= measured <= high:
                reasons.append(f"abnormal_{pair[0]}_{pair[1]}_bond:{key[0]}:{key[1]}")
        if "O" in names:
            low, high = _BACKBONE_BONDS[("C", "O")]
            if not low <= distance(names["C"], names["O"]) <= high:
                reasons.append(f"abnormal_C_O_bond:{key[0]}:{key[1]}")

    chain_breaks = 0
    for chain, residue_ids in residue_order.items():
        for current, following in zip(residue_ids, residue_ids[1:]):
            try:
                consecutive = int(following) == int(current) + 1
            except ValueError:
                consecutive = False
            if not consecutive:
                continue
            left = residues[(chain, current)]
            right = residues[(chain, following)]
            if "C" not in left or "N" not in right:
                chain_breaks += 1
                continue
            low, high = _BACKBONE_BONDS[("C", "N_NEXT")]
            if not low <= distance(left["C"], right["N"]) <= high:
                chain_breaks += 1
    if chain_breaks:
        reasons.append(f"chain_breaks:{chain_breaks}")
    return not reasons, chain_breaks, reasons


def _heavy_atom_clashes(atoms: list[dict], *, maximum_overlap_A: float, minimum_distance_A: float) -> tuple[int, float | None]:
    heavy = [atom for atom in atoms if atom["element"].upper() not in {"H", "D"}]
    if len(heavy) < 2:
        return 0, None
    xyz = np.asarray([atom["xyz"] for atom in heavy], dtype=float)
    radii = np.asarray([_VDW_RADII_A.get(atom["element"].upper(), 1.70) for atom in heavy])
    residue = [_residue_key(atom) for atom in heavy]
    try:
        numeric_residue = [int(key[1]) for key in residue]
    except ValueError:
        numeric_residue = [None] * len(residue)
    search_radius = float(2 * radii.max())
    pairs = cKDTree(xyz).query_pairs(search_radius, output_type="ndarray")
    clash_count = 0
    maximum_overlap = 0.0
    for left, right in pairs:
        left, right = int(left), int(right)
        if residue[left] == residue[right]:
            continue
        if (residue[left][0] == residue[right][0] and numeric_residue[left] is not None
                and numeric_residue[right] is not None
                and abs(numeric_residue[left] - numeric_residue[right]) <= 1):
            # Exclude covalent and near-neighbour contacts that need explicit
            # topology/hydrogen handling for a valid all-atom clash decision.
            continue
        distance = float(np.linalg.norm(xyz[left] - xyz[right]))
        overlap = float(radii[left] + radii[right] - distance)
        maximum_overlap = max(maximum_overlap, overlap)
        if distance < minimum_distance_A or overlap >= maximum_overlap_A:
            clash_count += 1
    return clash_count, maximum_overlap


def _aligned_pocket_rmsd(moving_atoms: list[dict], reference_atoms: list[dict], pocket_residues: list[str]) -> tuple[float | None, int]:
    selected = {str(value) for value in pocket_residues}

    def ca_map(atoms: list[dict]) -> dict[tuple[str, str], np.ndarray]:
        result = {}
        for atom in atoms:
            if atom["atom"] != "CA" or atom["element"].upper() != "C":
                continue
            if selected and atom["resid"] not in selected:
                continue
            result[(atom["chain"], atom["resid"])] = np.asarray(atom["xyz"], dtype=float)
        return result

    moving, reference = ca_map(moving_atoms), ca_map(reference_atoms)
    common = sorted(set(moving) & set(reference))
    if len(common) < 3:
        return None, len(common)
    mobile_xyz = np.asarray([moving[key] for key in common])
    reference_xyz = np.asarray([reference[key] for key in common])
    mobile_center = mobile_xyz.mean(axis=0)
    reference_center = reference_xyz.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile_xyz - mobile_center).T @ (reference_xyz - reference_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    aligned = (mobile_xyz - mobile_center) @ rotation + reference_center
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - reference_xyz) ** 2, axis=1))))
    return rmsd, len(common)


def run_step2(
    loaded: LoadedConfig, *, conformer_paths: list[str | Path], output_dir: str | Path,
    minimum_pass_fraction: float, maximum_clash_distance_A: float, maximum_energy_z: float,
    maximum_vdw_overlap_A: float = 0.4, reference_path: str | Path | None = None,
    pocket_residues: list[str] | None = None, maximum_pocket_rmsd_A: float = 2.0,
) -> Step2Result:
    if (not 0 <= minimum_pass_fraction <= 1 or maximum_clash_distance_A <= 0 or maximum_energy_z <= 0
            or maximum_vdw_overlap_A <= 0 or maximum_pocket_rmsd_A <= 0):
        raise ValueError("QC thresholds must be explicit valid positive TCC values")
    out = Path(output_dir)
    prepare_stage_dir(out)
    reference_atoms = read_pdb_atoms(reference_path) if reference_path and Path(reference_path).is_file() else []
    rows, passed_paths = [], []
    for path in map(Path, conformer_paths):
        atoms = read_pdb_atoms(path)
        xyz = np.asarray([atom["xyz"] for atom in atoms], dtype=float)
        finite = bool(len(xyz) and np.isfinite(xyz).all())
        geometry_ok, chain_breaks, geometry_reasons = _backbone_qc(atoms) if finite else (False, 0, ["missing_or_nonfinite_coordinates"])
        clash_count, maximum_overlap = _heavy_atom_clashes(
            atoms, maximum_overlap_A=maximum_vdw_overlap_A, minimum_distance_A=maximum_clash_distance_A,
        ) if finite else (0, None)
        clash_ok = finite and clash_count == 0
        pocket_rmsd, matched_ca = _aligned_pocket_rmsd(atoms, reference_atoms, pocket_residues or []) if reference_atoms else (None, 0)
        rmsd_available = pocket_rmsd is not None
        rmsd_ok = pocket_rmsd <= maximum_pocket_rmsd_A if rmsd_available else True
        # The heavy-atom-only clash scan is a warning screen, not a pass/fail
        # criterion: hydrogens and donor/acceptor typing are required to resolve
        # contacts reliably. It cannot establish physical validity on its own.
        passed = finite and geometry_ok and rmsd_ok
        reasons = list(geometry_reasons)
        warnings = []
        if finite and not clash_ok:
            warnings.append(f"heavy_atom_clash_candidates:{clash_count}")
        if reference_path and not rmsd_available:
            warnings.append("pocket_rmsd_unavailable:no_three_matched_backbone_CA_atoms")
        elif rmsd_available and not rmsd_ok:
            reasons.append("pocket_rmsd_above_threshold")
        rows.append({
            "state_id": path.stem,
            "file_path": path.name,
            "atom_count": len(atoms),
            "finite_coordinates": finite,
            "backbone_bond_geometry_ok": geometry_ok,
            "chain_break_count": chain_breaks,
            "heavy_atom_clash_screen_ok": clash_ok,
            "severe_heavy_atom_clash_count": clash_count,
            "maximum_vdw_overlap_A": maximum_overlap,
            "pocket_rmsd_A": pocket_rmsd,
            "pocket_rmsd_matched_CA_count": matched_ca,
            "energy_evaluation_available": False,
            "restrained_relaxation_available": False,
            "passed_geometry_proxy": passed,
            "failure_reasons": ";".join(reasons),
            "screening_warnings": ";".join(warnings),
        })
        if passed:
            passed_paths.append(path)

    fieldnames = list(rows[0]) if rows else [
        "state_id", "file_path", "atom_count", "finite_coordinates", "backbone_bond_geometry_ok",
        "chain_break_count", "heavy_atom_clash_screen_ok", "severe_heavy_atom_clash_count",
        "maximum_vdw_overlap_A", "pocket_rmsd_A", "pocket_rmsd_matched_CA_count",
        "energy_evaluation_available", "restrained_relaxation_available", "passed_geometry_proxy", "failure_reasons", "screening_warnings",
    ]
    table = out / "qc_table.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    all_rmsd_available = bool(rows) and all(row["pocket_rmsd_A"] is not None for row in rows)
    energy_available = False
    relaxation_available = False
    full_evidence = all_rmsd_available and energy_available and relaxation_available
    gate = evaluate_q(
        passed_count=len(passed_paths), total_count=len(conformer_paths), minimum_fraction=minimum_pass_fraction,
        config_hash=loaded.config_hash, evidence_complete=full_evidence,
    )
    (out / "qc_summary.json").write_text(json.dumps({
        "conformers": len(rows),
        "geometry_proxy_passed": len(passed_paths),
        "minimum_pass_fraction": minimum_pass_fraction,
        "full_qc_evidence_available": full_evidence,
        "pocket_rmsd_available_for_all_states": all_rmsd_available,
        "energy_evaluation_available": energy_available,
        "restrained_relaxation_available": relaxation_available,
        "limitations": [
            "backbone bond checks use broad geometric ranges, not a complete stereochemical validator",
            "backbone angle and dihedral distributions are not evaluated",
            "heavy-atom clash screen is advisory, omits hydrogens, and ignores same-residue and adjacent-residue contacts",
            "energy outliers and restrained relaxation require an installed and configured molecular mechanics engine",
        ],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(
        out,
        config_hash=loaded.config_hash,
        seed=loaded.config.seed,
        counts_in={"conformers": len(conformer_paths)},
        counts_out={"geometry_proxy_passed": len(passed_paths)},
        dropped=[{"id": row["state_id"], "reason": row["failure_reasons"]} for row in rows if not row["passed_geometry_proxy"]],
        extra={
            "full_qc_available": full_evidence,
            "maximum_vdw_overlap_A": maximum_vdw_overlap_A,
            "maximum_pocket_rmsd_A": maximum_pocket_rmsd_A,
            "reference_path": str(Path(reference_path).resolve()) if reference_path else None,
            "energy_outlier_cutoff_z_configured_but_unavailable": maximum_energy_z,
        },
    )
    return Step2Result(gate, passed_paths, table)
