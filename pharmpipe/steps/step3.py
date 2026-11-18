"""Step 3: aligned pocket clustering, reference-state coverage, and domain routing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..adapters.pocket_detector import MockPocketDetectorAdapter
from ..config.loader import LoadedConfig
from ..data.models import ConsequenceType, GateOutcome, GateResult, PocketAtom, PocketState
from ..gates.g1 import evaluate_g1
from ..runio import prepare_stage_dir, write_manifest
from ..stats.pocket_distance import chamfer_pocket_distance
from .structure_io import read_pdb_atoms
from .structure_io import write_pdb_atoms


@dataclass
class Step3Result:
    states: list[PocketState]
    g1: GateResult
    g2b: GateResult
    states_path: Path


def _align(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != moving.shape:
        raise ValueError("pocket alignment requires matching stable atom selections")
    ref_center, mov_center = reference.mean(axis=0), moving.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - mov_center).T @ (reference - ref_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = ref_center - mov_center @ rotation
    return (moving - mov_center) @ rotation + ref_center, transform


def run_step3(
    loaded: LoadedConfig, *, surviving_paths: list[str | Path], output_dir: str | Path,
    pocket_residues: list[str], source_types: dict[str, str], cluster_distance_A: float, k_states: int,
    minimum_reference_coverage: float, reference_coverage_tolerance_A: float,
    pocket_distance_weights: dict[str, float], pocket_distance_scales: dict[str, float],
    pocket_distance_lambda: float, reference_state_coordinates: list[np.ndarray] | None = None,
    training_pockets: list[dict[str, np.ndarray]] | None = None, ood_cutoff: float | None = None,
    pocket_detector: MockPocketDetectorAdapter | None = None,
) -> Step3Result:
    if cluster_distance_A <= 0 or k_states <= 0:
        raise ValueError("cluster distance and K must be explicit positive config values")
    out = Path(output_dir)
    prepare_stage_dir(out)
    (out / "pockets").mkdir()
    raw_states = []
    detector = pocket_detector or MockPocketDetectorAdapter()
    for path in map(Path, surviving_paths):
        atoms = read_pdb_atoms(path)
        if loaded.config.target.pocket_source == "detector":
            candidates = [PocketAtom(tuple(map(float, atom["xyz"])), atom["element"], atom["resname"], atom["resid"], atom["atom"]) for atom in atoms]
            detected = detector.detect(candidates, loaded.config.seed + len(raw_states))
            selected_keys = {(atom.element, atom.residue_name, atom.residue_id, atom.atom_name, atom.xyz) for atom in detected}
            selected = [atom for atom in atoms if (atom["element"], atom["resname"], atom["resid"], atom["atom"], tuple(map(float, atom["xyz"]))) in selected_keys]
        else:
            selected = [a for a in atoms if str(a["resid"]) in {str(x) for x in pocket_residues}]
        if not selected:
            raise ValueError(f"no pocket atoms were found for {path}")
        raw_states.append((path, selected, np.asarray([a["xyz"] for a in selected], dtype=float)))
    aligned, transforms = [], []
    if raw_states:
        reference = raw_states[0][2]
        for _, _, coords in raw_states:
            moved, transform = _align(reference, coords)
            aligned.append(moved)
            transforms.append(transform)
    pairwise = np.zeros((len(aligned), len(aligned)), dtype=float)
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            aligned_j, _ = _align(aligned[i], aligned[j])
            value = float(np.sqrt(np.mean(np.sum((aligned[i] - aligned_j) ** 2, axis=1))))
            pairwise[i, j] = pairwise[j, i] = value
    if len(aligned) > 1:
        clusters = fcluster(linkage(squareform(pairwise, checks=False), method="average"), t=cluster_distance_A, criterion="distance")
    else:
        clusters = np.ones(len(aligned), dtype=int)
    cluster_members: dict[int, list[int]] = {}
    for i, cluster_id in enumerate(clusters):
        cluster_members.setdefault(int(cluster_id), []).append(i)
    chosen = sorted(cluster_members, key=lambda c: (-len(cluster_members[c]), c))[:k_states]
    states = []
    for cluster_id in chosen:
        members = cluster_members[cluster_id]
        medoid_idx = min(members, key=lambda idx: float(np.sum(pairwise[idx, members])))
        path, atom_rows, coords = raw_states[medoid_idx]
        atoms = [PocketAtom(tuple(map(float, aligned[medoid_idx][idx])), row["element"], row["resname"], row["resid"], row["atom"]) for idx, row in enumerate(atom_rows)]
        source = source_types.get(path.stem)
        if source not in {"apo-seeded", "holo-seeded", "ligand-bound", "crystal"}:
            raise ValueError(f"source type for state {path.stem!r} must be provided from its ensemble provenance")
        aligned_path = out / "pockets" / f"{path.stem}.pdb"
        write_pdb_atoms(aligned_path, [dict(row, xyz=aligned[medoid_idx][i].tolist()) for i, row in enumerate(atom_rows)])
        state = PocketState(path.stem, source, aligned_path, atoms, cluster_id=str(cluster_id), weight=None, in_domain=False, provenance={"member_state_ids": [raw_states[i][0].stem for i in members], "alignment_transform": transforms[medoid_idx].tolist()}, alignment_transform=transforms[medoid_idx].tolist())
        if training_pockets and ood_cutoff is not None:
            candidate = {"coordinates": aligned[medoid_idx], "element": np.asarray([row["element"] for row in atom_rows])}
            distances = [chamfer_pocket_distance(candidate, pocket, channel_weights=pocket_distance_weights, channel_scales=pocket_distance_scales, lambda_mismatch=pocket_distance_lambda) for pocket in training_pockets]
            state.in_domain = min(distances) <= ood_cutoff
        states.append(state)
    refs = reference_state_coordinates or []
    covered = 0
    if refs:
        for ref in refs:
            ref_coords = np.asarray(ref, dtype=float)
            if any(state.pocket_atoms and ref_coords.shape == (len(state.pocket_atoms), 3) and np.sqrt(np.mean(np.sum((np.asarray([a.xyz for a in state.pocket_atoms]) - ref_coords) ** 2, axis=1))) <= reference_coverage_tolerance_A for state in states):
                covered += 1
    g1 = evaluate_g1(reference_state_count=len(refs), covered_state_count=covered, minimum_coverage=minimum_reference_coverage, config_hash=loaded.config_hash)
    g2b = GateResult("G2b", GateOutcome.NOT_EVALUABLE, ConsequenceType.FALLBACK, config_hash=loaded.config_hash, reason="retrospective scorer inputs are not available at Step 3")
    state_records = [{
        "state_id": state.state_id, "source": state.source, "coordinate_path": str(Path("pockets") / state.coordinate_path.name),
        "cluster_id": state.cluster_id, "weight": state.weight, "in_domain": state.in_domain,
        "provenance": state.provenance,
        "pocket_atoms": [asdict(atom) for atom in state.pocket_atoms],
    } for state in states]
    state_path = out / "states.json"
    state_path.write_text(json.dumps(state_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={"surviving_conformers": len(surviving_paths)}, counts_out={"pocket_states": len(states)}, dropped=[])
    return Step3Result(states, g1, g2b, state_path)
