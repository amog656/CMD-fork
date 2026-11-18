"""Code-enforced ligand and pharmacophore leakage exclusion rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class LigandRecord:
    ligand_id: str
    smiles: str
    pharmacophore: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExclusionSet:
    excluded_ids: set[str]
    reasons: dict[str, set[str]]


def _mol_and_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES in leakage input: {smiles!r}")
    return mol, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def pharmacophore_similarity(a, b, distance_tolerance: float) -> tuple[float, float]:
    """Return spatial feature-set Tanimoto and one-to-one same-type F1."""
    if distance_tolerance <= 0:
        raise ValueError("distance_tolerance must be positive")
    a, b = list(a), list(b)
    if not a and not b:
        return 1.0, 1.0
    if not a or not b:
        return 0.0, 0.0
    xyz_a = np.asarray([p.xyz for p in a], dtype=float)
    xyz_b = np.asarray([p.xyz for p in b], dtype=float)
    distances = cdist(xyz_a, xyz_b)
    same_type = np.asarray([[x.type == y.type for y in b] for x in a])
    rows, cols = __import__("scipy").optimize.linear_sum_assignment(np.where(same_type & (distances <= distance_tolerance), distances, 1e9))
    matches = int(np.count_nonzero(same_type[rows, cols] & (distances[rows, cols] <= distance_tolerance)))
    tanimoto = matches / (len(a) + len(b) - matches)
    precision, recall = matches / len(a), matches / len(b)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float(tanimoto), float(f1)


def build_exclusion_set(
    held_out: list[LigandRecord], candidates: list[LigandRecord], *,
    min_ecfp_tanimoto: float, min_pharmacophore_tanimoto: float,
    min_pharmacophore_f1: float, pharmacophore_tolerance: float,
) -> ExclusionSet:
    thresholds = (min_ecfp_tanimoto, min_pharmacophore_tanimoto, min_pharmacophore_f1)
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("similarity thresholds must be in [0, 1]")
    held_data = [(record, *_mol_and_fp(record.smiles), MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(record.smiles))) for record in held_out]
    reasons: dict[str, set[str]] = {record.ligand_id: {"held_out_active"} for record in held_out}
    excluded = set(reasons)
    for candidate in candidates:
        for held, held_mol, held_fp, held_scaffold in held_data:
            if candidate.ligand_id == held.ligand_id:
                reasons.setdefault(candidate.ligand_id, set()).add("held_out_active")
                excluded.add(candidate.ligand_id)
                continue
            mol, fp = _mol_and_fp(candidate.smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaffold and scaffold == held_scaffold:
                reasons.setdefault(candidate.ligand_id, set()).add("murcko_scaffold_match")
            if DataStructs.TanimotoSimilarity(fp, held_fp) >= min_ecfp_tanimoto:
                reasons.setdefault(candidate.ligand_id, set()).add("ecfp4_similarity")
            if candidate.pharmacophore and held.pharmacophore:
                tanimoto, f1 = pharmacophore_similarity(candidate.pharmacophore, held.pharmacophore, pharmacophore_tolerance)
                if tanimoto >= min_pharmacophore_tanimoto:
                    reasons.setdefault(candidate.ligand_id, set()).add("pharmacophore_set_similarity")
                if f1 >= min_pharmacophore_f1:
                    reasons.setdefault(candidate.ligand_id, set()).add("pharmacophore_feature_f1")
            if candidate.ligand_id in reasons:
                excluded.add(candidate.ligand_id)
    return ExclusionSet(excluded, reasons)


def assert_no_excluded_seed(seeds, exclusion_set: ExclusionSet) -> None:
    violations = []
    for seed in seeds:
        ligand_id = seed.get("ligand_id") if isinstance(seed, dict) else getattr(seed, "ligand_id", None)
        if ligand_id in exclusion_set.excluded_ids:
            violations.append(ligand_id)
    if violations:
        raise ValueError(f"excluded held-out ligand structures cannot be seeds: {sorted(set(violations))}")


def cluster_leakage_graph(node_ids, pairwise_distances, cutoff: float) -> dict[str, str]:
    """Connected-component clusters; graph components are kept intact in splits."""
    ids = list(node_ids)
    distances = np.asarray(pairwise_distances, dtype=float)
    if distances.shape != (len(ids), len(ids)) or cutoff < 0 or np.any(~np.isfinite(distances)):
        raise ValueError("distance matrix shape or cutoff is invalid")
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        left, right = find(i), find(j)
        if left != right:
            parent[right] = left

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if distances[i, j] <= cutoff or distances[j, i] <= cutoff:
                union(i, j)
    roots: dict[int, str] = {}
    result = {}
    for i, node_id in enumerate(ids):
        root = find(i)
        roots.setdefault(root, f"cluster_{len(roots)}")
        result[node_id] = roots[root]
    return result


def family_heldout_splits(
    node_ids, family_ids, leakage_clusters, *, held_out_families, calibration_families,
    minimum_train_families: int, minimum_held_out_families: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign whole leakage components to family splits; never split a component."""
    ids, families, clusters = list(node_ids), list(family_ids), list(leakage_clusters)
    if not (len(ids) == len(families) == len(clusters)):
        raise ValueError("node, family, and leakage-cluster arrays must have equal lengths")
    held_out, calibration = set(held_out_families), set(calibration_families)
    overlap = held_out & calibration
    if overlap:
        raise ValueError(f"family cannot be both held-out and calibration: {sorted(overlap)}")
    family_for_cluster: dict[str, set[str]] = {}
    for family, cluster in zip(families, clusters):
        family_for_cluster.setdefault(str(cluster), set()).add(str(family))
    split_for_cluster = {}
    for cluster, cluster_families in family_for_cluster.items():
        memberships = {"held_out" if f in held_out else "calibration" if f in calibration else "train" for f in cluster_families}
        if len(memberships) > 1:
            raise ValueError(f"leakage component {cluster!r} crosses requested family splits")
        split_for_cluster[cluster] = next(iter(memberships))
    result = {node: split_for_cluster[str(cluster)] for node, cluster in zip(ids, clusters)}
    counts = {
        "train_families": len({str(f) for f, split in zip(families, result.values()) if split == "train"}),
        "calibration_families": len({str(f) for f, split in zip(families, result.values()) if split == "calibration"}),
        "held_out_families": len({str(f) for f, split in zip(families, result.values()) if split == "held_out"}),
    }
    counts["glfo_evaluable"] = int(counts["train_families"] >= minimum_train_families and counts["held_out_families"] >= minimum_held_out_families)
    return result, counts


def calibration_ood_cutoff(nearest_training_distances, *, quantile: float, calibration_only: bool) -> float:
    """Frozen distance cutoff derived only from family-held-out calibration folds."""
    distances = np.asarray(nearest_training_distances, dtype=float)
    if not calibration_only:
        raise ValueError("OOD cutoff may only be computed from calibration-fold distances")
    if distances.ndim != 1 or not len(distances) or np.any(~np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("nearest-training distances must be a non-empty finite non-negative vector")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0,1)")
    return float(np.quantile(distances, quantile))
