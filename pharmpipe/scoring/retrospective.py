"""One shared retrospective scoring path for G2b and G4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..stats import auc, holm_adjust


@dataclass
class RetrospectiveScores:
    records: list[dict]
    auc_by_arm: dict[str, float]


def retrospective_scorer(
    compounds: list[dict], arms: dict[str, dict], *, constraint: str,
    score_function: Callable[[str, str, str, object | None, int], float], seed: int,
) -> RetrospectiveScores:
    """Score matched molecules against arm states; higher scores rank as more active."""
    if constraint not in {"none", "pharmacophore"}:
        raise ValueError("constraint must be 'none' or 'pharmacophore'")
    records = []
    for arm_index, (arm_name, arm) in enumerate(arms.items()):
        state_ids = arm.get("states", [])
        if not state_ids:
            raise ValueError(f"arm {arm_name!r} has no scoring states")
        model = arm.get("pharmacophore") if constraint == "pharmacophore" else None
        if constraint == "pharmacophore" and model is None:
            raise ValueError(f"arm {arm_name!r} lacks its pharmacophore constraint")
        for compound_index, compound in enumerate(compounds):
            scores = [(state_id, float(score_function(compound["smiles"], state_id, constraint, model, seed + arm_index + compound_index + state_index))) for state_index, state_id in enumerate(state_ids)]
            ordered = sorted(scores, key=lambda item: item[1])
            records.append({"arm": arm_name, "compound_id": compound["compound_id"], "label": int(compound["label"]), "group": compound["group"], "rank_score": -ordered[0][1], "best_state_id": ordered[0][0], "median_score": float(np.median([value for _, value in scores])), "score_range": float(max(value for _, value in scores) - min(value for _, value in scores)), "K": len(state_ids)})
    auc_by_arm = {}
    for arm_name in arms:
        arm_records = [row for row in records if row["arm"] == arm_name]
        auc_by_arm[arm_name] = auc([row["rank_score"] for row in arm_records], [row["label"] for row in arm_records])
    return RetrospectiveScores(records, auc_by_arm)


def paired_auc_inference(
    compounds: list[dict], scores_a: dict[str, float], scores_b: dict[str, float], *,
    n_boot: int, n_permutations: int, seed: int, confidence: float,
) -> dict:
    """Group bootstrap CI and within-group paired label-swap permutation p-value."""
    if n_boot <= 0 or n_permutations <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap and permutation counts must be positive and confidence in (0,1)")
    ids = [item["compound_id"] for item in compounds]
    labels = np.asarray([item["label"] for item in compounds], dtype=int)
    groups = np.asarray([item["group"] for item in compounds], dtype=object)
    a = np.asarray([scores_a[item] for item in ids], dtype=float)
    b = np.asarray([scores_b[item] for item in ids], dtype=float)
    observed = auc(b, labels) - auc(a, labels)
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[group] for group in picked])
        if len(np.unique(labels[rows])) == 2:
            bootstrap.append(auc(b[rows], labels[rows]) - auc(a[rows], labels[rows]))
    if not bootstrap:
        raise ValueError("group bootstrap never retained both labels; use groups containing both classes")
    confidence_low, confidence_high = np.quantile(bootstrap, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    null = np.empty(n_permutations)
    for permutation in range(n_permutations):
        perm_a, perm_b = a.copy(), b.copy()
        for group in unique:
            if rng.integers(0, 2):
                rows = indices[group]
                perm_a[rows], perm_b[rows] = b[rows], a[rows]
        null[permutation] = auc(perm_b, labels) - auc(perm_a, labels)
    p_value = float((np.count_nonzero(null >= observed) + 1) / (n_permutations + 1))
    return {"difference": float(observed), "ci": [float(confidence_low), float(confidence_high)], "p_value": p_value, "bootstrap": bootstrap, "permutation_null": null.tolist()}


def holm_primary_pvalue(primary_index: int, family_pvalues: list[float], alpha: float) -> tuple[float, bool]:
    adjusted, rejected = holm_adjust(family_pvalues, alpha)
    if not 0 <= primary_index < len(family_pvalues):
        raise ValueError("primary test index is outside the preregistered Holm family")
    return float(adjusted[primary_index]), bool(rejected[primary_index])
