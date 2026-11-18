"""Gate-specific wrappers around the single shared retrospective scorer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..config.loader import require_frozen
from ..scoring.retrospective import holm_primary_pvalue, paired_auc_inference, retrospective_scorer
from .g2b import evaluate_g2b
from .g4 import evaluate_g4


def run_g2b(
    compounds, *, ensemble_states, crystal_states, score_function, config_hash: str,
    min_actives: int, min_decoys: int, minimum_ci_lower_bound: float, n_boot: int, n_permutations: int, confidence: float, seed: int, freeze_record_path: str | Path,
):
    require_frozen(freeze_record_path, config_hash)
    arms = {"ensemble": {"states": ensemble_states}, "crystal": {"states": crystal_states}}
    scored = retrospective_scorer(compounds, arms, constraint="none", score_function=score_function, seed=seed)
    by_arm = {arm: {row["compound_id"]: row["rank_score"] for row in scored.records if row["arm"] == arm} for arm in arms}
    inference = paired_auc_inference(compounds, by_arm["crystal"], by_arm["ensemble"], n_boot=n_boot, n_permutations=n_permutations, seed=seed, confidence=confidence)
    counts = Counter(item["label"] for item in compounds)
    gate = evaluate_g2b(auc_gain=inference["difference"], ci_lower=inference["ci"][0], minimum_ci_lower_bound=minimum_ci_lower_bound, actives=counts[1], decoys=counts[0], minimum_actives=min_actives, minimum_decoys=min_decoys, config_hash=config_hash)
    return gate, scored, inference


def run_g4(
    compounds, *, arms, score_function, config_hash: str, minimum_auc_gain: float,
    alpha: float, min_actives: int, min_decoys: int, family_pvalues: list[float] | None,
    n_boot: int, n_permutations: int, confidence: float, seed: int, freeze_record_path: str | Path, primary_index: int = 0,
):
    require_frozen(freeze_record_path, config_hash)
    required_arms = {"crystal_diffphar", "crystal_pharmflow", "ensemble_diffphar", "ensemble_pharmflow"}
    if set(arms) != required_arms:
        raise ValueError(f"G4 requires the preregistered 2x2 arms: {sorted(required_arms)}")
    scored = retrospective_scorer(compounds, arms, constraint="pharmacophore", score_function=score_function, seed=seed)
    by_arm = {arm: {row["compound_id"]: row["rank_score"] for row in scored.records if row["arm"] == arm} for arm in arms}
    inference = paired_auc_inference(compounds, by_arm["crystal_pharmflow"], by_arm["ensemble_pharmflow"], n_boot=n_boot, n_permutations=n_permutations, seed=seed, confidence=confidence)
    family = family_pvalues if family_pvalues is not None else [inference["p_value"]]
    adjusted, _ = holm_primary_pvalue(primary_index, family, alpha)
    counts = Counter(item["label"] for item in compounds)
    gate = evaluate_g4(auc_gain=inference["difference"], minimum_auc_gain=minimum_auc_gain, adjusted_p_value=adjusted, alpha=alpha, actives=counts[1], decoys=counts[0], minimum_actives=min_actives, minimum_decoys=min_decoys, config_hash=config_hash)
    return gate, scored, inference
