"""Shared evaluation statistics with explicit randomness and thresholds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    confidence_interval: tuple[float, float]
    bootstrap_values: tuple[float, ...]


@dataclass(frozen=True)
class PermutationResult:
    statistic: float
    p_value: float
    null_statistics: tuple[float, ...]


@dataclass(frozen=True)
class F1Result:
    macro: float
    micro: float
    per_sample: tuple[float, ...]


@dataclass(frozen=True)
class BestOfKNullResult:
    observed_maximum_mean: float
    adjusted_p_value: float
    null_distribution: tuple[float, ...]


def grouped_paired_bootstrap(
    values_a: Sequence[float], values_b: Sequence[float], groups: Sequence[Any],
    n_boot: int, seed: int, confidence: float,
) -> BootstrapResult:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    group_ids = np.asarray(groups, dtype=object)
    if a.shape != b.shape or a.ndim != 1 or len(group_ids) != len(a):
        raise ValueError("paired values and groups must be one-dimensional and equal length")
    if not len(a) or n_boot <= 0 or not 0 < confidence < 1:
        raise ValueError("non-empty samples, positive n_boot, and confidence in (0, 1) are required")
    unique_groups = np.unique(group_ids)
    if not len(unique_groups):
        raise ValueError("at least one independent group is required")
    per_group = {g: np.flatnonzero(group_ids == g) for g in unique_groups}
    differences = b - a
    estimate = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    sampled = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        row_ids = np.concatenate([per_group[g] for g in picked])
        sampled[index] = float(np.mean(differences[row_ids]))
    tail = (1 - confidence) / 2
    low, high = np.quantile(sampled, [tail, 1 - tail])
    return BootstrapResult(estimate, (float(low), float(high)), tuple(map(float, sampled)))


def permutation_test(
    observed: Any,
    statistic: Callable[[Any], float],
    permutation_scheme: Callable[[Any, np.random.Generator], Any],
    n_permutations: int,
    seed: int,
    alternative: str = "greater",
) -> PermutationResult:
    if n_permutations <= 0 or alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("n_permutations must be positive and alternative must be supported")
    observed_value = float(statistic(observed))
    rng = np.random.default_rng(seed)
    null = np.asarray([statistic(permutation_scheme(observed, rng)) for _ in range(n_permutations)], dtype=float)
    if alternative == "greater":
        extreme = np.count_nonzero(null >= observed_value)
    elif alternative == "less":
        extreme = np.count_nonzero(null <= observed_value)
    else:
        extreme = np.count_nonzero(np.abs(null) >= abs(observed_value))
    p_value = float((extreme + 1) / (n_permutations + 1))
    return PermutationResult(observed_value, p_value, tuple(map(float, null)))


def holm_adjust(pvalues: Sequence[float], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a one-dimensional vector in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    m = len(p)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p) if m else sorted_p
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_sorted, 1)
    rejected = adjusted <= alpha
    return adjusted, rejected


def holm_inverted_ci(
    estimates: Sequence[float], standard_errors: Sequence[float], pvalues: Sequence[float], alpha: float,
) -> tuple[tuple[float, float], ...]:
    """Conservative Holm-inverted normal intervals, ordered by raw p-value."""
    estimates = np.asarray(estimates, dtype=float)
    errors = np.asarray(standard_errors, dtype=float)
    p = np.asarray(pvalues, dtype=float)
    if estimates.shape != errors.shape or p.shape != estimates.shape or np.any(errors < 0):
        raise ValueError("estimates, standard errors, and p-values must have equal shape")
    order = np.argsort(p, kind="stable")
    bounds: list[tuple[float, float] | None] = [None] * len(p)
    for rank, idx in enumerate(order):
        local_alpha = alpha / (len(p) - rank)
        z = _normal_quantile(1 - local_alpha / 2)
        bounds[idx] = (float(estimates[idx] - z * errors[idx]), float(estimates[idx] + z * errors[idx]))
    return tuple(x for x in bounds if x is not None)


def _normal_quantile(probability: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(probability))


def wilson_interval(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials or not 0 < confidence < 1:
        raise ValueError("require 0 <= successes <= trials, trials > 0, and confidence in (0, 1)")
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + confidence / 2))
    phat = successes / trials
    denominator = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denominator
    return float(max(0, center - radius)), float(min(1, center + radius))


def hungarian_f1(generated, reference, tol: float, same_type_only: bool = True) -> F1Result:
    """One-to-one feature F1, macro-primary over pockets and micro-secondary."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    if generated and isinstance(generated[0], (list, tuple)):
        if len(generated) != len(reference):
            raise ValueError("generated and reference pocket collections must have equal length")
        sample_results = [hungarian_f1(g, r, tol, same_type_only) for g, r in zip(generated, reference)]
        macro = float(np.mean([result.macro for result in sample_results])) if sample_results else 1.0
        total_gen = sum(len(g) for g in generated)
        total_ref = sum(len(r) for r in reference)
        total_matches = sum(round(result.micro * (len(g) + len(r)) / (2 if result.micro else 1)) for result, g, r in zip(sample_results, generated, reference))
        micro_precision = total_matches / total_gen if total_gen else (1.0 if total_ref == 0 else 0.0)
        micro_recall = total_matches / total_ref if total_ref else (1.0 if total_gen == 0 else 0.0)
        micro = 0.0 if micro_precision + micro_recall == 0 else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        return F1Result(macro, float(micro), tuple(result.macro for result in sample_results))
    generated = list(generated)
    reference = list(reference)
    if not generated and not reference:
        return F1Result(1.0, 1.0, (1.0,))
    if not generated or not reference:
        return F1Result(0.0, 0.0, (0.0,))
    gx = np.asarray([p.xyz for p in generated], dtype=float)
    rx = np.asarray([p.xyz for p in reference], dtype=float)
    distances = cdist(gx, rx)
    same = np.asarray([[g.type == r.type for r in reference] for g in generated])
    allowed = same if same_type_only else np.ones_like(same, dtype=bool)
    cost = np.where(allowed & (distances <= tol), distances, np.finfo(float).max / 100)
    rows, cols = linear_sum_assignment(cost)
    matches = int(np.count_nonzero(allowed[rows, cols] & (distances[rows, cols] <= tol)))
    precision = matches / len(generated)
    recall = matches / len(reference)
    score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return F1Result(float(score), float(score), (float(score),))


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape or len(np.unique(labels)) != 2:
        raise ValueError("AUC requires equal-length scores and binary labels containing both classes")
    return float(roc_auc_score(labels, scores))


def enrichment_factor(scores: Sequence[float], labels: Sequence[int], fraction: float) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape or not len(scores) or not 0 < fraction <= 1:
        raise ValueError("scores and labels must match, be non-empty, and fraction must be in (0, 1]")
    positives = np.count_nonzero(labels == 1)
    if positives == 0 or np.any((labels != 0) & (labels != 1)):
        raise ValueError("labels must be binary with at least one positive")
    n_top = max(1, int(np.ceil(len(scores) * fraction)))
    top = np.argsort(scores)[-n_top:]
    return float((np.count_nonzero(labels[top] == 1) / n_top) / (positives / len(scores)))


def best_of_k_null(score_matrix, K: int, n_perm: int, seed: int) -> BestOfKNullResult:
    """Selection-adjusted max-score null by shuffling compounds within each state slot."""
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or K != scores.shape[1] or n_perm <= 0 or np.any(~np.isfinite(scores)):
        raise ValueError("score matrix must be finite 2D, K must match its columns, and n_perm positive")
    rng = np.random.default_rng(seed)
    maxima = np.empty(n_perm, dtype=float)
    for index in range(n_perm):
        permuted = np.column_stack([rng.permutation(scores[:, slot]) for slot in range(K)])
        maxima[index] = np.max(permuted, axis=1).mean()
    observed = float(np.max(scores, axis=1).mean())
    p_value = float((np.count_nonzero(maxima >= observed) + 1) / (n_perm + 1))
    return BestOfKNullResult(observed, p_value, tuple(map(float, maxima)))


def weights_converged(
    replica_estimates, *, min_replicas: int, max_split_half_difference: float,
    min_rank_correlation: float, max_ci_width: float, seed: int, n_boot: int, confidence: float,
) -> tuple[bool, dict[str, float | int | None]]:
    """Apply shared C5 convergence checks to replica-by-state population estimates."""
    values = np.asarray(replica_estimates, dtype=float)
    if values.ndim != 2 or values.shape[0] < min_replicas or values.shape[1] == 0:
        return False, {"replicas": int(values.shape[0]) if values.ndim == 2 else 0}
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        return False, {"replicas": int(values.shape[0]), "reason": "invalid estimates"}
    first, second = values[::2], values[1::2]
    if not len(second):
        return False, {"replicas": len(values), "reason": "insufficient split halves"}
    split_difference = np.max(np.abs(first.mean(axis=0) - second.mean(axis=0)))
    from scipy.stats import spearmanr
    correlations = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            corr = spearmanr(values[i], values[j]).statistic
            if np.isfinite(corr):
                correlations.append(float(corr))
    rank_corr = min(correlations) if correlations else None
    rng = np.random.default_rng(seed)
    if n_boot <= 0 or not 0 < confidence < 1:
        raise ValueError("n_boot must be positive and confidence in (0, 1)")
    boot = np.asarray([values[rng.integers(0, len(values), len(values))].mean(axis=0) for _ in range(n_boot)])
    ci_low, ci_high = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    ci_width = float(np.max(np.quantile(boot, ci_high, axis=0) - np.quantile(boot, ci_low, axis=0)))
    passed = split_difference <= max_split_half_difference and rank_corr is not None and rank_corr >= min_rank_correlation and ci_width <= max_ci_width
    return bool(passed), {"replicas": len(values), "split_half_difference": float(split_difference), "minimum_rank_correlation": rank_corr, "maximum_ci_width": ci_width}
