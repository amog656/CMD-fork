"""Configurable correspondence-free, channel-aware pocket Chamfer distance."""

from __future__ import annotations

from itertools import permutations, product

import numpy as np
from scipy.spatial.distance import cdist


def _rotation_starts(points: np.ndarray) -> list[np.ndarray]:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered
    _, basis = np.linalg.eigh(covariance)
    starts = []
    for perm in permutations(range(3)):
        candidate = basis[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            rotation = candidate @ np.diag(signs)
            if np.linalg.det(rotation) > 0:
                starts.append(rotation)
    return starts


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _aligned_chamfer(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[np.ndarray, float]:
    target = coords_a - coords_a.mean(axis=0)
    source_centered = coords_b - coords_b.mean(axis=0)
    best_score = np.inf
    best_aligned = source_centered
    for initial in _rotation_starts(source_centered):
        aligned = source_centered @ initial
        previous = np.inf
        for _ in range(max(len(coords_a), len(coords_b))):
            distances = cdist(target, aligned)
            nearest_a = np.argmin(distances, axis=1)
            nearest_b = np.argmin(distances, axis=0)
            paired_source = np.vstack((aligned[nearest_a], aligned))
            paired_target = np.vstack((target, target[nearest_b]))
            rotation = _kabsch(paired_source, paired_target)
            aligned = (aligned - aligned.mean(axis=0)) @ rotation + target.mean(axis=0)
            distances = cdist(target, aligned)
            score = float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2)
            if abs(previous - score) <= np.finfo(float).eps:
                break
            previous = score
        if score < best_score:
            best_score, best_aligned = score, aligned
    return best_aligned, best_score


def chamfer_pocket_distance(
    pocket_a: dict[str, np.ndarray], pocket_b: dict[str, np.ndarray], *,
    channel_weights: dict[str, float], channel_scales: dict[str, float],
    lambda_mismatch: float,
) -> float:
    """Symmetric channel-weighted point-cloud distance with rigid alignment.

    Each pocket contains ``coordinates`` (N x 3) and aligned per-point arrays for
    the configured channels. Channel scales normalize continuous distances to [0, 1].
    """
    if "coordinates" not in pocket_a or "coordinates" not in pocket_b:
        raise ValueError("each pocket requires coordinates")
    if set(channel_weights) != set(channel_scales) or any(w < 0 for w in channel_weights.values()):
        raise ValueError("channel weights and scales must have matching keys and non-negative weights")
    if lambda_mismatch < 0 or any(scale <= 0 for scale in channel_scales.values()):
        raise ValueError("lambda_mismatch must be non-negative and channel scales positive")
    coords_a = np.asarray(pocket_a["coordinates"], dtype=float)
    coords_b = np.asarray(pocket_b["coordinates"], dtype=float)
    if coords_a.ndim != 2 or coords_b.ndim != 2 or coords_a.shape[1:] != (3,) or coords_b.shape[1:] != (3,) or not len(coords_a) or not len(coords_b):
        raise ValueError("coordinates must be non-empty N x 3 arrays")
    aligned_b, _ = _aligned_chamfer(coords_a, coords_b)
    aligned_a_reverse, _ = _aligned_chamfer(coords_b, coords_a)
    centered_a = coords_a - coords_a.mean(axis=0)
    centered_b = coords_b - coords_b.mean(axis=0)
    distances_ab = cdist(centered_a, aligned_b)
    distances_ba = cdist(centered_b, aligned_a_reverse)
    weights_sum = sum(channel_weights.values())
    if weights_sum <= 0:
        raise ValueError("at least one channel weight must be positive")
    channel_sum = 0.0
    for channel, weight in channel_weights.items():
        values_a = np.asarray(pocket_a[channel])
        values_b = np.asarray(pocket_b[channel])
        if len(values_a) != len(coords_a) or len(values_b) != len(coords_b):
            raise ValueError(f"channel {channel!r} must have one value per point")
        if values_a.ndim == 1:
            mismatch = np.not_equal(values_a[:, None], values_b[None, :]).astype(float)
        else:
            mismatch = np.linalg.norm(values_a[:, None, :] - values_b[None, :, :], axis=-1) / channel_scales[channel]
        spatial_ab = distances_ab / channel_scales[channel]
        spatial_ba = distances_ba / channel_scales[channel]
        spatial_ab[spatial_ab <= np.finfo(float).eps * 16] = 0
        spatial_ba[spatial_ba <= np.finfo(float).eps * 16] = 0
        costs_ab = np.minimum(1.0, spatial_ab + mismatch)
        costs_ba = np.minimum(1.0, spatial_ba + mismatch.T)
        normalized = min(1.0, float((costs_ab.min(axis=1).mean() + costs_ba.min(axis=1).mean()) / 2))
        channel_sum += weight * normalized
    mismatch_fraction = abs(len(coords_a) - len(coords_b)) / max(len(coords_a), len(coords_b))
    distance = (channel_sum / weights_sum + lambda_mismatch * min(1.0, mismatch_fraction)) / (1 + lambda_mismatch)
    return float(distance)
