"""Step 5: typed consensus and differential pharmacophore hypotheses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from ..config.loader import LoadedConfig
from ..data.models import ConsequenceType, GateOutcome, GateResult, PharmModel, PharmPoint
from ..gates.d_state import evaluate_d_state
from ..gates.g3 import evaluate_g3
from ..runio import prepare_stage_dir, write_manifest


@dataclass
class Step5Result:
    consensus: PharmModel
    differential: PharmModel | None
    g3: GateResult | None
    d_state: GateResult | None
    output_path: Path


def _connected_groups(points: list[tuple[str, PharmPoint]], distance: float) -> list[list[tuple[str, PharmPoint]]]:
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (state_i, point_i) in enumerate(points):
        for j in range(i + 1, len(points)):
            state_j, point_j = points[j]
            if point_i.type == point_j.type and np.linalg.norm(np.asarray(point_i.xyz) - point_j.xyz) <= distance:
                left, right = find(i), find(j)
                parent[right] = left
    groups: dict[int, list[tuple[str, PharmPoint]]] = {}
    for index, point in enumerate(points):
        groups.setdefault(find(index), []).append(point)
    return list(groups.values())


def merge_consensus(models: list[PharmModel], *, total_state_count: int, group_distance_A: float, vote_fraction: float) -> PharmModel:
    if total_state_count <= 0 or group_distance_A <= 0 or not 0 <= vote_fraction <= 1:
        raise ValueError("state count and distance must be positive; vote fraction must be in [0,1]")
    flattened = [(model.supporting_state_ids[0] if model.supporting_state_ids else model.model_id, point) for model in models for point in model.points]
    groups = _connected_groups(flattened, group_distance_A)
    kept_points, kept_votes, supporting = [], [], set()
    for group in groups:
        state_ids = sorted({state_id for state_id, _ in group})
        votes = len(state_ids)
        if votes / total_state_count >= vote_fraction:
            coords = np.mean([point.xyz for _, point in group], axis=0)
            kept_points.append(PharmPoint(tuple(map(float, coords)), group[0][1].type))
            kept_votes.append(votes)
            supporting.update(state_ids)
    return PharmModel("consensus", kept_points, "consensus", sorted(supporting), kept_votes)


def build_differential(
    consensus: PharmModel, target_models: list[PharmModel], off_target_models: dict[str, list[PharmModel]], *,
    presence_tolerance_A: float, minimum_selectivity_score: float, n_boot: int, confidence: float, seed: int,
    minimum_ci_lower_bound: float,
) -> tuple[PharmModel, list[dict]]:
    if not off_target_models:
        raise ValueError("differential model is undefined when no off-targets exist")
    rng = np.random.default_rng(seed)
    retained, metadata = [], []
    for index, candidate in enumerate(consensus.points):
        target_presence = np.asarray([float(any(any(other.type == candidate.type and np.linalg.norm(np.asarray(other.xyz) - candidate.xyz) <= presence_tolerance_A for other in model.points) for model in [state_model])) for state_model in target_models])
        if not len(target_presence):
            continue
        off_presence = []
        for protein_models in off_target_models.values():
            presence = [float(any(other.type == candidate.type and np.linalg.norm(np.asarray(other.xyz) - candidate.xyz) <= presence_tolerance_A for other in model.points)) for model in protein_models]
            off_presence.append(np.asarray(presence, dtype=float))
        point_score = float(target_presence.mean() - max(values.mean() for values in off_presence if len(values)))
        bootstrap_scores = []
        if n_boot <= 0 or not 0 < confidence < 1:
            raise ValueError("n_boot must be positive and confidence in (0,1)")
        for _ in range(n_boot):
            target_sample = rng.choice(target_presence, size=len(target_presence), replace=True).mean()
            off_sample = max(rng.choice(values, size=len(values), replace=True).mean() for values in off_presence if len(values))
            bootstrap_scores.append(float(target_sample - off_sample))
        lower = float(np.quantile(bootstrap_scores, (1 - confidence) / 2))
        if point_score >= minimum_selectivity_score and lower > minimum_ci_lower_bound:
            retained.append(candidate)
        metadata.append({"point_index": index, "score": point_score, "bootstrap_lower_bound": lower})
    model = PharmModel("differential", retained, "differential", consensus.supporting_state_ids, [1] * len(retained))
    return model, metadata


def run_step5(
    loaded: LoadedConfig, *, target_models: list[PharmModel], total_state_count: int, output_dir: str | Path,
    group_distance_A: float, vote_fraction: float, off_target_models: dict[str, list[PharmModel]] | None = None,
    presence_tolerance_A: float | None = None, minimum_selectivity_score: float | None = None,
    n_boot: int | None = None, confidence: float | None = None,
    g3_minimum_auc: float | None = None, g3_maximum_p: float | None = None,
    minimum_differential_ci_lower_bound: float | None = None,
) -> Step5Result:
    out = Path(output_dir)
    prepare_stage_dir(out)
    consensus = merge_consensus(target_models, total_state_count=total_state_count, group_distance_A=group_distance_A, vote_fraction=vote_fraction)
    differential, differential_stats = None, []
    g3 = None
    if loaded.config.off_targets:
        if off_target_models and presence_tolerance_A is not None and minimum_selectivity_score is not None and n_boot is not None and confidence is not None and minimum_differential_ci_lower_bound is not None:
            differential, differential_stats = build_differential(consensus, target_models, off_target_models, presence_tolerance_A=presence_tolerance_A, minimum_selectivity_score=minimum_selectivity_score, n_boot=n_boot, confidence=confidence, seed=loaded.config.seed, minimum_ci_lower_bound=minimum_differential_ci_lower_bound)
        if g3_minimum_auc is None or g3_maximum_p is None:
            raise ValueError("mode with off-targets requires configured G3 thresholds")
        g3 = evaluate_g3(auc_value=None, permutation_p=None, labelled_units=0, minimum_units=loaded.config.n_minima.G3_labelled_units, minimum_auc=g3_minimum_auc, maximum_p=g3_maximum_p, config_hash=loaded.config_hash)
    d_state = evaluate_d_state(potency_auc_drop=None, orthogonal_evidence=False, config_hash=loaded.config_hash) if loaded.config.mode == "D" else None
    output = {"consensus": {"model_id": consensus.model_id, "points": [asdict(x) for x in consensus.points], "vote_counts": consensus.vote_counts, "supporting_state_ids": consensus.supporting_state_ids}, "differential": None if differential is None else {"points": [asdict(x) for x in differential.points], "statistics": differential_stats}}
    path = out / "models.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={"state_models": len(target_models)}, counts_out={"consensus_points": len(consensus.points), "differential_points": len(differential.points) if differential else 0}, dropped=[])
    return Step5Result(consensus, differential, g3, d_state, path)
