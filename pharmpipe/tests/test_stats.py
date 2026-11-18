import numpy as np
import pytest

from pharmpipe.data.models import PharmModel, PharmPoint
from pharmpipe.leakage import LigandRecord, assert_no_excluded_seed, build_exclusion_set, calibration_ood_cutoff, family_heldout_splits, pharmacophore_similarity
from pharmpipe.stats import auc, best_of_k_null, enrichment_factor, grouped_paired_bootstrap, holm_adjust, holm_inverted_ci, hungarian_f1, permutation_test, weights_converged, wilson_interval
from pharmpipe.stats.pocket_distance import chamfer_pocket_distance
from pharmpipe.scoring.retrospective import retrospective_scorer
from pharmpipe.gates.fep import evaluate_fep
from pharmpipe.constants import RT_KCAL_PER_MOL_AT_REFERENCE_TEMPERATURE
import math
from pharmpipe.steps.pharmacophore_graph import pharmacophore_graph
from scipy.stats import norm


def test_holm_known_answer():
    adjusted, rejected = holm_adjust([0.01, 0.02, 0.04], alpha=0.05)
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])
    assert rejected.all()


def test_wilson_zero_of_five_is_computed_and_bounded():
    for successes in (0, 1):
        low, high = wilson_interval(successes, 5, confidence=0.95)
        z = norm.ppf(1 - (1 - 0.95) / 2)
        p = successes / 5
        denominator = 1 + z**2 / 5
        center = (p + z**2 / (2 * 5)) / denominator
        radius = z * np.sqrt(p * (1 - p) / 5 + z**2 / (4 * 5**2)) / denominator
        assert np.allclose([low, high], [max(0, center - radius), min(1, center + radius)])


def test_grouped_bootstrap_is_seeded_and_paired():
    result_a = grouped_paired_bootstrap([1, 2, 3, 4], [2, 3, 5, 6], ["x", "x", "y", "y"], 100, 7, 0.95)
    result_b = grouped_paired_bootstrap([1, 2, 3, 4], [2, 3, 5, 6], ["x", "x", "y", "y"], 100, 7, 0.95)
    assert result_a == result_b
    assert result_a.estimate == 1.5


def test_hungarian_f1_identical_and_type_mismatch():
    point = PharmPoint((0.0, 0.0, 0.0), "HDON")
    other_type = PharmPoint((0.0, 0.0, 0.0), "HACC")
    assert hungarian_f1([point], [point], 1.0).macro == 1.0
    assert hungarian_f1([point], [other_type], 1.0).micro == 0.0


def test_hungarian_f1_macro_and_micro_over_pockets():
    donor = PharmPoint((0.0, 0.0, 0.0), "HDON")
    empty = []
    result = hungarian_f1([[donor], empty], [[donor], [donor]], 1.0)
    assert result.macro == 0.5
    assert result.micro == 2 / 3


def test_pharmacophore_similarity_and_leakage_seed_assertion():
    feature = PharmPoint((1.0, 2.0, 3.0), "HDON")
    assert pharmacophore_similarity([feature], [feature], 1.0) == (1.0, 1.0)
    held = LigandRecord("held", "CCO", (feature,))
    analog = LigandRecord("analog", "CCCO", (feature,))
    excluded = build_exclusion_set([held], [analog], min_ecfp_tanimoto=1.0, min_pharmacophore_tanimoto=0.8, min_pharmacophore_f1=0.8, pharmacophore_tolerance=1.0)
    assert "analog" in excluded.excluded_ids
    try:
        assert_no_excluded_seed([{"ligand_id": "analog"}], excluded)
    except ValueError:
        pass
    else:
        raise AssertionError("excluded seed was accepted")


def test_family_split_keeps_leakage_components_whole_and_cutoff_is_calibration_only():
    split, counts = family_heldout_splits(["a", "b", "c", "d"], ["f1", "f2", "f3", "f4"], ["c1", "c2", "c3", "c4"], held_out_families={"f4"}, calibration_families={"f3"}, minimum_train_families=2, minimum_held_out_families=1)
    assert split == {"a": "train", "b": "train", "c": "calibration", "d": "held_out"}
    assert counts["glfo_evaluable"] == 1
    with pytest.raises(ValueError, match="crosses"):
        family_heldout_splits(["a", "b"], ["f1", "f2"], ["same", "same"], held_out_families={"f2"}, calibration_families=set(), minimum_train_families=1, minimum_held_out_families=1)
    expected = np.quantile([0.1, 0.2, 0.4, 0.8], 0.95)
    assert calibration_ood_cutoff([0.1, 0.2, 0.4, 0.8], quantile=0.95, calibration_only=True) == expected
    with pytest.raises(ValueError, match="calibration"):
        calibration_ood_cutoff([0.1, 0.2], quantile=0.95, calibration_only=False)


def test_pocket_distance_self_and_rigid_transform_invariant():
    coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    base = {"coordinates": coords, "element": np.asarray(["C", "N", "O", "C"])}
    theta = 0.73
    rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = {"coordinates": coords @ rotation + np.asarray([5.0, -2.0, 4.0]), "element": base["element"]}
    kwargs = {"channel_weights": {"element": 1.0}, "channel_scales": {"element": 1.0}, "lambda_mismatch": 0.1}
    assert chamfer_pocket_distance(base, base, **kwargs) == 0
    assert chamfer_pocket_distance(base, moved, **kwargs) < 1e-6
    altered = {"coordinates": coords + np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]]), "element": base["element"]}
    assert np.isclose(chamfer_pocket_distance(base, altered, **kwargs), chamfer_pocket_distance(altered, base, **kwargs))


def test_group_bootstrap_preserves_clustered_uncertainty():
    differences = np.asarray([0] * 50 + [10] * 50)
    groups = ["low"] * 50 + ["high"] * 50
    grouped = grouped_paired_bootstrap(np.zeros(len(differences)), differences, groups, 1000, 10, 0.95)
    rng = np.random.default_rng(10)
    item_samples = np.asarray([rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(1000)])
    item_width = np.quantile(item_samples, 0.975) - np.quantile(item_samples, 0.025)
    grouped_width = grouped.confidence_interval[1] - grouped.confidence_interval[0]
    assert grouped_width > item_width


def test_four_point_pharmacophore_distances_are_computed_and_invariant():
    points = [
        PharmPoint((1, 2, 3), "HDON"), PharmPoint((4, 2, 3), "HACC"),
        PharmPoint((5, 5, 2), "AROM"), PharmPoint((2, 6, 4), "HYBL"),
    ]
    model = PharmModel("four", points, "per-state")
    graph = pharmacophore_graph(model, edge_distance_mode="euclidean_angstrom")
    coords = np.asarray([point.xyz for point in points], dtype=float)
    expected = np.asarray([[np.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))) for right in coords] for left in coords])
    distances = np.asarray(graph["edge_distances_A"])
    assert np.allclose(distances, expected, atol=0.001)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0)
    angle = 0.63
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    moved = [PharmPoint(tuple(point.xyz @ rotation + np.asarray([2, -1, 4])), point.type) for point in points]
    moved_graph = pharmacophore_graph(PharmModel("moved", moved, "per-state"), edge_distance_mode="euclidean_angstrom")
    assert np.allclose(distances, moved_graph["edge_distances_A"], atol=0.001)
    permutation = [2, 0, 3, 1]
    shuffled = pharmacophore_graph(PharmModel("shuffle", [points[i] for i in permutation], "per-state"), edge_distance_mode="euclidean_angstrom")
    assert np.allclose(np.asarray(shuffled["edge_distances_A"]), distances[np.ix_(permutation, permutation)])


def test_remaining_statistics_reproducibility_and_bounds():
    assert auc([0.9, 0.1], [1, 0]) == 1.0
    assert enrichment_factor([0.9, 0.1], [1, 0], 0.5) == 2.0
    null_a = best_of_k_null([[1, 2], [3, 4]], 2, 25, 19)
    null_b = best_of_k_null([[1, 2], [3, 4]], 2, 25, 19)
    assert null_a == null_b
    assert 0 < null_a.adjusted_p_value <= 1
    passed, diagnostics = weights_converged(np.tile([0.4, 0.6], (4, 1)), min_replicas=3, max_split_half_difference=0.1, min_rank_correlation=0.8, max_ci_width=0.2, seed=3, n_boot=100, confidence=0.95)
    assert passed and diagnostics["replicas"] == 4
    intervals = holm_inverted_ci([0.1, 0.2], [0.01, 0.02], [0.01, 0.04], 0.05)
    assert len(intervals) == 2 and intervals[0][0] < 0.1 < intervals[0][1]


def test_pluggable_permutation_test_uses_seed():
    def statistic(data):
        return float(np.mean(data))

    def permute(data, rng):
        return rng.permutation(data)

    first = permutation_test(np.asarray([1.0, 2.0, 3.0]), statistic, permute, 50, 7)
    second = permutation_test(np.asarray([1.0, 2.0, 3.0]), statistic, permute, 50, 7)
    assert first == second


def test_fep_gate_treats_crossing_intervals_as_not_evaluable():
    threshold = -RT_KCAL_PER_MOL_AT_REFERENCE_TEMPERATURE * math.log(10)
    crossing = evaluate_fep(delta_delta_g=threshold, confidence_interval=(threshold - 0.2, threshold + 0.2), selectivity_goal_fold=10, replica_count=3, minimum_replicas=3, converged=True, config_hash="hash")
    passing = evaluate_fep(delta_delta_g=threshold - 0.2, confidence_interval=(threshold - 0.4, threshold - 0.1), selectivity_goal_fold=10, replica_count=3, minimum_replicas=3, converged=True, config_hash="hash")
    assert crossing.outcome.value == "NOT_EVALUABLE"
    assert passing.outcome.value == "PASS"


def test_shared_retrospective_scorer_supports_both_constraint_modes():
    compounds = [{"compound_id": "a", "smiles": "active", "label": 1, "group": "g"}, {"compound_id": "d", "smiles": "decoy", "label": 0, "group": "g"}]

    def score(smiles, state, constraint, model, seed):
        return -1.0 if constraint == "pharmacophore" and state == "ensemble" and smiles == "active" else 1.0

    plain = retrospective_scorer(compounds, {"ensemble": {"states": ["ensemble"]}, "crystal": {"states": ["crystal"]}}, constraint="none", score_function=score, seed=1)
    constrained = retrospective_scorer(compounds, {"ensemble": {"states": ["ensemble"], "pharmacophore": object()}}, constraint="pharmacophore", score_function=score, seed=1)
    assert len(plain.records) == 4
    assert constrained.auc_by_arm["ensemble"] == 1.0
