"""Gate-level forced demonstrations for the seven preregistered toy scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config.loader import freeze, load_tcc
from .data.models import ConsequenceType, GateOutcome, GateResult, RunState
from .gates.consequences import apply_consequence
from .gates.g1 import evaluate_g1
from .gates.g3 import evaluate_g3
from .gates.g4 import evaluate_g4
from .gates.glfo import evaluate_glfo
from .gates.q import evaluate_q
from .gates.retrospective import run_g4
from .report import generate_summary
from .runio import append_gate_result, write_manifest


SCENARIOS = ("q_break_30", "no_reference_states", "no_decoys", "small_families", "shuffled_labels", "perfect_pharmacophore", "mode_a")


def _threshold(loaded, key):
    if key not in loaded.config.thresholds:
        raise ValueError(f"TCC thresholds.{key} is required for forced demos")
    return loaded.config.thresholds[key]


def _perfect_g4(loaded, output_dir: Path) -> GateResult:
    active_n = loaded.config.n_minima.G4_actives
    decoy_n = loaded.config.n_minima.G4_decoys
    count = max(active_n, decoy_n)
    compounds = []
    for index in range(count):
        compounds.extend([
            {"compound_id": f"active_{index}", "smiles": "CCO", "label": 1, "group": f"series_{index}"},
            {"compound_id": f"decoy_{index}", "smiles": "c1ccccc1", "label": 0, "group": f"series_{index}"},
        ])
    arms = {
        "crystal_diffphar": {"states": ["crystal"], "pharmacophore": SimpleNamespace(model_id="crystal_diffphar")},
        "crystal_pharmflow": {"states": ["crystal"], "pharmacophore": SimpleNamespace(model_id="crystal_pharmflow")},
        "ensemble_diffphar": {"states": ["ensemble"], "pharmacophore": SimpleNamespace(model_id="ensemble_diffphar")},
        "ensemble_pharmflow": {"states": ["ensemble"], "pharmacophore": SimpleNamespace(model_id="ensemble_pharmflow")},
    }

    def score(smiles, state_id, constraint, model, seed):
        is_active = smiles == "CCO"
        if model.model_id == "ensemble_pharmflow":
            return -10.0 if is_active else -1.0
        if model.model_id == "crystal_pharmflow":
            return -1.0 if is_active else -10.0
        return -1.0 if is_active else -1.0

    freeze_record = output_dir / "freeze.json"
    freeze(loaded, freeze_record)
    gate, _, _ = run_g4(
        compounds, arms=arms, score_function=score, config_hash=loaded.config_hash,
        minimum_auc_gain=_threshold(loaded, "g4_minimum_auc_gain"), alpha=_threshold(loaded, "familywise_alpha"),
        min_actives=active_n, min_decoys=decoy_n, family_pvalues=None,
        n_boot=_threshold(loaded, "bootstrap_replicates"), n_permutations=_threshold(loaded, "toy_demo_permutations"),
        confidence=_threshold(loaded, "confidence_level"), seed=loaded.config.seed, freeze_record_path=freeze_record,
    )
    return gate


def run_forced_demo(loaded, scenario: str, output_root: str | Path) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown demo scenario {scenario!r}; expected one of {SCENARIOS}")
    out = Path(output_root) / f"demo_{scenario}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "gates").mkdir()
    (out / "report").mkdir()
    state = RunState(mode=loaded.config.mode)
    gate_log = out / "gates" / "gate_log.jsonl"

    def apply(gate):
        apply_consequence(gate, state)
        append_gate_result(gate_log, gate)

    if scenario == "q_break_30":
        total = _threshold(loaded, "toy_demo_conformer_count")
        broken_fraction = _threshold(loaded, "toy_demo_broken_fraction")
        passed = total - int(np.ceil(total * broken_fraction))
        apply(evaluate_q(passed_count=passed, total_count=total, minimum_fraction=_threshold(loaded, "minimum_qc_pass_fraction"), config_hash=loaded.config_hash, evidence_complete=True))
    elif scenario == "no_reference_states":
        apply(evaluate_g1(reference_state_count=0, covered_state_count=0, minimum_coverage=_threshold(loaded, "minimum_reference_coverage"), config_hash=loaded.config_hash))
    elif scenario == "no_decoys":
        apply(evaluate_g4(auc_gain=None, minimum_auc_gain=_threshold(loaded, "g4_minimum_auc_gain"), adjusted_p_value=None, alpha=_threshold(loaded, "familywise_alpha"), actives=loaded.config.n_minima.G4_actives, decoys=0, minimum_actives=loaded.config.n_minima.G4_actives, minimum_decoys=loaded.config.n_minima.G4_decoys, config_hash=loaded.config_hash, reason="forced demo removed all decoys"))
    elif scenario == "small_families":
        apply(evaluate_glfo(trained=True, train_families=_threshold(loaded, "minimum_train_families") - 1, held_out_families=_threshold(loaded, "minimum_held_out_families") - 1, minimum_train_families=_threshold(loaded, "minimum_train_families"), minimum_held_out_families=_threshold(loaded, "minimum_held_out_families"), config_hash=loaded.config_hash))
    elif scenario == "shuffled_labels":
        count = _threshold(loaded, "toy_demo_label_count")
        rng = np.random.default_rng(loaded.config.seed)
        labels = np.arange(count) % 2
        rng.shuffle(labels)
        scores = rng.random(count)
        from sklearn.metrics import roc_auc_score
        auc_value = float(roc_auc_score(labels, scores))
        null = [float(roc_auc_score(rng.permutation(labels), scores)) for _ in range(_threshold(loaded, "toy_demo_permutations"))]
        p_value = float((sum(value >= auc_value for value in null) + 1) / (len(null) + 1))
        apply(evaluate_g3(auc_value=auc_value, permutation_p=p_value, labelled_units=count, minimum_units=loaded.config.n_minima.G3_labelled_units, minimum_auc=_threshold(loaded, "g3_minimum_auc"), maximum_p=_threshold(loaded, "g3_maximum_p"), config_hash=loaded.config_hash))
    elif scenario == "perfect_pharmacophore":
        apply(_perfect_g4(loaded, out))
    elif scenario == "mode_a":
        apply(GateResult("G0", GateOutcome.PASS, ConsequenceType.STOP, config_hash=loaded.config_hash, reason="toy target is applicable"))
        state.premarked_gates["G3"] = "skipped because Mode A has no off-targets"
        state.premarked_gates["D-state"] = "skipped because the target is not Mode D"
    generate_summary(gate_log, state, out / "report" / "summary.md")
    write_manifest(out, config_hash=loaded.config_hash, seed=loaded.config.seed, counts_in={}, counts_out={"gate_results": len(state.gate_log)}, dropped=[], extra={"forced_demo": scenario})
    return out


def main():
    parser = argparse.ArgumentParser(description="Run preregistered forced gate demonstrations")
    parser.add_argument("--tcc", default=str(Path(__file__).parent / "toy_data" / "toy_tcc.yaml"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", choices=[*SCENARIOS, "all"], required=True)
    args = parser.parse_args()
    loaded = load_tcc(args.tcc)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS if args.force == "all" else (args.force,)
    for scenario in scenarios:
        print(run_forced_demo(loaded, scenario, args.output_root))


if __name__ == "__main__":
    main()
