"""Mock-capable file-oriented pipeline driver."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .adapters.pharmflow import MockPharmFlowAdapter
from .adapters.cmd_flow_matching import CMDFlowMatchingAdapter
from .adapters.gcpg import MockGCPGAdapter
from .adapters.cmd_gcpg import CMDGCPGAdapter
from .adapters.docking import MockDockingAdapter, VinaDockingAdapter
from .config.loader import freeze, load_tcc
from .data.models import ConsequenceType, GateOutcome, GateResult, Molecule, PocketState, RunState
from .gates.consequences import apply_consequence
from .gates.g4 import evaluate_g4
from .leakage import ExclusionSet, LigandRecord, build_exclusion_set
from .report import generate_summary
from .runio import append_gate_result, create_run_folder
from .steps.step0 import run_step0
from .steps.step1 import run_step1
from .steps.step2 import run_step2
from .steps.step3 import run_step3
from .steps.step4 import run_step4
from .steps.step5 import run_step5
from .steps.step6 import run_step6
from .steps.step7 import run_step7
from .steps.step8 import run_step8
from .steps.step9 import run_step9


def _required(config, key):
    if key not in config.thresholds:
        raise ValueError(f"TCC thresholds.{key} is required; pipeline will not invent a value")
    return config.thresholds[key]


def _read_ligands(path: str | None) -> list[LigandRecord]:
    if not path or not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [LigandRecord(row.get("ligand_id") or row.get("compound_id") or f"ligand_{index}", row["smiles"]) for index, row in enumerate(rows) if row.get("smiles")]


def run_pipeline(tcc_path: str | Path, *, structure_path: str | Path | None, output_root: str | Path) -> Path:
    loaded = load_tcc(tcc_path)
    selected_structure = structure_path or loaded.config.target.structure_path
    thresholds = loaded.config.thresholds
    precomputed_ensemble_dir = thresholds.get("precomputed_ensemble_dir")
    pharmflow_repo = thresholds.get("pharmflow_repository_dir")
    pharmflow_checkpoint = thresholds.get("pharmflow_checkpoint_path")
    pharmflow_keys = (pharmflow_repo, pharmflow_checkpoint)
    if any(pharmflow_keys) and not all(pharmflow_keys):
        raise ValueError("Set both thresholds.pharmflow_repository_dir and thresholds.pharmflow_checkpoint_path to enable real PharmFlow")
    input_paths = [selected_structure] if selected_structure else []
    if precomputed_ensemble_dir:
        ensemble_dir = Path(precomputed_ensemble_dir)
        input_paths.extend([ensemble_dir / ".done", *sorted(ensemble_dir.glob("conf_*.pdb"))])
    if pharmflow_checkpoint:
        input_paths.append(pharmflow_checkpoint)
    if pharmflow_repo:
        repository = Path(pharmflow_repo)
        input_paths.extend(sorted(path for path in repository.rglob("*.py") if ".git" not in path.parts and ".venv" not in path.parts and "venv" not in path.parts))
    run_dir = create_run_folder(output_root, loaded, tcc_path, input_paths=input_paths)
    freeze(loaded, run_dir / "config" / "freeze.json")
    run_state = RunState(mode=loaded.config.mode)
    gate_log = run_dir / "gates" / "gate_log.jsonl"

    def apply(gate: GateResult) -> bool:
        apply_consequence(gate, run_state)
        append_gate_result(gate_log, gate)
        return run_state.target_run_stopped

    step0 = run_step0(loaded, target_structure=structure_path, output_dir=run_dir / "step0")
    run_state.premarked_gates.update(step0.applicability_report["premarked_gates"])
    if apply(step0.gate):
        generate_summary(gate_log, run_state, run_dir / "report" / "summary.md")
        return run_dir

    structure = Path(structure_path or loaded.config.target.structure_path or "")
    actives = _read_ligands(loaded.config.reference.actives_path)
    decoys = _read_ligands(loaded.config.reference.decoys_path)
    held_out = actives
    exclusion = build_exclusion_set(
        held_out, [*actives, *decoys],
        min_ecfp_tanimoto=_required(loaded.config, "leakage_ecfp4_tanimoto"),
        min_pharmacophore_tanimoto=_required(loaded.config, "leakage_pharmacophore_tanimoto"),
        min_pharmacophore_f1=_required(loaded.config, "leakage_pharmacophore_f1"),
        pharmacophore_tolerance=_required(loaded.config, "pharmacophore_match_distance_A"),
    )
    seed_type = _required(loaded.config, "seed_type")
    seed_record = {"seed_id": _required(loaded.config, "seed_id"), "seed_type": seed_type, "structure_path": structure, "ligand_id": thresholds.get("seed_ligand_id")}
    stage1 = run_step1(
        loaded,
        seeds=[seed_record],
        held_out=held_out,
        exclusion_set=exclusion,
        n_samples_per_seed=_required(loaded.config, "n_samples_per_seed"),
        temperature=_required(loaded.config, "sampling_temperature_K"),
        output_dir=run_dir / "step1",
        precomputed_ensemble_dir=precomputed_ensemble_dir,
    )
    stage2 = run_step2(
        loaded,
        conformer_paths=stage1.conformer_paths,
        output_dir=run_dir / "step2",
        minimum_pass_fraction=_required(loaded.config, "minimum_qc_pass_fraction"),
        maximum_clash_distance_A=_required(loaded.config, "maximum_clash_distance_A"),
        maximum_vdw_overlap_A=_required(loaded.config, "maximum_vdw_overlap_A"),
        maximum_energy_z=_required(loaded.config, "maximum_energy_z"),
        reference_path=structure,
        pocket_residues=step0.pocket_residues,
        maximum_pocket_rmsd_A=_required(loaded.config, "maximum_pocket_rmsd_A"),
    )
    if apply(stage2.gate):
        generate_summary(gate_log, run_state, run_dir / "report" / "summary.md")
        return run_dir

    if run_state.ensemble_branch_stopped:
        stage3_inputs = [structure]
        source_types = {structure.stem: "crystal"}
    else:
        stage3_inputs = stage2.surviving_paths
        source_by_id = {row["state_id"]: row["seed_type"] for row in csv.DictReader((run_dir / "step1" / "ensemble_table.csv").open("r", encoding="utf-8"))}
        source_types = {Path(path).stem: source_by_id.get(Path(path).stem, seed_type) for path in stage3_inputs}
    stage3 = run_step3(
        loaded, surviving_paths=stage3_inputs, output_dir=run_dir / "step3", pocket_residues=step0.pocket_residues,
        source_types=source_types, cluster_distance_A=_required(loaded.config, "pocket_cluster_distance_A"),
        k_states=1 if run_state.ensemble_branch_stopped else loaded.config.K_states,
        minimum_reference_coverage=_required(loaded.config, "minimum_reference_coverage"),
        reference_coverage_tolerance_A=_required(loaded.config, "reference_coverage_tolerance_A"),
        pocket_distance_weights=_required(loaded.config, "pocket_distance_weights"),
        pocket_distance_scales=_required(loaded.config, "pocket_distance_scales"),
        pocket_distance_lambda=_required(loaded.config, "pocket_distance_lambda"),
    )
    for gate in (stage3.g1, stage3.g2b):
        if apply(gate):
            generate_summary(gate_log, run_state, run_dir / "report" / "summary.md")
            return run_dir

    pharmflow_adapter = MockPharmFlowAdapter()
    if pharmflow_repo:
        pharmflow_adapter = CMDFlowMatchingAdapter(
            repository_dir=pharmflow_repo,
            checkpoint_path=pharmflow_checkpoint,
            samples_per_state=int(_required(loaded.config, "pharmflow_samples_per_state")),
            timesteps=int(_required(loaded.config, "pharmflow_timesteps")),
            python_executable=thresholds.get("pharmflow_python_executable") or "python",
        )
    stage4 = run_step4(loaded, states=stage3.states, output_dir=run_dir / "step4", minimum_train_families=_required(loaded.config, "minimum_train_families"), minimum_held_out_families=_required(loaded.config, "minimum_held_out_families"), adapter=pharmflow_adapter)
    for gate in (stage4.g_lfo, stage4.g2a):
        if apply(gate):
            generate_summary(gate_log, run_state, run_dir / "report" / "summary.md")
            return run_dir
    stage5 = run_step5(
        loaded, target_models=stage4.models, total_state_count=max(len(stage3.states), 1), output_dir=run_dir / "step5",
        group_distance_A=loaded.config.merge.group_distance_A, vote_fraction=loaded.config.merge.vote_fraction,
        off_target_models={} if loaded.config.off_targets else None,
        **({"presence_tolerance_A": _required(loaded.config, "differential_presence_tolerance_A"),
           "minimum_selectivity_score": _required(loaded.config, "differential_minimum_score"),
           "minimum_differential_ci_lower_bound": _required(loaded.config, "minimum_differential_ci_lower_bound"),
           "n_boot": _required(loaded.config, "bootstrap_replicates"), "confidence": _required(loaded.config, "confidence_level"),
           "g3_minimum_auc": _required(loaded.config, "g3_minimum_auc"), "g3_maximum_p": _required(loaded.config, "g3_maximum_p")} if loaded.config.off_targets else {}),
    )
    for gate in (stage5.g3, stage5.d_state):
        if gate and apply(gate):
            generate_summary(gate_log, run_state, run_dir / "report" / "summary.md")
            return run_dir

    baseline_state = next((state for state in stage3.states if state.source == "crystal"), stage3.states[0])
    baseline_model = MockPharmFlowAdapter().predict(baseline_state, loaded.config.seed)
    g4 = evaluate_g4(auc_gain=None, minimum_auc_gain=_required(loaded.config, "g4_minimum_auc_gain"), adjusted_p_value=None, alpha=_required(loaded.config, "familywise_alpha"), actives=len(actives), decoys=len(decoys), minimum_actives=loaded.config.n_minima.G4_actives, minimum_decoys=loaded.config.n_minima.G4_decoys, config_hash=loaded.config_hash, reason="G4 requires confirmatory PharmFlow and DiffPhar arm scores; only mock PharmFlow is in this build")
    if apply(g4):
        generate_summary(gate_log, run_state, run_dir / "report" / "summary.md")
        return run_dir

    gcpg_repo = thresholds.get("gcpg_repository_dir")
    gcpg_model = thresholds.get("gcpg_model_path")
    gcpg_tokenizer = thresholds.get("gcpg_tokenizer_path")
    gcpg_adapter = MockGCPGAdapter()
    if gcpg_repo and gcpg_model and gcpg_tokenizer:
        gcpg_adapter = CMDGCPGAdapter(
            repository_dir=gcpg_repo,
            model_path=gcpg_model,
            tokenizer_path=gcpg_tokenizer,
            python_executable=thresholds.get("gcpg_python_executable") or sys.executable,
            device=thresholds.get("gcpg_device") or "cpu",
        )

    stage6 = run_step6(loaded, run_state=run_state, models=[stage5.consensus] if stage5.consensus.points else [], crystal_baseline_model=baseline_model, output_dir=run_dir / "step6", count_per_model=_required(loaded.config, "generation_count_per_model"), repeats=_required(loaded.config, "generation_repeats"), conformer_count=_required(loaded.config, "conformer_count"), edge_distance_mode=_required(loaded.config, "edge_distance_mode"), match_distance_A=_required(loaded.config, "pharmacophore_match_distance_A"), max_unmatched_points=_required(loaded.config, "max_unmatched_pharmacophore_points"), adapter=gcpg_adapter)
    stage7 = run_step7(loaded, molecules_csv=stage6.molecules_csv, output_dir=run_dir / "step7", drop_alerts=_required(loaded.config, "drop_structural_alerts"), max_ecfp4_similarity=_required(loaded.config, "maximum_known_analog_similarity"), drop_known_scaffolds=_required(loaded.config, "drop_known_scaffolds"), drop_property_failures=_required(loaded.config, "drop_property_failures"))
    docking_adapter: MockDockingAdapter | VinaDockingAdapter = MockDockingAdapter()
    vina_exe = thresholds.get("vina_executable") or (Path("bin") / "vina.exe" if (Path("bin") / "vina.exe").is_file() else None)
    if thresholds.get("docking_engine") == "vina" or (vina_exe and thresholds.get("docking_engine") != "mock"):
        state_paths = {state.state_id: Path(state.coordinate_path) for state in stage3.states if state.coordinate_path}
        docking_adapter = VinaDockingAdapter(
            vina_executable=vina_exe or "bin/vina.exe",
            receptor_paths=state_paths,
            default_receptor_path=structure if structure.is_file() else None,
            exhaustiveness=int(thresholds.get("vina_exhaustiveness", 4)),
            box_size_A=float(thresholds.get("vina_box_size_A", 20.0)),
        )

    stage8 = run_step8(
        loaded,
        filtered_molecules_csv=stage7["filtered_molecules"],
        target_states=[state.state_id for state in stage3.states],
        off_target_states={},
        output_dir=run_dir / "step8",
        mmgbsa_top_count=_required(loaded.config, "mmgbsa_top_count"),
        minimum_fep_replicas=_required(loaded.config, "minimum_fep_replicas"),
        docking=docking_adapter,
    )
    if apply(stage8["gate"]):
        generate_summary(gate_log, run_state, run_dir / "report" / "summary.md")
        return run_dir
    assay_path = thresholds.get("assay_csv")
    stage9 = run_step9(loaded, assay_csv=assay_path, compounds=[], output_dir=run_dir / "step9", potency_goal=_required(loaded.config, "potency_goal"), endpoint=_required(loaded.config, "assay_endpoint"), expected_units=_required(loaded.config, "assay_units"), md_replicas=_required(loaded.config, "md_replicas"), minimum_descriptive_n=_required(loaded.config, "minimum_descriptive_n"), wilson_confidence=_required(loaded.config, "confidence_level"))
    apply(stage9["gate"])
    generate_summary(gate_log, run_state, run_dir / "report" / "summary.md")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock-capable ensemble-aware pharmacophore pipeline")
    parser.add_argument("--tcc", required=True)
    parser.add_argument("--structure", required=False)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run_dir = run_pipeline(args.tcc, structure_path=args.structure, output_root=args.output_root)
    print(run_dir)


if __name__ == "__main__":
    main()
