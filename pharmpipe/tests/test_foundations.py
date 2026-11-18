import csv
import json
from pathlib import Path

import pytest

from pharmpipe.config.loader import freeze, load_tcc, require_frozen
from pharmpipe.adapters import MockASAMTAdapter, MockDockingAdapter, MockGCPGAdapter, MockPharmFlowAdapter
from pharmpipe.data.models import PharmModel, PharmPoint, PocketAtom, PocketState
from pharmpipe.data.models import ConsequenceType, GateOutcome, GateResult, RunState
from pharmpipe.data.model_registry import ModelVersion, ModelVersionRegistry
from pharmpipe.gates.consequences import apply_consequence
from pharmpipe.runio import append_gate_result, create_run_folder, write_manifest
from pharmpipe.steps.step0 import run_step0
from pharmpipe.steps.step1 import run_step1
from pharmpipe.steps.step2 import run_step2
from pharmpipe.steps.step3 import run_step3
from pharmpipe.steps.step5 import merge_consensus, run_step5
from pharmpipe.steps.step6 import run_step6
from pharmpipe.steps.pharmacophore_graph import pharmacophore_graph
from pharmpipe.steps.step7 import run_step7
from pharmpipe.steps.step8 import run_step8
from pharmpipe.steps.step9 import run_step9, _fold_assessment
from pharmpipe.run_pipeline import run_pipeline
from pharmpipe.toy_target import SCENARIOS, run_forced_demo
from pharmpipe.leakage import ExclusionSet


def sample_tcc(path, **changes):
    data = {
        "target": {"name": "toy", "construct": {"chain": "A", "residues": [1, 20]}, "pocket_source": "residues", "pocket_residues": ["10"]},
        "off_targets": [], "mode": "A",
    }
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_tcc_mode_must_match_off_targets(tmp_path):
    path = sample_tcc(tmp_path / "bad.yaml", mode="A", off_targets=[{"name": "x", "structure_path": "x.pdb"}])
    with pytest.raises(ValueError, match="disagrees"):
        load_tcc(path)


def test_freeze_requires_matching_hash(tmp_path):
    loaded = load_tcc(sample_tcc(tmp_path / "tcc.yaml"))
    record = tmp_path / "freeze.json"
    frozen_hash = freeze(loaded, record)
    assert frozen_hash
    assert require_frozen(record, loaded.config_hash)["config_hash"] == loaded.config_hash
    with pytest.raises(FileExistsError):
        freeze(loaded, record)


def test_g4_ne_blocks_confirmatory_generation_and_logs(tmp_path):
    state = RunState()
    result = GateResult("G4", GateOutcome.NOT_EVALUABLE, ConsequenceType.STOP, reason="missing decoys")
    apply_consequence(result, state)
    assert not state.confirmatory_generation_allowed
    assert state.use_crystal_baseline_only and state.exploratory_only
    assert state.gate_log == [result]
    path = tmp_path / "gates" / "gate_log.jsonl"
    append_gate_result(path, result)
    assert json.loads(path.read_text(encoding="utf-8"))['outcome'] == "NOT_EVALUABLE"


def test_run_directory_refuses_overwrite_and_manifest_records_provenance(tmp_path):
    tcc = sample_tcc(tmp_path / "tcc.yaml")
    loaded = load_tcc(tcc)
    run_dir = create_run_folder(tmp_path / "runs", loaded, tcc)
    manifest = write_manifest(run_dir / "step0", config_hash=loaded.config_hash, seed=loaded.config.seed)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["config_hash"] == loaded.config_hash
    assert "versions" in record
    with pytest.raises(FileExistsError):
        create_run_folder(tmp_path / "runs", loaded, tcc)


def test_mock_adapters_are_seed_deterministic(tmp_path):
    atoms = [PocketAtom((float(i), 0.0, 0.0), "C", "ALA", str(i), "CA") for i in range(4)]
    state = PocketState("s0", "crystal", tmp_path / "x.pdb", atoms)
    asamt = MockASAMTAdapter()
    assert (asamt.sample([a.xyz for a in atoms], 2, 300, 4)[0] == asamt.sample([a.xyz for a in atoms], 2, 300, 4)[0]).all()
    pharm = MockPharmFlowAdapter().predict(state, 3)
    assert pharm == MockPharmFlowAdapter().predict(state, 3)
    gcpg = MockGCPGAdapter()
    assert gcpg.generate(pharm, 10, {}, 2) == gcpg.generate(pharm, 10, {}, 2)
    assert all(smiles for smiles in gcpg.generate(pharm, 10, {}, 2))
    assert MockDockingAdapter().score("CCO", "s0", 9) == MockDockingAdapter().score("CCO", "s0", 9)


def test_cmd_flow_matching_adapter():
    from pharmpipe.adapters import CMDFlowMatchingAdapter
    repo_dir = Path("pharmflow_repo")
    ckpt_path = repo_dir / "untrained_flow_matching.ckpt"
    pdb_path = Path("conf_0002.pdb")
    if not (repo_dir.exists() and ckpt_path.exists() and pdb_path.exists()):
        return
    state = PocketState(
        state_id="conf_0002",
        source="crystal",
        coordinate_path=pdb_path.resolve(),
        pocket_atoms=[PocketAtom((0.0, 0.0, 0.0), "C", "ALA", "1", "CA")],
        cluster_id="1",
        weight=None,
        in_domain=True,
    )
    adapter = CMDFlowMatchingAdapter(
        repository_dir=repo_dir,
        checkpoint_path=ckpt_path,
        samples_per_state=1,
        timesteps=10,
    )
    model = adapter.predict(state, seed=42)
    assert model.model_id == "pharmflow_conf_0002"
    assert len(model.points) > 0
    assert all(pt.type in ("AROM", "HYBL", "POSC", "NEGI", "HACC", "HDON", "LHYBL", "OTHERS") for pt in model.points)


def test_cmd_gcpg_adapter():
    from pharmpipe.adapters import CMDGCPGAdapter
    repo_dir = Path("cmdgen_repo")
    model_path = repo_dir / "GCPG" / "checkpoints" / "model_checkpoints" / "checkpoints" / "final_model.pth"
    tokenizer_path = repo_dir / "GCPG" / "checkpoints" / "model_checkpoints" / "checkpoints" / "tokenizer_r_iso.pkl"
    if not (repo_dir.exists() and model_path.exists() and tokenizer_path.exists()):
        return
    adapter = CMDGCPGAdapter(
        repository_dir=repo_dir,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device="cpu",
    )
    assert adapter.is_available
    model = PharmModel(
        model_id="test_gcpg_model",
        points=[
            PharmPoint((10.0, 12.0, 15.0), "AROM"),
            PharmPoint((14.0, 11.5, 16.0), "HYBL"),
            PharmPoint((11.0, 16.0, 14.5), "HACC"),
        ],
        origin="consensus",
    )
    smiles = adapter.generate(model, count=1, property_gates={}, seed=42)
    assert len(smiles) >= 1
    assert isinstance(smiles[0], str)
    assert len(smiles[0]) > 0


def test_vina_docking_adapter():
    from pharmpipe.adapters import VinaDockingAdapter
    vina_exe = Path("bin/vina.exe")
    pdb_path = Path("conf_0002.pdb")
    if not (vina_exe.exists() and pdb_path.exists()):
        return
    adapter = VinaDockingAdapter(
        vina_executable=vina_exe,
        receptor_paths={"conf_0002": pdb_path},
        exhaustiveness=1,
    )
    assert adapter.is_available
    score = adapter.score("c1ccccc1NC(=O)C", "conf_0002", seed=42)
    assert isinstance(score, float)
    assert score < 0.0


def test_step0_missing_structure_is_not_evaluable(tmp_path):
    tcc = sample_tcc(tmp_path / "tcc.yaml")
    loaded = load_tcc(tcc)
    result = run_step0(loaded, target_structure=tmp_path / "absent.pdb", output_dir=tmp_path / "step0")
    assert result.gate.outcome == GateOutcome.NOT_EVALUABLE
    state = RunState()
    apply_consequence(result.gate, state)
    assert state.target_run_stopped
    assert (tmp_path / "step0" / "applicability_report.json").is_file()


def test_step0_without_decoys_continues_and_premarks_retrospective_gates(tmp_path):
    active = tmp_path / "actives.csv"
    active.write_text("ligand_id,smiles\na1,CCO\n", encoding="utf-8")
    tcc = sample_tcc(tmp_path / "tcc.yaml", reference={"actives_path": str(active)})
    content = json.loads(tcc.read_text(encoding="utf-8"))
    content["target"]["pocket_residues"] = ["1"]
    tcc.write_text(json.dumps(content), encoding="utf-8")
    loaded = load_tcc(tcc)
    structure = Path(__file__).parents[1] / "toy_data" / "toy.pdb"
    result = run_step0(loaded, target_structure=structure, output_dir=tmp_path / "step0")
    assert result.gate.outcome == GateOutcome.PASS
    assert result.applicability_report["premarked_gates"]["G4"] == "decoys missing"
    assert result.applicability_report["premarked_gates"]["G2b"] == "decoys missing"


def test_steps1_to3_write_files_and_missing_reference_is_ne(tmp_path):
    source = tmp_path / "seed.pdb"
    atoms = [("N", 0.0, 0.0, 0.0, "N"), ("CA", 1.45, 0.0, 0.0, "C"), ("C", 2.0, 1.4, 0.0, "C"), ("O", 1.5, 2.4, 0.0, "O")]
    source.write_text("".join(
        f"ATOM  {index:5d} {name:>4s} ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n"
        for index, (name, x, y, z, element) in enumerate(atoms, start=1)
    ) + "END\n", encoding="utf-8")
    loaded = load_tcc(sample_tcc(tmp_path / "tcc.yaml"))
    stage1 = run_step1(loaded, seeds=[{"seed_id": "seed0", "seed_type": "apo-seeded", "structure_path": source}], held_out=[], exclusion_set=ExclusionSet(set(), {}), n_samples_per_seed=1, temperature=300, output_dir=tmp_path / "step1")
    stage2 = run_step2(loaded, conformer_paths=stage1.conformer_paths, output_dir=tmp_path / "step2", minimum_pass_fraction=1.0, maximum_clash_distance_A=0.1, maximum_energy_z=3.0)
    assert stage2.gate.outcome == GateOutcome.NOT_EVALUABLE
    stage3 = run_step3(loaded, surviving_paths=stage2.surviving_paths, output_dir=tmp_path / "step3", pocket_residues=["1"], source_types={"conf_0000": "apo-seeded"}, cluster_distance_A=1.5, k_states=1, minimum_reference_coverage=0.9, reference_coverage_tolerance_A=1.5, pocket_distance_weights={"element": 1.0}, pocket_distance_scales={"element": 1.0}, pocket_distance_lambda=0.1)
    assert stage3.g1.outcome == GateOutcome.NOT_EVALUABLE
    assert stage3.states[0].weight is None
    assert stage3.states_path.is_file()


def test_step1_loads_precomputed_asam_frames_without_sampling(tmp_path):
    source = tmp_path / "seed.pdb"
    atom_line = "ATOM      1   CA ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    source.write_text(atom_line + "END\n", encoding="utf-8")
    ensemble = tmp_path / "asam"
    ensemble.mkdir()
    for index, x in enumerate((1.0, 2.0), start=1):
        frame = atom_line[:30] + f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}" + atom_line[54:]
        (ensemble / f"conf_{index:04d}.pdb").write_text(frame + "END\n", encoding="utf-8")
    (ensemble / ".done").write_text("COMPLETED_2_FRAMES\n", encoding="utf-8")

    class FailingSampler:
        version = "must-not-run"

        def sample(self, *args, **kwargs):
            raise AssertionError("precomputed mode must not invoke a sampler")

    loaded = load_tcc(sample_tcc(tmp_path / "tcc.yaml"))
    stage1 = run_step1(
        loaded,
        seeds=[{"seed_id": "seed0", "seed_type": "apo-seeded", "structure_path": source}],
        held_out=[],
        exclusion_set=ExclusionSet(set(), {}),
        n_samples_per_seed=2,
        temperature=300,
        output_dir=tmp_path / "step1-precomputed",
        adapter=FailingSampler(),
        precomputed_ensemble_dir=ensemble,
    )

    assert len(stage1.conformer_paths) == 2
    with stage1.ensemble_table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sampler_version"] for row in rows] == ["aSAM-precomputed"] * 2
    manifest = json.loads((tmp_path / "step1-precomputed" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["ensemble_source"] == "precomputed-aSAM"


def test_q_not_evaluable_stops_only_ensemble_branch():
    state = RunState()
    result = GateResult("Q", GateOutcome.NOT_EVALUABLE, ConsequenceType.STOP, reason="physical checks unavailable")
    apply_consequence(result, state)
    assert state.ensemble_branch_stopped
    assert not state.target_run_stopped
    assert state.use_crystal_baseline_only


def test_consensus_of_repeated_identical_models_preserves_points_and_vote_count():
    points = [PharmPoint((float(i), 0.0, 0.0), "HDON") for i in range(4)]
    models = [PharmModel(f"m{i}", points, "per-state", [f"s{i}"], [1] * len(points)) for i in range(5)]
    consensus = merge_consensus(models, total_state_count=5, group_distance_A=0.1, vote_fraction=1.0)
    assert consensus.points == points
    assert consensus.vote_counts == [5] * len(points)


def test_step5_mode_a_skips_differential_path(tmp_path):
    loaded = load_tcc(sample_tcc(tmp_path / "tcc.yaml"))
    model = PharmModel("one", [PharmPoint((1.0, 2.0, 3.0), "HDON")], "per-state", ["s0"], [1])
    result = run_step5(loaded, target_models=[model], total_state_count=1, output_dir=tmp_path / "step5", group_distance_A=1.0, vote_fraction=1.0)
    assert result.differential is None
    assert result.g3 is None
    assert result.output_path.is_file()


def test_step6_exploratory_uses_crystal_model_and_writes_properties(tmp_path):
    loaded = load_tcc(sample_tcc(tmp_path / "tcc.yaml"))
    state = RunState(confirmatory_generation_allowed=False)
    crystal = PharmModel("crystal_baseline", [PharmPoint((0.0, 0.0, 0.0), "OTHERS")], "per-state", ["crystal"], [1])
    generated = PharmModel("ensemble_consensus", [PharmPoint((0.0, 0.0, 0.0), "OTHERS")], "consensus", ["s0"], [1])
    result = run_step6(loaded, run_state=state, models=[generated], crystal_baseline_model=crystal, output_dir=tmp_path / "step6", count_per_model=2, repeats=1, conformer_count=1, edge_distance_mode="euclidean_angstrom", match_distance_A=1.0, max_unmatched_points=1)
    assert result.molecules_csv.is_file()
    assert result.property_table.is_file()
    assert all(molecule.flags["exploratory"] for molecule in result.molecules)
    assert {molecule.source_model_id for molecule in result.molecules} <= {"crystal_baseline"}
    graph = pharmacophore_graph(crystal, edge_distance_mode="euclidean_angstrom")
    assert graph["edge_distances_A"] == [[0.0]]


def test_steps7_to9_keep_attrition_scoring_and_censored_bounds_explicit(tmp_path):
    loaded = load_tcc(sample_tcc(tmp_path / "tcc.yaml"))
    generated_csv = tmp_path / "molecules.csv"
    generated_csv.write_text('compound_id,smiles,properties\nmol0,CCO,"{""MW"":46.07,""LogP"":0.0,""QED"":0.4,""SAS"":1.5}"\n', encoding="utf-8")
    filtered = run_step7(loaded, molecules_csv=generated_csv, output_dir=tmp_path / "step7", drop_alerts=False, max_ecfp4_similarity=None, drop_known_scaffolds=False, drop_property_failures=False)
    assert len(filtered["kept"]) == 1
    scoring = run_step8(loaded, filtered_molecules_csv=filtered["filtered_molecules"], target_states=["s0"], off_target_states={}, output_dir=tmp_path / "step8", mmgbsa_top_count=1, minimum_fep_replicas=3)
    assert scoring["gate"].outcome == GateOutcome.NOT_EVALUABLE
    assert scoring["docking"].is_file()
    assert _fold_assessment({"value": "100", "relation": "=", "assay_type": "x", "assay_conditions": "y"}, {"value": "10000", "relation": ">", "assay_type": "x", "assay_conditions": "y"}, 10)["ratio_lower_bound"] == 100


def test_step9_mixed_endpoints_raise(tmp_path):
    loaded = load_tcc(sample_tcc(tmp_path / "tcc.yaml"))
    assay = tmp_path / "assays.csv"
    assay.write_text("compound_id,protein,endpoint,value,relation,units,assay_type,assay_conditions,replicate_id,validity\nm0,toy,Ki,1,=,nM,bio,buffer,r1,valid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="endpoints or units"):
        run_step9(loaded, assay_csv=assay, compounds=[{"compound_id": "m0"}], output_dir=tmp_path / "step9", potency_goal=100, endpoint="IC50", expected_units="nM", md_replicas=3, minimum_descriptive_n=5, wilson_confidence=0.95)


def test_mock_end_to_end_smoke_and_determinism(tmp_path):
    tcc = Path(__file__).parents[1] / "toy_data" / "toy_tcc.yaml"
    first = run_pipeline(tcc, structure_path=None, output_root=tmp_path / "run_a")
    second = run_pipeline(tcc, structure_path=None, output_root=tmp_path / "run_b")
    assert (first / "gates" / "gate_log.jsonl").is_file()
    assert (first / "report" / "summary.md").is_file()
    files_a = {path.relative_to(first).as_posix(): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    files_b = {path.relative_to(second).as_posix(): path.read_bytes() for path in second.rglob("*") if path.is_file()}
    assert files_a == files_b


def test_all_forced_gate_demos_write_reports(tmp_path):
    tcc = Path(__file__).parents[1] / "toy_data" / "toy_tcc.yaml"
    loaded = load_tcc(tcc)
    outcomes = {}
    for scenario in SCENARIOS:
        demo = run_forced_demo(loaded, scenario, tmp_path)
        assert (demo / "report" / "summary.md").is_file()
        outcomes[scenario] = json.loads((demo / "gates" / "gate_log.jsonl").read_text(encoding="utf-8").splitlines()[0])["outcome"]
    assert outcomes["q_break_30"] == "FAIL"
    assert outcomes["no_reference_states"] == "NOT_EVALUABLE"
    assert outcomes["no_decoys"] == "NOT_EVALUABLE"
    assert outcomes["small_families"] == "NOT_EVALUABLE"
    assert outcomes["shuffled_labels"] != "PASS"
    assert outcomes["perfect_pharmacophore"] == "PASS"


def test_model_version_registry_prevents_test_leakage_and_overwrite(tmp_path):
    registry = ModelVersionRegistry(tmp_path / "models.json")
    record = ModelVersion("v1", "artifact", None, ("train_a",), ("test_a",), 0.95, "freeze")
    registry.register(record)
    assert registry.get("v1")["artifact_hash"] == "artifact"
    with pytest.raises(FileExistsError):
        registry.register(record)
    leaking = ModelVersion("v2", "artifact", "v1", ("test_a",), ("test_a",), 0.95, "freeze")
    with pytest.raises(ValueError, match="held-out"):
        registry.register(leaking)
