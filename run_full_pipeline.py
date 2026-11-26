"""
run_full_pipeline.py - Master Pipeline Orchestrator for Enhanced CMD-Gen.

Executes the complete 4-step structure-based drug generation workflow:
1. aSAM Protein Flexibility -> Dynamic Breathing States (S1...S5)
2. PharmFlow -> Continuous Flow Matching 3D Pharmacophore Point Clouds
3. Consensus GMM Merging -> Consensus Interaction Blueprint (.posp)
4. GCPG -> Conditional De Novo Molecular Generation (SMILES & 3D SDF)
5. Multi-State AutoDock Vina Docking -> Top Selective Drug Hits

Monitored and regulated by the 11-Gate Quality Control Framework:
G0, Q, G1, G-LFO, G2a, G2b, G3, D-State, G4 (GO/NO-GO), FEP, G7-8.
"""

from __future__ import annotations

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PHARMFLOW_CKPT = PROJECT_ROOT / "checkpoints_pharmflow" / "best_model_epoch_46.ckpt"
DEFAULT_GCPG_PYTHON = Path(r"C:\Users\hanut sai\miniconda3\envs\gcpg\python.exe")
DEFAULT_MEEKO_PYTHON = Path(r"C:\Users\hanut sai\miniconda3\envs\diff_phar\python.exe")


def print_banner():
    banner = r"""
================================================================================
          CMD-GEN ENHANCED: DYNAMIC ENSEMBLE DE NOVO DRUG DISCOVERY
   [aSAM Dynamics] -> [PharmFlow CFM] -> [GMM Consensus] -> [GCPG Generation]
                           [Multi-State Docking]
================================================================================
"""
    print(banner)


def print_gate_report(gate_results: List[GateResult]):
    print("\n" + "="*80)
    print(f"{'GATE ID':<10} {'GATE NAME':<35} {'STATUS':<12} {'CONSEQUENCE'}")
    print("-" * 80)
    for g in gate_results:
        status_color = g.status
        print(f"{g.gate_id:<10} {g.name:<35} {status_color:<12} {g.consequence}")
        print(f"   |-- {g.message}")
    print("="*80 + "\n")


def generate_markdown_report(
    output_path: Path,
    target_pdb: str,
    pocket_center: List[float],
    pocket_radius: float,
    pocket_note: str,
    allow_bad_receptor_residues: bool,
    k_states: int,
    exhaustiveness: int,
    gate_results: List[GateResult],
    top_candidates: List[Dict[str, Any]],
    blueprint_data: Dict[str, Any],
):
    lines = [
        "# CMD-Gen Enhanced: Pipeline Run Report",
        "",
        f"**Target Receptor:** `{Path(target_pdb).name}`  ",
        f"**Pocket Center (Å):** `({pocket_center[0]:.2f}, {pocket_center[1]:.2f}, {pocket_center[2]:.2f})`; radius `{pocket_radius:.1f} Å`  ",
        f"**Pocket Center Source:** {pocket_note}  ",
        f"**Meeko Residue Handling:** {'unmatched residues may be omitted' if allow_bad_receptor_residues else 'strict; preparation fails on unmatched residues'}  ",
        f"**Conformational Ensemble:** {k_states} aSAM-sampled state{'s' if k_states != 1 else ''}  ",
        f"**Flow Matching Backbone:** Continuous Flow Matching (Optimal Transport, Euler integration)  ",
        f"**Molecular Generator:** GCPG Conditioned Graph Transformer  ",
        f"**Docking:** AutoDock Vina multi-conformation screen (exhaustiveness {exhaustiveness})  ",
        "",
        "---",
        "",
        "## 1. Checks and evidence status",
        "",
        "PASS means the specific computational check ran and passed. NOT EVALUATED means required independent benchmark or experimental data were unavailable. Docking scores are computational prioritization scores, not binding free energies or potency measurements.",
        "",
        "| Gate ID | Stage | Gate Name | Status | Action / Consequence | Details |",
        "|---|---|---|---|---|---|",
    ]

    for g in gate_results:
        status_badge = f"**{g.status}**"
        lines.append(f"| `{g.gate_id}` | Stage | {g.name} | {status_badge} | `{g.consequence}` | {g.message} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. 3D Consensus Pharmacophore Blueprint",
        "",
        f"Generated **{blueprint_data['n_features']} consensus pharmacophore features** from sampled structures. State agreement is reported per feature:",
        "",
        "| Feature ID | Pharmacophore Type | Coordinates (X, Y, Z) | State Agreement | Interaction Dispersion |",
        "|---|---|---|---|---|",
    ])

    for i, feat in enumerate(blueprint_data['features']):
        c = feat['center']
        coord_str = f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
        lines.append(f"| Anchor #{i+1} | `{feat['type']}` | {coord_str} | {feat['state_coverage']*100:.0f}% | {feat['dispersion']:.3f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Top-Performing Drug Candidates (Multi-State Docking)",
        "",
        "| Rank | Candidate | Canonical SMILES | Mean $\\Delta G$ (kcal/mol) | Worst-Case $\\Delta G$ | Cross-State Variance |",
        "|---|---|---|---|---|---|",
    ])

    for i, cand in enumerate(top_candidates):
        mean_aff = cand['mean_affinity_kcal_mol']
        worst_aff = cand['worst_case_affinity_kcal_mol']
        var = cand['ensemble_variance']
        lines.append(f"| **#{i+1}** | `{cand['name']}` | `{cand['smiles']}` | **{mean_aff:.2f}** | {worst_aff:.2f} | {var:.3f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Interpretation and limitations",
        "",
        "- This run reports model-generated pharmacophore features, generated structures, and Vina scores only.",
        "- No comparative benchmark, off-target free-energy calculation, or wet-lab dose-response data were included. Gates requiring those data remain NOT EVALUATED.",
        "- Vina scores are approximate docking rankings and do not establish affinity, selectivity, IC50, safety, or efficacy.",
    ])

    output_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"\n[Report] Comprehensive markdown report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CMD-Gen Enhanced 5-Stage Pipeline Runner")
    parser.add_argument("--pdb", type=str, default=r"asam_engine\examples\input\6h49_A.pdb", help="Target receptor PDB")
    parser.add_argument("--pocket_center", type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'), help="Known binding-pocket center in PDB Angstrom coordinates")
    parser.add_argument("--pocket_radius", type=float, default=10.0, help="Pocket selection radius in Angstroms")
    parser.add_argument("--pocket_note", type=str, default="User supplied; independently validate the binding site", help="Provenance note for the pocket center, recorded in the report")
    parser.add_argument("--k_states", type=int, default=2, help="Number of breathing conformational states (default: 2)")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_PHARMFLOW_CKPT), help="PharmFlow Flow Matching checkpoint")
    parser.add_argument("--gcpg_python", type=str, default=str(DEFAULT_GCPG_PYTHON), help="Python executable for the GCPG runtime")
    parser.add_argument("--meeko_python", type=str, default=None, help="Python environment containing Meeko 0.8 and a compatible RDKit (defaults to the diff_phar env)")
    parser.add_argument("--vina_exe", type=str, default="vina.exe", help="AutoDock Vina executable")
    parser.add_argument("--allow_bad_receptor_residues", action="store_true", help="Allow Meeko to omit residues it cannot template; report this setting")
    parser.add_argument("--n_mol", type=int, default=6, help="Number of molecules to generate")
    parser.add_argument("--exhaustiveness", type=int, default=2, help="AutoDock Vina exhaustiveness (lower is faster)")
    parser.add_argument("--out_dir", type=str, default="pipeline_run_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device for PharmFlow and aSAM (cuda or cpu; aSAM falls back to CPU if CUDA fails)")
    args = parser.parse_args()

    meeko_python = args.meeko_python or str(DEFAULT_MEEKO_PYTHON)
    for label, path in (
        ("Receptor PDB", args.pdb),
        ("PharmFlow checkpoint", args.ckpt),
        ("GCPG Python", args.gcpg_python),
        ("Meeko Python", meeko_python),
        ("AutoDock Vina", args.vina_exe),
    ):
        if not Path(path).is_file():
            parser.error(f"{label} not found: {path}")

    # Keep --help usable in lightweight environments without importing the ML stack.
    from gates import PipelineGates, GateResult
    from step1_asam import ASAMEnsembleGenerator
    from step2_pharmflow import PharmFlowSampler
    from step3_consensus import ConsensusPharmacophoreMerger
    from step4_gcpg import GCPGMoleculeGenerator
    from step5_docking import MultiStateDockingEngine

    print_banner()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gate_results: List[GateResult] = []
    start_time = time.time()

    # =========================================================================
    # STEP 1: aSAM Protein Flexibility & Pocket States
    # =========================================================================
    print(f"\n>>> STEP 1: aSAM Conformational Ensemble Generation ({args.k_states} states)")
    if args.k_states > 1:
        attempts = [args.k_states, args.k_states, 1, 1, 1]
    else:
        attempts = [1, 1, 1]
    ensemble_pdbs = None
    step1_gates = []
    actual_k_states = args.k_states
    last_gate_q_error = None
    for attempt_idx, requested_states in enumerate(attempts, start=1):
        attempt_dir = out_dir / ("step1_ensemble" if attempt_idx == 1 else f"step1_ensemble_retry{attempt_idx - 1}")
        asam_gen = ASAMEnsembleGenerator(output_dir=str(attempt_dir), device=args.device)
        try:
            ensemble_pdbs, step1_gates = asam_gen.generate_states(
                receptor_pdb=args.pdb,
                k_states=requested_states,
                temperature_k=310.0,
                pocket_center=np.asarray(args.pocket_center, dtype=float),
                pocket_radius=args.pocket_radius,
            )
            actual_k_states = requested_states
            if requested_states != args.k_states:
                print(f"[aSAM] Gate Q failed for {args.k_states} states; completed with {requested_states}-state fallback.")
            break
        except RuntimeError as exc:
            if "Gate Q FAILED" not in str(exc):
                raise
            last_gate_q_error = exc
            print(f"[aSAM] Gate Q failed on attempt {attempt_idx}/{len(attempts)} ({requested_states} states): {exc}")
            if attempt_idx == len(attempts):
                raise
    if ensemble_pdbs is None:
        raise last_gate_q_error or RuntimeError("aSAM ensemble generation failed.")
    all_gate_results.extend(step1_gates)
    print(f"Generated {len(ensemble_pdbs)} breathing states: {[Path(p).name for p in ensemble_pdbs]}")

    # =========================================================================
    # STEP 2: PharmFlow (Continuous Flow Matching 3D Pharmacophores)
    # =========================================================================
    print(f"\n>>> STEP 2: PharmFlow Continuous Flow Matching Sampling")
    pharmflow = PharmFlowSampler(
        ckpt_path=args.ckpt,
        device=args.device,
        pocket_center=np.asarray(args.pocket_center, dtype=float),
        pocket_radius=args.pocket_radius,
    )
    glfo_res = PipelineGates.not_evaluated(
        'G-LFO', 'Family Generalization',
        'Training/test family manifests are not present in this checkout.'
    )
    all_gate_results.append(glfo_res)

    all_state_phars, step2_gates = pharmflow.sample_ensemble_pharmacophores(
        ensemble_pdb_paths=ensemble_pdbs,
        num_phars=8,
        num_steps=50,
        samples_per_state=3,
    )
    all_gate_results.extend(step2_gates)

    # =========================================================================
    # STEP 3: Consensus GMM Merging & Milestone Gate G4
    # =========================================================================
    print(f"\n>>> STEP 3: Consensus GMM Merging Across Breathing States")
    merger = ConsensusPharmacophoreMerger(output_dir=str(out_dir / "step3_consensus"))
    posp_file, blueprint_data, step3_gates = merger.merge_states(
        all_state_phars=all_state_phars,
        n_consensus_features=7,
    )
    all_gate_results.extend(step3_gates)

    # =========================================================================
    # STEP 4: GCPG De Novo Molecular Generation
    # =========================================================================
    print(f"\n>>> STEP 4: GCPG Conditional Molecular Generation & 3D Conformer Building")
    gcpg_gen = GCPGMoleculeGenerator(
        gcpg_python=args.gcpg_python,
        output_dir=str(out_dir / "step4_molecules"),
    )
    generation_kwargs = dict(
        posp_path=posp_file,
        mw_target=400.0,
        logp_target=3.5,
        qed_target=0.65,
    )
    smiles_list, sdf_file = gcpg_gen.generate_molecules(n_mol=args.n_mol, **generation_kwargs)
    retries = 0
    while not smiles_list and retries < 2:
        retries += 1
        retry_n = max(args.n_mol * (2 ** retries), 12)
        print(f"[GCPG] No valid molecules; retrying with n_mol={retry_n} (attempt {retries}/2)...")
        smiles_list, sdf_file = gcpg_gen.generate_molecules(n_mol=retry_n, **generation_kwargs)
    if not smiles_list:
        print("[STOP] GCPG produced no valid molecules after retries. Try increasing --n_mol or rerunning.")
        sys.exit(2)
    if not Path(sdf_file).is_file() or Path(sdf_file).stat().st_size == 0:
        print(f"[STOP] GCPG did not write a dockable SDF: {sdf_file}")
        sys.exit(2)
    print(f"Generated {len(smiles_list)} valid molecules. 3D structures saved to: {Path(sdf_file).name}")

    # =========================================================================
    # STEP 5: Multi-Conformation AutoDock Vina Docking
    # =========================================================================
    print(f"\n>>> STEP 5: Multi-Conformation AutoDock Vina Verification & Scoring")
    docking_engine = MultiStateDockingEngine(
        vina_exe=args.vina_exe,
        meeko_python=meeko_python,
        allow_bad_receptor_residues=args.allow_bad_receptor_residues,
        output_dir=str(out_dir / "step5_docking"),
    )
    top_candidates, step5_gates = docking_engine.run_ensemble_docking(
        sdf_path=sdf_file,
        ensemble_pdb_paths=ensemble_pdbs,
        top_k=8,
        exhaustiveness=args.exhaustiveness,
        pharmacophore_centers=np.array([feature['center'] for feature in blueprint_data['features']]),
    )
    all_gate_results.extend(step5_gates)

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    elapsed = time.time() - start_time
    print_gate_report(all_gate_results)

    print(f"\n{'='*80}")
    print(f"{'RANK':<6} {'CANDIDATE':<14} {'MEAN AFFINITY':<18} {'WORST-CASE':<14} {'VARIANCE'}")
    print(f"{'-'*80}")
    for i, c in enumerate(top_candidates):
        print(f"#{i+1:<5} {c['name']:<14} {c['mean_affinity_kcal_mol']:<6.2f} kcal/mol   {c['worst_case_affinity_kcal_mol']:<6.2f} kcal/mol {c['ensemble_variance']:<6.3f}")
    print(f"{'='*80}")
    print(f"Total Pipeline Runtime: {elapsed:.2f} seconds.")

    # Write executive Markdown report
    report_file = out_dir / "PIPELINE_VALIDATION_REPORT.md"
    generate_markdown_report(
        output_path=report_file,
        target_pdb=args.pdb,
        pocket_center=args.pocket_center,
        pocket_radius=args.pocket_radius,
        pocket_note=args.pocket_note,
        allow_bad_receptor_residues=args.allow_bad_receptor_residues,
        k_states=actual_k_states,
        exhaustiveness=args.exhaustiveness,
        gate_results=all_gate_results,
        top_candidates=top_candidates,
        blueprint_data=blueprint_data,
    )


if __name__ == "__main__":
    main()
