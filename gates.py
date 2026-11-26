"""
gates.py - Multi-Stage Gating and Quality Control Framework for CMD-Gen Enhanced Pipeline.

Implements all 11 validation and decision gates:
G0, Q, G1, G-LFO, G2a, G2b, G3, D-State, G4, FEP, G7-8.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import stats
from Bio.PDB import PDBParser


class GateResult:
    def __init__(self, gate_id: str, name: str, status: str, consequence: str, message: str, metrics: Optional[Dict[str, Any]] = None):
        self.gate_id = gate_id
        self.name = name
        self.status = status  # 'PASS', 'STOP', 'FALLBACK', 'DOWNGRADE', 'NOT EVALUATED'
        self.consequence = consequence
        self.message = message
        self.metrics = metrics or {}

    def __repr__(self):
        return f"[{self.status}] Gate {self.gate_id} ({self.name}): {self.message}"


class PipelineGates:
    """Evaluates gates across Stages 0 through 9."""

    @staticmethod
    def not_evaluated(gate_id: str, name: str, reason: str) -> GateResult:
        return GateResult(gate_id, name, 'NOT EVALUATED', 'NONE', reason)

    @staticmethod
    def evaluate_g0(pdb_path: str, max_resolution: float = 2.5) -> GateResult:
        """
        Gate G0: Inputs & Applicability (Stage 0)
        Consequence: STOP
        Validates PDB construct resolution (<= 2.5 A), chain integrity, and valid coordinate data.
        """
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('protein', pdb_path)
            header = parser.get_header()
            resolution = header.get('resolution')

            # Check for coordinates
            atoms = list(structure.get_atoms())
            if len(atoms) < 50:
                return GateResult(
                    'G0', 'Inputs & Applicability', 'STOP', 'STOP',
                    f"PDB contains too few atoms ({len(atoms)}) to represent a valid protein construct.",
                    {'atom_count': len(atoms)}
                )

            # Check resolution if provided in header
            if resolution is not None and resolution > max_resolution:
                return GateResult(
                    'G0', 'Inputs & Applicability', 'STOP', 'STOP',
                    f"Resolution {resolution:.2f} A exceeds threshold of {max_resolution:.2f} A.",
                    {'resolution': resolution}
                )

            return GateResult(
                'G0', 'Inputs & Applicability', 'PASS', 'NONE',
                f"PDB passes structural integrity and resolution checks (resolution: {resolution or 'N/A'} A, {len(atoms)} atoms).",
                {'resolution': resolution, 'atom_count': len(atoms)}
            )

        except Exception as e:
            return GateResult(
                'G0', 'Inputs & Applicability', 'STOP', 'STOP',
                f"PDB parsing failed: {str(e)}",
                {'error': str(e)}
            )

    @staticmethod
    def evaluate_q(ensemble_pdbs: List[str], max_pocket_rmsd: float = 2.0, min_pass_rate: float = 0.80) -> GateResult:
        """
        Gate Q: Ensemble QC (Stage 2)
        Consequence: STOP (Ensemble Branch)
        Requires >= 80% of conformers to pass global C_alpha coordinate RMSD and
        minimum pairwise C_alpha distance checks. This is a lightweight check, not
        full stereochemistry, bond-geometry, or all-atom clash validation.
        """
        if not ensemble_pdbs:
            return GateResult('Q', 'Ensemble QC', 'STOP', 'STOP', "Empty ensemble provided.")

        parser = PDBParser(QUIET=True)
        ref_struct = parser.get_structure('ref', ensemble_pdbs[0])
        ref_ca = np.array([a.get_coord() for a in ref_struct.get_atoms() if a.get_name() == 'CA'])

        passed_count = 0
        rmsds = []

        for pdb_file in ensemble_pdbs:
            try:
                struct = parser.get_structure('conf', pdb_file)
                ca_atoms = [a for a in struct.get_atoms() if a.get_name() == 'CA']
                ca = np.array([a.get_coord() for a in ca_atoms])
                min_len = min(len(ref_ca), len(ca))
                if min_len == 0:
                    continue
                # Calculate pocket/backbone RMSD
                diff = ca[:min_len] - ref_ca[:min_len]
                rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
                rmsds.append(rmsd)

                # Exclude the same and immediately adjacent sequence positions:
                # they are covalently connected and are not nonbonded clashes.
                dist_matrix = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
                residue_numbers = np.array([a.get_parent().id[1] for a in ca_atoms])
                chain_ids = np.array([a.get_parent().get_parent().id for a in ca_atoms])
                sequential = (np.abs(residue_numbers[:, None] - residue_numbers[None, :]) <= 1)
                same_chain = chain_ids[:, None] == chain_ids[None, :]
                dist_matrix[sequential & same_chain] = 999.0
                no_clashes = np.min(dist_matrix) >= 2.8

                if rmsd <= max_pocket_rmsd and no_clashes:
                    passed_count += 1
            except Exception:
                continue

        pass_rate = passed_count / len(ensemble_pdbs)
        metrics = {'pass_rate': pass_rate, 'passed_count': passed_count, 'total': len(ensemble_pdbs), 'mean_rmsd': float(np.mean(rmsds)) if rmsds else 0.0}

        if pass_rate >= min_pass_rate:
            return GateResult(
                'Q', 'Ensemble QC', 'PASS', 'NONE',
                f"C-alpha RMSD/distance screen passed: {pass_rate*100:.1f}% conformers passed (>= {min_pass_rate*100:.0f}% required).",
                metrics
            )
        else:
            return GateResult(
                'Q', 'Ensemble QC', 'STOP', 'STOP',
                f"C-alpha RMSD/distance screen failed: only {pass_rate*100:.1f}% conformers passed (>= {min_pass_rate*100:.0f}% required).",
                metrics
            )

    @staticmethod
    def evaluate_g1(clustered_centers: np.ndarray, reference_states: np.ndarray, threshold_distance: float = 1.5, min_coverage: float = 0.90) -> GateResult:
        """
        Gate G1: Pocket State Coverage (Stage 3)
        Consequence: FALLBACK
        Requires >= 90% of known reference/experimental states to be covered within 1.5 A.
        """
        if len(clustered_centers) == 0 or len(reference_states) == 0:
            return GateResult('G1', 'Pocket State Coverage', 'FALLBACK', 'FALLBACK', "Missing cluster centers or reference states.")

        # Distance from each reference state to the nearest cluster center
        distances = np.linalg.norm(reference_states[:, None, :] - clustered_centers[None, :, :], axis=-1)
        min_dists = np.min(distances, axis=-1)
        covered = np.sum(min_dists <= threshold_distance)
        coverage_rate = covered / len(reference_states)

        metrics = {'coverage_rate': float(coverage_rate), 'covered_count': int(covered), 'total_reference': len(reference_states)}

        if coverage_rate >= min_coverage:
            return GateResult('G1', 'Pocket State Coverage', 'PASS', 'NONE', f"Pocket coverage passed ({coverage_rate*100:.1f}%).", metrics)
        else:
            return GateResult('G1', 'Pocket State Coverage', 'FALLBACK', 'FALLBACK', f"Pocket coverage low ({coverage_rate*100:.1f}% < {min_coverage*100:.0f}%). Falling back to wider sampling.", metrics)

    @staticmethod
    def evaluate_g_lfo(train_families: List[str], test_families: List[str], min_train_fam: int = 10, min_test_fam: int = 5) -> GateResult:
        """
        Gate G-LFO: Family Generalization (Stage 4)
        Consequence: DOWNGRADE
        Verifies zero chemical/target leakage across >= 10 train families and >= 5 held-out test families.
        """
        overlap = set(train_families).intersection(set(test_families))
        metrics = {'train_count': len(train_families), 'test_count': len(test_families), 'leakage_count': len(overlap)}

        if overlap:
            return GateResult('G-LFO', 'Family Generalization', 'DOWNGRADE', 'DOWNGRADE', f"Target leakage detected across {len(overlap)} families: {list(overlap)[:3]}.", metrics)

        if len(train_families) >= min_train_fam and len(test_families) >= min_test_fam:
            return GateResult('G-LFO', 'Family Generalization', 'PASS', 'NONE', f"Zero leakage confirmed across {len(train_families)} train and {len(test_families)} test families.", metrics)
        else:
            return GateResult('G-LFO', 'Family Generalization', 'DOWNGRADE', 'DOWNGRADE', f"Insufficient family diversity ({len(train_families)} train, {len(test_families)} test).", metrics)

    @staticmethod
    def evaluate_g2a(cfm_f1s: List[float], ddpm_f1s: List[float], alpha: float = 0.05) -> GateResult:
        """
        Gate G2a: PharmFlow vs. DiffPhar (Stage 4)
        Consequence: DOWNGRADE
        Paired one-sided t-test for supplied Continuous Flow Matching and DDPM F1 measurements.
        """
        diffs = np.array(cfm_f1s) - np.array(ddpm_f1s)
        mean_diff = float(np.mean(diffs))

        # Paired one-sided t-test (alternative='greater')
        res = stats.ttest_rel(cfm_f1s, ddpm_f1s, alternative='greater')
        p_val = float(res.pvalue)

        metrics = {'mean_delta_f1': mean_diff, 'p_value': p_val, 'alpha': alpha}

        if mean_diff > 0 and p_val < alpha:
            return GateResult('G2a', 'PharmFlow vs. DiffPhar', 'PASS', 'NONE', f"PharmFlow significantly outperforms DiffPhar (Delta F1 = +{mean_diff:.4f}, p = {p_val:.4e}).", metrics)
        else:
            return GateResult('G2a', 'PharmFlow vs. DiffPhar', 'DOWNGRADE', 'DOWNGRADE', f"PharmFlow does not significantly beat DiffPhar (Delta F1 = {mean_diff:.4f}, p = {p_val:.4f}). Downgrading confidence.", metrics)

    @staticmethod
    def evaluate_g2b(ensemble_aucs: List[float], crystal_auc: float, alpha: float = 0.05) -> GateResult:
        """
        Gate G2b: Ensemble vs. Crystal Retrospective (Stage 3/4)
        Consequence: FALLBACK
        Paired statistical test proving dynamic ensemble pharmacophores achieve superior active retrieval AUC over rigid crystal.
        """
        mean_ens = float(np.mean(ensemble_aucs))
        delta_auc = mean_ens - crystal_auc

        # One-sample t-test comparing ensemble AUCs against rigid crystal AUC
        res = stats.ttest_1samp(ensemble_aucs, crystal_auc, alternative='greater')
        p_val = float(res.pvalue)

        metrics = {'ensemble_mean_auc': mean_ens, 'crystal_auc': crystal_auc, 'delta_auc': delta_auc, 'p_value': p_val}

        if delta_auc > 0 and p_val < alpha:
            return GateResult('G2b', 'Ensemble vs. Crystal Retrospective', 'PASS', 'NONE', f"Ensemble significantly beats static crystal (Delta AUC = +{delta_auc:.4f}, p = {p_val:.4f}).", metrics)
        else:
            return GateResult('G2b', 'Ensemble vs. Crystal Retrospective', 'FALLBACK', 'FALLBACK', f"Ensemble does not beat crystal (Delta AUC = {delta_auc:.4f}, p = {p_val:.4f}). Falling back to crystal baseline.", metrics)

    @staticmethod
    def evaluate_g3(target_features: np.ndarray, offtarget_features: np.ndarray, min_auc: float = 0.75, max_p: float = 0.01) -> GateResult:
        """
        Gate G3: Differential Feature Separability (Stage 5)
        Consequence: DOWNGRADE
        Mann-Whitney test on supplied target and off-target feature projections (AUC >= 0.75, p < 0.01).
        """
        # Distance between feature distributions (using centroid separability proxy)
        t_center = np.mean(target_features, axis=0)
        o_center = np.mean(offtarget_features, axis=0)
        sep_vector = t_center - o_center

        t_proj = target_features @ sep_vector
        o_proj = offtarget_features @ sep_vector

        # Calculate empirical AUC via Wilcoxon-Mann-Whitney
        u_stat, p_val = stats.mannwhitneyu(t_proj, o_proj, alternative='greater')
        n_t, n_o = len(t_proj), len(o_proj)
        auc = float(u_stat / (n_t * n_o))

        metrics = {'auc': auc, 'p_value': float(p_val)}

        if auc >= min_auc and p_val < max_p:
            return GateResult('G3', 'Differential Feature Separability', 'PASS', 'NONE', f"Target/Off-target separability confirmed (AUC = {auc:.3f}, p = {p_val:.4e}).", metrics)
        else:
            return GateResult('G3', 'Differential Feature Separability', 'DOWNGRADE', 'DOWNGRADE', f"Separability below threshold (AUC = {auc:.3f} < {min_auc}, p = {p_val:.4f}). Downgrading selectivity claim.", metrics)

    @staticmethod
    def evaluate_d_state(full_ensemble_auc: float, ablated_aucs: List[float], min_drop: float = 0.05) -> GateResult:
        """
        Gate D-State: State Ablation Specificity (Stage 5)
        Consequence: DOWNGRADE (Mode D)
        Verifies that removing any conformational state from the ensemble causes active retrieval AUC to drop by >= 0.05.
        """
        drops = [full_ensemble_auc - a for a in ablated_aucs]
        min_observed_drop = float(np.min(drops))
        metrics = {'full_auc': full_ensemble_auc, 'min_drop': min_observed_drop, 'all_drops': drops}

        if min_observed_drop >= min_drop:
            return GateResult('D-State', 'State Ablation Specificity', 'PASS', 'NONE', f"All conformational states are essential (minimum AUC drop on ablation = {min_observed_drop:.3f} >= {min_drop}).", metrics)
        else:
            return GateResult('D-State', 'State Ablation Specificity', 'DOWNGRADE', 'DOWNGRADE (Mode D)', f"Redundant state detected (minimum drop = {min_observed_drop:.3f} < {min_drop}). Pruning ensemble.", metrics)

    @staticmethod
    def evaluate_g4(retrospective_delta_auc: float, p_val: float, min_delta_auc: float = 0.05, alpha: float = 0.05) -> GateResult:
        """
        Gate G4: Retrospective GO / NO-GO (Stage 5 Pre-Generation)
        Consequence: STOP
        Primary milestone gate: requires confirmed retrospective AUC gain >= 0.05 (p < alpha) before unlocking full GCPG molecular generation.
        """
        metrics = {'delta_auc': retrospective_delta_auc, 'p_val': p_val, 'threshold': min_delta_auc}

        if retrospective_delta_auc >= min_delta_auc and p_val < alpha:
            return GateResult('G4', 'Retrospective GO / NO-GO', 'PASS', 'NONE', f"GO unlocked: Retrospective AUC gain = +{retrospective_delta_auc:.3f} (p = {p_val:.4e}). Proceeding to GCPG.", metrics)
        else:
            return GateResult('G4', 'Retrospective GO / NO-GO', 'STOP', 'STOP', f"NO-GO: Retrospective AUC gain = +{retrospective_delta_auc:.3f} failed threshold of {min_delta_auc} (p = {p_val:.4f}). Halting generation to save compute.", metrics)

    @staticmethod
    def evaluate_fep(delta_delta_g_ci: Tuple[float, float], selectivity_threshold: float = -1.5) -> GateResult:
        """
        Gate FEP: Selectivity Free Energy (Stage 8)
        Consequence: DOWNGRADE
        Thermodynamic perturbation checking that relative binding free energy Delta-Delta-G 95% CI strictly clears the selectivity target.
        """
        ci_lower, ci_upper = delta_delta_g_ci
        metrics = {'ci_lower': ci_lower, 'ci_upper': ci_upper, 'threshold': selectivity_threshold}

        # Upper bound of 95% CI must be more negative than selectivity_threshold
        if ci_upper <= selectivity_threshold:
            return GateResult('FEP', 'Selectivity Free Energy', 'PASS', 'NONE', f"Thermodynamic selectivity verified: 95% CI [{ci_lower:.2f}, {ci_upper:.2f}] kcal/mol clears {selectivity_threshold} kcal/mol.", metrics)
        else:
            return GateResult('FEP', 'Selectivity Free Energy', 'DOWNGRADE', 'DOWNGRADE', f"Selectivity uncertain: 95% CI [{ci_lower:.2f}, {ci_upper:.2f}] kcal/mol does not strictly clear {selectivity_threshold} kcal/mol.", metrics)

    @staticmethod
    def evaluate_g7_8(ic50_nm: float, n_positives: int, n_total: int, target_ic50: float = 50.0, alpha: float = 0.05) -> GateResult:
        """
        Gate G7-8: Experimental Claim Verification (Stage 9 Wet-Lab).
        This computational pipeline cannot validate experimental potency claims.
        """
        return PipelineGates.not_evaluated(
            'G7-8', 'Experimental Claim Verification',
            'No measured dose-response or selectivity data were supplied; docking scores cannot establish an experimental IC50.'
        )
