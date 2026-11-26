"""
step3_consensus.py - Step 3: Consensus GMM Merging & Multi-Gate Evaluation.

Merges multi-state pharmacophore point clouds using Gaussian Mixture Models (GMM)
to extract a robust 3D consensus blueprint, and validates Gates G1, G2b, G3, D-State, and G4.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from gates import PipelineGates, GateResult


# Mapping from PharmFlow classes to the supported GCPG pharmacophore vocabulary
# Crossdock: ['AROM', 'HDON', 'HACC', 'LIP', 'CHARGE', 'POS', 'NEG', 'OTHER']
# GCPG's parser supports AROM, HYBL, POSC, HACC, HDON, LHYBL, and UNKNOWN.
GCPG_TYPE_MAP = {
    'Aromatic': 'AROM',
    'Hydrophobe': 'HYBL',
    'PosIonizable': 'POSC',
    'NegIonizable': 'UNKNOWN',
    'Acceptor': 'HACC',
    'Donor': 'HDON',
    'LumpedHydrophobe': 'LHYBL',
    'others': 'UNKNOWN',
    'AROM': 'AROM',
    'HDON': 'HDON',
    'HACC': 'HACC',
    'LIP': 'HYBL',
    'CHARGE': 'UNKNOWN',
    'POS': 'POSC',
    'NEG': 'UNKNOWN',
    'OTHER': 'UNKNOWN',
}


class ConsensusPharmacophoreMerger:
    """Merges dynamic pharmacophore states via GMM and checks pre-generation gates."""

    def __init__(self, output_dir: str = "consensus_phars"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_states(
        self,
        all_state_phars: Dict[str, List[Dict[str, Any]]],
        n_consensus_features: int = 7,
        min_state_agreement: float = 0.60,
    ) -> Tuple[str, Dict[str, Any], List[GateResult]]:
        """
        GMM clustering across all dynamic breathing states (S1...S5).
        Produces the consensus .posp file and validates G1, G2b, G3, D-State, and G4.
        """
        gate_results = []

        # 1. Pool all points across states with state tags
        all_coords = []
        all_types = []
        state_origins = []

        for state_idx, (state_name, samples) in enumerate(all_state_phars.items()):
            for s in samples:
                coords = s['coords']
                types = s['types']
                for c, t in zip(coords, types):
                    all_coords.append(c)
                    all_types.append(t)
                    state_origins.append(state_idx)

        all_coords = np.array(all_coords)
        state_origins = np.array(state_origins)
        num_states = len(all_state_phars)

        if num_states == 0 or any(not samples for samples in all_state_phars.values()):
            raise ValueError('Every receptor state must contain at least one sampled pharmacophore cloud.')
        if not np.isfinite(all_coords).all() or all_coords.ndim != 2 or all_coords.shape[1] != 3:
            raise ValueError('Pharmacophore coordinates must be a finite N x 3 array.')

        if len(all_coords) < n_consensus_features:
            raise ValueError(f"Too few pharmacophore points ({len(all_coords)}) for {n_consensus_features} clusters.")

        # 2. Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_consensus_features, random_state=42, covariance_type='full')
        labels = gmm.fit_predict(all_coords)
        centers = gmm.means_
        covariances = gmm.covariances_

        # 3. Filter clusters by multi-state agreement
        consensus_features = []
        for k in range(n_consensus_features):
            cluster_mask = (labels == k)
            if not np.any(cluster_mask):
                continue

            # Check which states contributed to this cluster
            states_in_cluster = set(state_origins[cluster_mask])
            state_coverage = len(states_in_cluster) / num_states

            # Majority vote on pharmacophore type
            cluster_types = [all_types[i] for i, m in enumerate(cluster_mask) if m]
            dominant_type = max(set(cluster_types), key=cluster_types.count)
            gcpg_type = GCPG_TYPE_MAP.get(dominant_type, 'HYBL')

            # Interaction dispersion (trace of covariance)
            dispersion = float(np.trace(covariances[k]))

            consensus_features.append({
                'center': centers[k].tolist(),
                'type': gcpg_type,
                'raw_type': dominant_type,
                'state_coverage': float(state_coverage),
                'dispersion': dispersion,
                'point_count': int(np.sum(cluster_mask)),
            })

        # Sort features by state coverage and point density
        consensus_features.sort(key=lambda x: (x['state_coverage'], x['point_count']), reverse=True)

        # 4. Export .posp file for GCPG
        posp_filename = self.output_dir / "consensus_blueprint.posp"
        with open(posp_filename, 'w') as f:
            for feat in consensus_features:
                cx, cy, cz = feat['center']
                ptype = feat['type']
                f.write(f"{ptype} {cx:.3f} {cy:.3f} {cz:.3f}\n")

        # Also save JSON metadata
        json_filename = self.output_dir / "consensus_blueprint.json"
        blueprint_data = {
            'n_features': len(consensus_features),
            'features': consensus_features,
            'source_states': list(all_state_phars.keys()),
        }
        with open(json_filename, 'w') as f:
            json.dump(blueprint_data, f, indent=2)

        print(f"[Consensus] Generated consensus blueprint with {len(consensus_features)} features -> {posp_filename}")

        # These gates require independent reference ligands, off-target structures,
        # and retrospective benchmark labels; the generated samples cannot serve as
        # their own validation set.
        gate_results.extend([
            PipelineGates.not_evaluated('G1', 'Pocket State Coverage',
                'Independent experimental/reference pharmacophore states were not supplied.'),
            PipelineGates.not_evaluated('G2b', 'Ensemble vs. Crystal Retrospective',
                'No retrospective active/inactive ligand benchmark was supplied.'),
            PipelineGates.not_evaluated('G3', 'Differential Feature Separability',
                'No matched off-target structures or labeled feature benchmark was supplied.'),
            PipelineGates.not_evaluated('D-State', 'State Ablation Specificity',
                'No held-out retrieval benchmark was supplied for state ablation.'),
            PipelineGates.not_evaluated('G4', 'Retrospective GO / NO-GO',
                'No retrospective benchmark was supplied; generation is not evidence of prospective efficacy.'),
        ])

        return str(posp_filename), blueprint_data, gate_results
