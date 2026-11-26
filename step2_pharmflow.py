"""
step2_pharmflow.py - Step 2: Continuous Flow Matching Pharmacophore Generation.

Uses the trained Flow Matching checkpoint (best_model_epoch_46.ckpt) to predict
the velocity field and sample 3D pharmacophore point clouds for each pocket state.
Supports both full-atom (11 atom types) and CA-only (20 amino acid types) representations.
"""

import os
import pathlib
import sys
import types

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one

from flow_matching import FlowMatching
from equivariant_diffusion.dynamics import EGNNDynamics
from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from gates import PipelineGates, GateResult


class PharmFlowSampler:
    """Samples 3D pharmacophores from protein pocket states using Continuous Flow Matching."""

    def __init__(self, ckpt_path: str, device: str = 'cpu', pocket_center: Optional[np.ndarray] = None, pocket_radius: float = 10.0):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.ckpt_path = Path(ckpt_path)
        self.pocket_center = np.asarray(pocket_center, dtype=float) if pocket_center is not None else None
        self.pocket_radius = float(pocket_radius)
        if self.pocket_center is None or self.pocket_center.shape != (3,):
            raise ValueError('A 3D --pocket_center is required to select the protein binding pocket.')
        if not np.isfinite(self.pocket_center).all() or self.pocket_radius <= 0:
            raise ValueError('Pocket center must be finite and pocket radius must be positive.')
        self.dataset_name = 'crossdock_full'
        self.model, self.residue_nf, self.phar_nf = self._load_model()

    def _load_model(self) -> Tuple[FlowMatching, int, int]:
        print(f"[PharmFlow] Loading checkpoint: {self.ckpt_path}")
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"PharmFlow checkpoint not found: {self.ckpt_path}")
        # Newer Windows Python checkpoints pickle pathlib._local.WindowsPath.
        # Provide that private module only while unpickling; map WindowsPath to a
        # usable local path class on POSIX without mutating pathlib globally.
        old_pathlib_local = sys.modules.get('pathlib._local')
        old_path = getattr(pathlib, '__path__', None)
        local_pathlib = types.ModuleType('pathlib._local')
        local_pathlib.WindowsPath = pathlib.Path
        local_pathlib.PosixPath = pathlib.Path
        pathlib.__path__ = []
        sys.modules['pathlib._local'] = local_pathlib
        try:
            ckpt = torch.load(str(self.ckpt_path), map_location=self.device, weights_only=False)
        finally:
            if old_path is None:
                del pathlib.__path__
            else:
                pathlib.__path__ = old_path
            if old_pathlib_local is None:
                sys.modules.pop('pathlib._local', None)
            else:
                sys.modules['pathlib._local'] = old_pathlib_local

        # Detect dataset from hyperparameters
        hparams = ckpt.get('hyper_parameters', {})
        if 'dataset' in hparams:
            self.dataset_name = hparams['dataset']

        # Handle PyTorch Lightning state_dict vs direct model_state_dict
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            prefix = 'model.dynamics.'
        elif 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
            prefix = 'dynamics.'
        else:
            sd = ckpt
            prefix = ''

        dynamics_sd = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                dynamics_sd[k[len(prefix):]] = v
            elif 'dynamics.' in k:
                dynamics_sd[k.split('dynamics.')[1]] = v

        # Inferred architectural dimensions from checkpoint weights
        residue_nf = dynamics_sd['residue_encoder.0.weight'].shape[1]
        phar_nf = dynamics_sd['phar_encoder.0.weight'].shape[1]
        joint_nf = dynamics_sd['phar_encoder.2.weight'].shape[0] if 'phar_encoder.2.weight' in dynamics_sd else 64
        hidden_nf = dynamics_sd['egnn.embedding.weight'].shape[0]
        n_layers = max([int(k.split('e_block_')[1].split('.')[0]) for k in dynamics_sd.keys() if 'e_block_' in k]) + 1
        attention = any('att_mlp' in k for k in dynamics_sd.keys())

        print(f"[PharmFlow] Inferred architecture: residue_nf={residue_nf}, phar_nf={phar_nf}, hidden_nf={hidden_nf}, n_layers={n_layers}")

        dynamics = EGNNDynamics(
            phar_nf=phar_nf,
            residue_nf=residue_nf,
            n_dims=3,
            joint_nf=joint_nf,
            device=self.device,
            hidden_nf=hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=n_layers,
            attention=attention,
            tanh=False,
            norm_constant=1,
            inv_sublayers=1,
            sin_embedding=False,
            normalization_factor=100,
            aggregation_method='sum',
            edge_cutoff=6.0,
            update_pocket_coords=False,
        )

        missing, unexpected = dynamics.load_state_dict(dynamics_sd, strict=False)
        print(f"[PharmFlow] Loaded dynamics backbone (missing={len(missing)}, unexpected={len(unexpected)})")

        fm = FlowMatching(
            dynamics=dynamics,
            dataset_name=self.dataset_name,
            phar_nf=phar_nf,
            residue_nf=residue_nf,
            n_dims=3,
            lambda_h=1.0,
        ).to(self.device)
        fm.eval()
        return fm, residue_nf, phar_nf

    def build_pocket_dict(self, pdb_path: str, max_nodes: int = 400) -> Dict[str, torch.Tensor]:
        """Extracts pocket coordinates and one-hots matching the model's residue_nf."""
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure('pocket', pdb_path)[0]

        coords = []
        types = []

        if self.residue_nf == 11:
            # Full-atom representation
            atom_encoder = dataset_params['crossdock_full']['atom_encoder']
            for atom in struct.get_atoms():
                elem = atom.element.strip().capitalize() if atom.element else atom.name[0]
                coord = atom.get_coord()
                if elem != 'H' and np.linalg.norm(coord - self.pocket_center) <= self.pocket_radius:
                    coords.append(coord)
                    types.append(atom_encoder.get(elem, atom_encoder.get('others', 10)))
        else:
            # CA-only representation
            aa_encoder = dataset_params[self.dataset_name]['aa_encoder']
            for res in struct.get_residues():
                resname = res.get_resname()
                if 'CA' in res and resname in three_to_one:
                    one_letter = three_to_one[resname]
                    coord = res['CA'].get_coord()
                    if one_letter in aa_encoder and np.linalg.norm(coord - self.pocket_center) <= self.pocket_radius:
                        coords.append(coord)
                        types.append(aa_encoder[one_letter])

        if len(coords) == 0:
            raise ValueError(f"No valid atoms/residues found in {pdb_path}")

        coords = np.asarray(coords)
        types = np.asarray(types)
        nearest = np.argsort(np.linalg.norm(coords - self.pocket_center, axis=1))[:max_nodes]
        coords = coords[nearest]
        types = types[nearest]

        pocket_x = torch.tensor(coords, device=self.device, dtype=FLOAT_TYPE)
        pocket_types = torch.tensor(types, device=self.device, dtype=INT_TYPE)
        pocket_h = F.one_hot(pocket_types, num_classes=self.residue_nf).to(FLOAT_TYPE)

        return {
            'x': pocket_x,
            'one_hot': pocket_h,
            'size': torch.tensor([len(pocket_x)], device=self.device, dtype=INT_TYPE),
            'mask': torch.zeros(len(pocket_x), device=self.device, dtype=INT_TYPE),
        }

    @torch.no_grad()
    def sample_pharmacophores_for_state(
        self,
        pdb_path: str,
        num_phars: int = 8,
        num_steps: int = 50,
        n_samples: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Samples pharmacophores for a single pocket state using Continuous Flow Matching.
        Returns list of sampled clouds: [{'coords': np.ndarray, 'types': List[str], 'classes': np.ndarray}]
        """
        pocket = self.build_pocket_dict(pdb_path)
        phar_decoder = dataset_params[self.dataset_name]['phar_decoder']
        sampled_results = []

        for s in range(n_samples):
            num_nodes = torch.tensor([num_phars], device=self.device, dtype=INT_TYPE)
            x_pred, h_pred, _, _ = self.model.sample_given_pocket(
                pocket=pocket,
                num_nodes_phar=num_nodes,
                num_steps=num_steps,
                device=self.device,
            )

            coords_np = x_pred.cpu().numpy()
            class_indices = h_pred.argmax(dim=-1).cpu().numpy()
            type_names = [phar_decoder[idx] for idx in class_indices]

            sampled_results.append({
                'coords': coords_np,
                'class_indices': class_indices,
                'types': type_names,
            })

        return sampled_results

    def sample_ensemble_pharmacophores(
        self,
        ensemble_pdb_paths: List[str],
        num_phars: int = 8,
        num_steps: int = 50,
        samples_per_state: int = 3,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[GateResult]]:
        """
        Samples pharmacophores across all breathing states (S1...S5).
        Samples pharmacophores. Benchmark comparisons require external labeled evaluation data.
        """
        all_state_phars = {}
        for state_path in ensemble_pdb_paths:
            state_name = Path(state_path).stem
            print(f"[PharmFlow] Sampling flow matching pharmacophores for {state_name}...")
            phars = self.sample_pharmacophores_for_state(
                state_path,
                num_phars=num_phars,
                num_steps=num_steps,
                n_samples=samples_per_state,
            )
            all_state_phars[state_name] = phars

        g2a_res = PipelineGates.not_evaluated(
            'G2a', 'PharmFlow vs. DiffPhar',
            'No held-out labeled benchmark set or paired DiffPhar baseline scores were supplied.'
        )
        return all_state_phars, [g2a_res]
