"""
step1_asam.py - Step 1: aSAM Protein Flexibility & Pocket Conformation Sampling.

Generates or samples K realistic breathing states (S1...S5) of the target receptor,
runs Gate G0 (Input Validation) and Gate Q (Ensemble QC), and extracts pocket representations.
"""

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from gates import PipelineGates, GateResult


class PocketSelect(Select):
    """Filter residues within cutoff of pocket center."""
    def __init__(self, pocket_res_ids: List[int], chain_id: str = 'A'):
        self.pocket_res_ids = set(pocket_res_ids)
        self.chain_id = chain_id

    def accept_residue(self, residue):
        return residue.get_parent().id == self.chain_id and residue.id[1] in self.pocket_res_ids


class ASAMEnsembleGenerator:
    """Manages protein flexibility and ensemble generation via aSAM."""

    def __init__(self, output_dir: str = "ensemble_states", device: str = "auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()

    def _resolve_device(self) -> str:
        requested = (self.device or "auto").lower()
        if requested not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"aSAM device must be auto, cuda, or cpu; got {self.device!r}")
        if requested == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError:
            return "cpu"
        if not torch.cuda.is_available():
            print("[aSAM] CUDA unavailable; using CPU.")
            return "cpu"
        return "cuda"

    def generate_states(
        self,
        receptor_pdb: str,
        k_states: int = 5,
        temperature_k: float = 310.0,
        pocket_center: Optional[np.ndarray] = None,
        pocket_radius: float = 10.0,
    ) -> Tuple[List[str], List[GateResult]]:
        """
        Generate K breathing conformations of the receptor.
        Validates through Gate G0 and Gate Q.
        """
        gate_results = []
        receptor_path = Path(receptor_pdb)
        asam_device = self._resolve_device()
        if k_states < 1:
            raise ValueError('k_states must be at least 1')

        # --- Gate G0: Validate Input ---
        g0_res = PipelineGates.evaluate_g0(str(receptor_path))
        gate_results.append(g0_res)
        if g0_res.status == 'STOP':
            raise ValueError(f"Gate G0 FAILED: {g0_res.message}")

        # Generate a real model ensemble. Synthetic coordinate perturbations are
        # deliberately not used as substitutes for aSAM inference.
        sam_root = Path(__file__).parent / 'asam_engine'
        sam_script = sam_root / 'src' / 'sam' / 'scripts' / 'generate_ensemble.py'
        if not sam_script.is_file():
            raise FileNotFoundError(f"Bundled aSAM inference script not found: {sam_script}")
        prefix = self.output_dir / f"{receptor_path.stem}_asam"
        runner = self.output_dir / '_run_asam_compat.py'
        runner.write_text(textwrap.dedent(f'''\
            import runpy
            import sys
            import types
            import torch
            if not hasattr(torch, 'accelerator'):
                # Diffusers 0.41 queries this API at import time; a CPU fallback
                # keeps it compatible with the project's Torch 2.1 environment.
                torch.accelerator = types.SimpleNamespace(is_available=lambda: False)
            script = {str(sam_script.resolve())!r}
            sys.argv = [script, *sys.argv[1:]]
            runpy.run_path(script, run_name='__main__')
        '''))
        env = os.environ.copy()
        src_path = str(sam_root / 'src')
        env['PYTHONPATH'] = src_path + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        batch_size = max(1, min(k_states, 4))
        print(f"[aSAM] Inference device: {asam_device}; batch size: {batch_size}")
        cmd = [sys.executable, str(runner), '--config', 'atlas', '--init', str(receptor_path.resolve()),
               '--out_path', str(prefix), '--out_fmt', 'xtc', '--n_samples', str(k_states),
               '--batch_size', str(batch_size),
               '--device', asam_device, '--no_minimize', '--no_download']
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode and asam_device == 'cuda':
            print("[aSAM] CUDA inference failed; retrying on CPU ...")
            for suffix in ('.traj.xtc', '.top.pdb'):
                partial = Path(str(prefix) + suffix)
                if partial.exists():
                    partial.unlink()
            cmd[cmd.index('--device') + 1] = 'cpu'
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode:
            raise RuntimeError(
                'aSAM inference failed. Install asam_engine dependencies and provide model weights.\n'
                f'Command: {cmd}\n{result.stdout}\n{result.stderr}'
            )
        try:
            import mdtraj
        except ImportError as exc:
            raise RuntimeError('mdtraj is required to export aSAM conformations as PDB states.') from exc
        traj_path = prefix.with_name(prefix.name + '.traj.xtc')
        top_path = prefix.with_name(prefix.name + '.top.pdb')
        trajectory = mdtraj.load(str(traj_path), top=str(top_path))
        if trajectory.n_frames < k_states:
            raise RuntimeError(f'aSAM returned {trajectory.n_frames} frames; requested {k_states}.')
        reference = mdtraj.load(str(receptor_path.resolve()))
        generated_ca_atoms = [atom for atom in trajectory.topology.atoms if atom.name == 'CA']
        reference_ca_atoms = [atom for atom in reference.topology.atoms if atom.name == 'CA']
        generated_ca = [atom.index for atom in generated_ca_atoms]
        reference_ca = [atom.index for atom in reference_ca_atoms]
        generated_sequence = [atom.residue.name for atom in generated_ca_atoms]
        reference_sequence = [atom.residue.name for atom in reference_ca_atoms]
        if not generated_ca or len(generated_ca) != len(reference_ca) or generated_sequence != reference_sequence:
            raise RuntimeError('Could not establish matching C-alpha atoms for aSAM structure alignment.')
        trajectory.superpose(reference, frame=0, atom_indices=generated_ca, ref_atom_indices=reference_ca)
        # aSAM can rebuild atoms missing from an experimental crystal model
        # (for example, an unresolved lysine side chain). Use its returned
        # topology as the output structure so every generated atom has a slot
        # when writing the sampled coordinates, while retaining the existing
        # sequence check against the uploaded receptor above.
        reference_structure = self.parser.get_structure('reference', str(top_path.resolve()))
        reference_residues = list(reference_structure[0].get_residues())
        generated_residues = list(trajectory.topology.residues)
        if len(reference_residues) != len(generated_residues):
            raise RuntimeError('aSAM and input PDB have different residue counts; cannot map generated atoms safely.')
        for source_residue, generated_residue in zip(reference_residues, generated_residues):
            if source_residue.get_resname() != generated_residue.name:
                raise RuntimeError(
                    f'aSAM residue sequence mismatch at {source_residue.id}: '
                    f'{source_residue.get_resname()} vs {generated_residue.name}'
                )
        ensemble_files = []
        for idx in range(k_states):
            state_file = self.output_dir / f"{receptor_path.stem}_state_{idx + 1}.pdb"
            for source_residue, generated_residue in zip(reference_residues, generated_residues):
                for generated_atom in generated_residue.atoms:
                    atom_name = generated_atom.name
                    if atom_name not in source_residue:
                        raise RuntimeError(f'aSAM atom {generated_residue.name}:{atom_name} is absent from the input PDB.')
                    source_residue[atom_name].set_coord(trajectory.xyz[idx, generated_atom.index] * 10.0)
            self.io.set_structure(reference_structure)
            self.io.save(str(state_file))
            ensemble_files.append(str(state_file))

        # --- Gate Q: Ensemble QC ---
        q_res = PipelineGates.evaluate_q(ensemble_files, max_pocket_rmsd=2.0, min_pass_rate=0.80)
        gate_results.append(q_res)
        if q_res.status == 'STOP':
            raise RuntimeError(f"Gate Q FAILED: {q_res.message}")

        return ensemble_files, gate_results
