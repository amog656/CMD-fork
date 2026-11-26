"""
step5_docking.py - Multi-State AutoDock Vina Docking.

Docks GCPG-generated candidate molecules across supplied receptor conformations (S1...SK).
Vina scores are computational ranking signals. Free-energy and wet-lab potency/selectivity
gates remain unevaluated without independent data.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from gates import PipelineGates, GateResult


class MultiStateDockingEngine:
    """Performs ensemble docking with AutoDock Vina and evaluates binding thermodynamics."""

    def __init__(
        self,
        vina_exe: str = "vina.exe",
        meeko_python: str = sys.executable,
        allow_bad_receptor_residues: bool = False,
        output_dir: str = "docking_results",
    ):
        self.vina_exe = Path(vina_exe).resolve()
        self.meeko_python = Path(meeko_python).resolve()
        self.allow_bad_receptor_residues = allow_bad_receptor_residues
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _meeko_command(self, name: str) -> List[str]:
        """Resolve Meeko's console command from PATH or the active Python env."""
        candidates = []
        if os.name == 'nt':
            candidates.append(self.meeko_python.parent / 'Scripts' / f'{name}.exe')
        else:
            candidates.append(self.meeko_python.parent / 'bin' / f'{name}.py')
            candidates.append(self.meeko_python.parent / 'bin' / name)
        for candidate in candidates:
            if candidate.is_file():
                return [str(candidate)]
        found = shutil.which(name)
        if found:
            return [found]
        raise FileNotFoundError(f'{name} not found on PATH or beside {self.meeko_python}')

    def prepare_ligand_pdbqt(self, mol: Chem.Mol, out_pdbqt: Path) -> bool:
        """Use Meeko's supported parameterizer to produce valid flexible PDBQT."""
        try:
            if mol.GetNumConformers() == 0:
                return False
            sdf_path = out_pdbqt.with_suffix('.sdf')
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            result = subprocess.run(
                [*self._meeko_command('mk_prepare_ligand'), '-i', str(sdf_path), '-o', str(out_pdbqt)],
                capture_output=True, text=True,
            )
            sdf_path.unlink(missing_ok=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            return out_pdbqt.is_file() and out_pdbqt.stat().st_size > 0
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f'Meeko ligand preparation failed for {out_pdbqt.name}: {exc}') from exc

    def _receptor_without_hydrogens(self, receptor_pdb: Path) -> Path:
        """Meeko 0.8 template matching is fragile with explicit hydrogens from aSAM."""
        sanitized = self.output_dir / f"{receptor_pdb.stem}_noH.pdb"
        lines = receptor_pdb.read_text(encoding='utf-8', errors='replace').splitlines()
        kept = []
        removed = 0
        for line in lines:
            if line.startswith(('ATOM  ', 'HETATM')) and line[76:78].strip().upper() in ('H', 'D'):
                removed += 1
                continue
            kept.append(line)
        if removed:
            print(f"[Docking] Removed {removed} hydrogens from {receptor_pdb.name} before Meeko receptor preparation.")
        sanitized.write_text("\n".join(kept) + "\n", encoding='utf-8')
        return sanitized

    def prepare_receptor_pdbqt(self, receptor_pdb: Path, out_pdbqt: Path) -> bool:
        """Prepare receptor with Meeko; raw PDB-to-PDBQT string formatting is invalid."""
        meeko_pdb = self._receptor_without_hydrogens(receptor_pdb)
        receptor_command = self._meeko_command('mk_prepare_receptor')
        help_result = subprocess.run(receptor_command + ['--help'], capture_output=True, text=True)
        help_text = help_result.stdout + help_result.stderr
        legacy_cli = '--read_pdb' in help_text
        if legacy_cli:
            cmd = [*receptor_command, '--read_pdb', str(meeko_pdb), '--write_pdbqt', str(out_pdbqt)]
        else:
            # Meeko 0.7+ takes an output basename and appends .pdbqt itself.
            out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
            output_base = out_pdbqt.with_suffix('')
            cmd = [*receptor_command, '--pdb', str(meeko_pdb), '-o', str(output_base), '--skip_gpf']
        # aSAM writes generic HIS labels while retaining whichever ring hydrogens
        # it sampled. Meeko cannot resolve a generic HIS when the observed atoms
        # tie between HID/HIE templates, so pass the tautomer implied by those Hs.
        try:
            lines = receptor_pdb.read_text(encoding='utf-8', errors='replace').splitlines()
            histidines: Dict[Tuple[str, str], set] = {}
            for line in lines:
                if line.startswith(('ATOM  ', 'HETATM')) and line[17:20].strip() == 'HIS':
                    key = (line[21:22].strip(), line[22:27].strip())
                    if line[76:78].strip().upper() in ('H', 'D'):
                        histidines.setdefault(key, set()).add(line[12:16].strip())
                    else:
                        histidines.setdefault(key, set())
            templates = []
            for (chain, residue), atoms in histidines.items():
                ring_h = atoms & {'HD1', 'HE2'}
                template = 'HIP' if ring_h == {'HD1', 'HE2'} else ('HID' if 'HD1' in ring_h else 'HIE')
                templates.append(f"{chain}:{residue}={template}")
            if templates and legacy_cli:
                cmd.extend(['-n', ','.join(templates)])
        except OSError as exc:
            raise RuntimeError(f'Could not inspect receptor histidines in {receptor_pdb}: {exc}') from exc
        # Older Meeko accepts explicit alternate-location and residue flags.
        if legacy_cli:
            cmd.extend(['--default_altloc', 'A'])
            if self.allow_bad_receptor_residues:
                cmd.append('--delete_bad_res')
        elif self.allow_bad_receptor_residues:
            # Meeko 0.7+ handles residue-template failures differently; keep
            # the default strict behavior rather than passing an unknown flag.
            pass
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
        )
        if not legacy_cli:
            generated = out_pdbqt.with_suffix('')
            generated_pdbqt = generated.with_suffix('.pdbqt')
            if generated_pdbqt.is_file() and generated_pdbqt != out_pdbqt:
                generated_pdbqt.replace(out_pdbqt)
        if result.returncode or not out_pdbqt.is_file():
            raise RuntimeError(f'Meeko receptor preparation failed for {receptor_pdb}: {result.stderr.strip()}')
        return True

    def get_pocket_box(self, receptor_pdb: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates center of mass and bounding box dimensions for docking."""
        coords = []
        with open(receptor_pdb, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        coords = np.asarray(coords, dtype=float)
        if coords.size == 0:
            raise ValueError(f'No alpha-carbon coordinates found in {receptor_pdb}')
        center = np.mean(coords, axis=0)
        size = np.array([22.0, 22.0, 22.0])  # standard pocket docking box
        return center, size

    def dock_molecule_single_state(
        self,
        ligand_pdbqt: Path,
        receptor_pdbqt: Path,
        center: np.ndarray,
        size: np.ndarray,
        out_docked_pdbqt: Path,
        exhaustiveness: int = 8,
    ) -> Optional[float]:
        """Runs AutoDock Vina for a single ligand-receptor pair."""
        cmd = [
            str(self.vina_exe),
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--center_x", f"{center[0]:.3f}",
            "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{size[0]:.1f}",
            "--size_y", f"{size[1]:.1f}",
            "--size_z", f"{size[2]:.1f}",
            "--exhaustiveness", str(exhaustiveness),
            "--out", str(out_docked_pdbqt),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(f'Vina failed ({res.returncode}): {res.stderr.strip()}')

        # Vina 1.2.7 can write a progress bar to stdout instead of the affinity
        # table on larger receptors. Parse the normalized stdout first, then the
        # authoritative output PDBQT.
        for line in res.stdout.replace('\r', '\n').splitlines():
            # Matches Vina table output (e.g., "   1      -8.4      0.000      0.000")
            match = re.match(r"^\s*1\s+(-?\d+\.\d+)", line)
            if match:
                return float(match.group(1))

        if out_docked_pdbqt.is_file():
            for line in out_docked_pdbqt.read_text(encoding='utf-8', errors='replace').splitlines():
                match = re.match(r'^REMARK VINA RESULT:\s+(-?\d+\.\d+)', line)
                if match:
                    return float(match.group(1))

        raise RuntimeError(
            f'Vina returned no affinity for {ligand_pdbqt.name}; '
            f'stdout tail: {res.stdout[-1000:]!r}; stderr tail: {res.stderr[-1000:]!r}'
        )

    def run_ensemble_docking(
        self,
        sdf_path: str,
        ensemble_pdb_paths: List[str],
        top_k: int = 10,
        exhaustiveness: int = 8,
        pharmacophore_centers: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], List[GateResult]]:
        """
        Docks all molecules from SDF across all supplied receptor states.
        Independent free-energy and experimental gates are marked unevaluated.
        """
        gate_results = []
        sdf = Path(sdf_path)
        if not sdf.is_file() or sdf.stat().st_size == 0:
            raise RuntimeError(f'No dockable ligand SDF was produced: {sdf_path}')
        if not ensemble_pdb_paths:
            raise ValueError('At least one receptor conformation is required for docking.')
        suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
        mols = [m for m in suppl if m is not None]

        if not mols:
            raise RuntimeError(f'No valid molecules could be read from {sdf_path}')

        print(f"[Docking] Preparing {len(mols)} candidate molecules and {len(ensemble_pdb_paths)} receptor states...")

        # Prepare receptor PDBQTs
        receptor_pdbqts = []
        centers = []
        for state_pdb in ensemble_pdb_paths:
            p_in = Path(state_pdb)
            p_out = self.output_dir / f"{p_in.stem}.pdbqt"
            self.prepare_receptor_pdbqt(p_in, p_out)
            receptor_pdbqts.append(p_out)
            c, s = self.get_pocket_box(p_in)
            if pharmacophore_centers is not None and len(pharmacophore_centers):
                anchors = np.asarray(pharmacophore_centers, dtype=float)
                c = anchors.mean(axis=0)
                extent = np.ptp(anchors, axis=0) + 12.0
                s = np.maximum(extent, 20.0)
            centers.append((c, s))

        results = []

        for mol_idx, mol in enumerate(mols):
            name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'mol_{mol_idx+1}'
            smiles = mol.GetProp('SMILES') if mol.HasProp('SMILES') else Chem.MolToSmiles(mol)

            ligand_pdbqt = self.output_dir / f"{name}.pdbqt"
            if not self.prepare_ligand_pdbqt(mol, ligand_pdbqt):
                continue

            state_affinities = []
            for state_idx, (rec_pdbqt, (c, s)) in enumerate(zip(receptor_pdbqts, centers)):
                out_docked = self.output_dir / f"{name}_state_{state_idx+1}_out.pdbqt"
                affinity = self.dock_molecule_single_state(
                    ligand_pdbqt=ligand_pdbqt,
                    receptor_pdbqt=rec_pdbqt,
                    center=c,
                    size=s,
                    out_docked_pdbqt=out_docked,
                    exhaustiveness=exhaustiveness,
                )
                state_affinities.append(affinity)

            mean_aff = float(np.mean(state_affinities))
            min_aff = float(np.max(state_affinities))  # worst-case affinity (least negative)
            variance = float(np.var(state_affinities))

            results.append({
                'name': name,
                'smiles': smiles,
                'state_affinities': state_affinities,
                'mean_affinity_kcal_mol': round(mean_aff, 2),
                'worst_case_affinity_kcal_mol': round(min_aff, 2),
                'ensemble_variance': round(variance, 3),
            })

        # Rank candidates by mean binding affinity (lowest kcal/mol is strongest)
        results.sort(key=lambda x: x['mean_affinity_kcal_mol'])
        top_candidates = results[:top_k]

        gate_results.extend([
            PipelineGates.not_evaluated('FEP', 'Selectivity Free Energy',
                'Vina docking scores are not free-energy perturbation calculations; no off-target FEP data were supplied.'),
            PipelineGates.not_evaluated('G7-8', 'Experimental Claim Verification',
                'No measured dose-response or selectivity data were supplied.'),
        ])

        return top_candidates, gate_results
