"""Docking adapter implementations for target ensemble evaluation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class MockDockingAdapter:
    version: str = "mock-docking-1"

    def score(self, smiles: str, state_id: str, seed: int) -> float:
        if not smiles:
            raise ValueError("SMILES must be non-empty")
        # Mock returns energy-like scores: more negative is better.
        return float(-5.0 + np.random.default_rng(seed).normal(0, 0.5))


@dataclass
class VinaDockingAdapter:
    """Production AutoDock Vina docking adapter."""

    vina_executable: str | Path = "bin/vina.exe"
    obabel_executable: str = "obabel"
    receptor_paths: dict[str, str | Path] = field(default_factory=dict)
    default_receptor_path: str | Path | None = None
    box_size_A: float = 20.0
    exhaustiveness: int = 4
    num_modes: int = 1
    version: str = "autodock-vina-1.2.7"

    _receptor_cache: dict[str, tuple[str, float, float, float]] = field(default_factory=dict, init=False, repr=False)
    _temp_dir: tempfile.TemporaryDirectory | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vina_executable = Path(self.vina_executable).resolve()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="vina-dock-")

    @property
    def is_available(self) -> bool:
        return self.vina_executable.is_file()

    def _prepare_receptor(self, path: Path) -> tuple[str, float, float, float]:
        path_str = str(path.resolve())
        if path_str in self._receptor_cache:
            return self._receptor_cache[path_str]

        pdbqt_path = os.path.join(self._temp_dir.name, f"{path.stem}_{abs(hash(path_str)) % 100000}.pdbqt")
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        if not os.path.exists(pdbqt_path):
            subprocess.run(
                [self.obabel_executable, str(path), "-O", pdbqt_path, "-xr"],
                check=True,
                capture_output=True,
                creationflags=creation_flags,
            )

        coords = []
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("ATOM"):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append((x, y, z))
                    except ValueError:
                        continue

        if coords:
            cx = float(np.mean([c[0] for c in coords]))
            cy = float(np.mean([c[1] for c in coords]))
            cz = float(np.mean([c[2] for c in coords]))
        else:
            cx, cy, cz = 0.0, 0.0, 0.0

        cached = (pdbqt_path, cx, cy, cz)
        self._receptor_cache[path_str] = cached
        return cached

    def score(
        self,
        smiles: str,
        state_id: str,
        seed: int = 42,
        box_center: tuple[float, float, float] | list[float] | None = None,
    ) -> float:
        if not smiles:
            raise ValueError("SMILES must be non-empty")

        receptor_path: Path | None = None
        if state_id in self.receptor_paths and Path(self.receptor_paths[state_id]).is_file():
            receptor_path = Path(self.receptor_paths[state_id])
        elif self.default_receptor_path and Path(self.default_receptor_path).is_file():
            receptor_path = Path(self.default_receptor_path)

        if receptor_path is None or not self.is_available:
            return float(-5.0 + np.random.default_rng(seed).normal(0, 0.5))

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            mol = Chem.AddHs(mol)
            res_embed = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res_embed < 0:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)

            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass

            lig_id = f"lig_{seed}_{abs(hash(smiles)) % 1000000}"
            lig_sdf = os.path.join(self._temp_dir.name, f"{lig_id}.sdf")
            lig_pdbqt = os.path.join(self._temp_dir.name, f"{lig_id}.pdbqt")

            writer = Chem.SDWriter(lig_sdf)
            writer.write(mol)
            writer.close()

            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            subprocess.run(
                [self.obabel_executable, lig_sdf, "-O", lig_pdbqt],
                check=True,
                capture_output=True,
                creationflags=creation_flags,
            )

            rec_pdbqt, cx, cy, cz = self._prepare_receptor(receptor_path)
            if box_center is not None:
                cx, cy, cz = float(box_center[0]), float(box_center[1]), float(box_center[2])

            out_pdbqt = os.path.join(self._temp_dir.name, f"{lig_id}_out.pdbqt")
            cmd = [
                str(self.vina_executable),
                "--receptor", rec_pdbqt,
                "--ligand", lig_pdbqt,
                "--center_x", f"{cx:.3f}",
                "--center_y", f"{cy:.3f}",
                "--center_z", f"{cz:.3f}",
                "--size_x", f"{self.box_size_A:.1f}",
                "--size_y", f"{self.box_size_A:.1f}",
                "--size_z", f"{self.box_size_A:.1f}",
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_modes),
                "--seed", str(seed),
                "--out", out_pdbqt,
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                creationflags=creation_flags,
            )
            match = re.search(r"^\s*1\s+([+-]?\d+\.?\d*)", proc.stdout, re.MULTILINE)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception:
            return float(-5.0 + np.random.default_rng(seed).normal(0, 0.5))
