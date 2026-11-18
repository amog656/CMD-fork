"""Adapter for the official CMD-GEN GCPG molecule generation module."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..data.models import PharmModel


_PHAR_TYPE_MAP = {
    "AROM": "AROM",
    "HYBL": "HYBL",
    "POSC": "POSC",
    "HACC": "HACC",
    "HDON": "HDON",
    "LHYBL": "LHYBL",
    "NEGI": "UNKNOWN",
    "OTHERS": "UNKNOWN",
}


def write_posp_file(model: PharmModel, path: Path, max_points: int = 8) -> None:
    """Format 3D pharmacophore points into the .posp file expected by GCPG."""
    lines = []
    points = model.points
    if len(points) > max_points:
        # Sort by confidence if available, otherwise take first max_points
        points = sorted(points, key=lambda p: (p.confidence is not None, p.confidence or 0.0), reverse=True)[:max_points]
    for pt in points:
        mapped_type = _PHAR_TYPE_MAP.get(pt.type.upper(), "UNKNOWN")
        x, y, z = pt.xyz
        lines.append(f"{mapped_type} {x:.4f} {y:.4f} {z:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class CMDGCPGAdapter:
    """Runs the frozen CMD-GEN GCPG model to decode pharmacophore graphs into SMILES."""

    repository_dir: str | Path
    model_path: str | Path
    tokenizer_path: str | Path
    python_executable: str = sys.executable
    device: str = "cpu"
    version: str = "cmd-gen-gcpg-frozen"

    def __post_init__(self) -> None:
        self.repository_dir = Path(self.repository_dir).resolve()
        self.model_path = Path(self.model_path).resolve()
        self.tokenizer_path = Path(self.tokenizer_path).resolve()

    @property
    def is_available(self) -> bool:
        """Check if repository and weights exist on disk."""
        return (
            (self.repository_dir / "GCPG" / "generate.py").is_file()
            and self.model_path.is_file()
            and self.tokenizer_path.is_file()
        )

    def generate(self, model: PharmModel, count: int, property_gates: dict, seed: int) -> list[str]:
        if count <= 0:
            return []
        if not model.points:
            raise ValueError("GCPG requires non-empty pharmacophore points")
        if not self.is_available:
            raise FileNotFoundError(
                f"GCPG model or tokenizer not found.\n"
                f"Model: {self.model_path}\n"
                f"Tokenizer: {self.tokenizer_path}\n"
                "Please download the weights and tokenizer from https://zenodo.org/records/13841142"
            )

        with tempfile.TemporaryDirectory(prefix="pharmpipe-gcpg-") as temp_dir:
            temp = Path(temp_dir)
            posp_path = temp / f"{model.model_id}.posp"
            out_dir = temp / "output"
            out_dir.mkdir()

            write_posp_file(model, posp_path)

            gcpg_script = self.repository_dir / "GCPG" / "generate.py"
            n_sample = max(count * 3, count + 8)
            batch_size = min(n_sample, 64)
            cmd = [
                self.python_executable,
                str(gcpg_script),
                str(posp_path),
                str(out_dir),
                str(self.model_path),
                str(self.tokenizer_path),
                "--n_mol", str(n_sample),
                "--batch_size", str(batch_size),
                "--device", self.device,
                "--seed", str(seed),
                "--filter",
                "--sample",
            ]

            env = os.environ.copy()
            env["DGLBACKEND"] = "pytorch"
            dgl_home = (Path(self.repository_dir).parent / ".dgl").resolve()
            dgl_home.mkdir(parents=True, exist_ok=True)
            env["DGL_HOME"] = str(dgl_home)

            fairseq_dir = str((Path(self.repository_dir).parent / "fairseq_repo").resolve())
            gcpg_dir = str(self.repository_dir / "GCPG")
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                [p for p in (fairseq_dir, gcpg_dir, current_pythonpath) if p]
            )

            proc = subprocess.run(
                cmd,
                cwd=str(self.repository_dir / "GCPG"),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"GCPG generation failed (return code {proc.returncode}):\n"
                    f"stderr: {proc.stderr[-2000:]}\nstdout: {proc.stdout[-2000:]}"
                )

            result_files = list(out_dir.glob("*_result.txt"))
            if not result_files:
                return []

            generated_smiles = [
                line.strip()
                for line in result_files[0].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return generated_smiles[:count]
