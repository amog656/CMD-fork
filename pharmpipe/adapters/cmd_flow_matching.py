"""Adapter for the CMD-fork DiffPhar Flow Matching checkpoint."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from ..data.models import PharmModel, PharmPoint, PocketState
from ..steps.structure_io import read_pdb_atoms


_FEATURE_TYPES = {
    "Aromatic": "AROM",
    "Hydrophobe": "HYBL",
    "PosIonizable": "POSC",
    "NegIonizable": "NEGI",
    "Acceptor": "HACC",
    "Donor": "HDON",
    "LumpedHydrophobe": "LHYBL",
    "others": "OTHERS",
}


def _pocket_residue_ids(pdb_path: Path) -> list[str]:
    atoms = read_pdb_atoms(pdb_path)
    residue_ids = sorted(
        {
            (atom["chain"], atom["resid"])
            for atom in atoms
            if atom["atom"] == "CA" and atom["element"].upper() == "C"
        },
        key=lambda item: (
            item[0],
            0 if item[1].lstrip("-").isdigit() else 1,
            int(item[1]) if item[1].lstrip("-").isdigit() else item[1],
        ),
    )
    if not residue_ids:
        raise ValueError(f"PharmFlow needs C-alpha atoms to define pocket residues: {pdb_path}")
    return [f"{chain}:{resid}" for chain, resid in residue_ids]


def _cluster_samples(state_id: str, samples: dict, seed: int) -> PharmModel:
    """Pool repeat samples and choose a Gaussian-mixture size by BIC per type."""
    coordinates_by_type: dict[str, list[list[float]]] = defaultdict(list)
    for sample_name in sorted(samples):
        sample = samples[sample_name]
        if not isinstance(sample, dict):
            raise ValueError(f"Invalid PharmFlow sample {sample_name!r} for state {state_id!r}")
        for feature_name, values in sample.items():
            if feature_name not in _FEATURE_TYPES:
                raise ValueError(f"Unknown PharmFlow feature class {feature_name!r}")
            for xyz in values:
                coordinates_by_type[_FEATURE_TYPES[feature_name]].append(xyz)

    points = []
    for type_index, feature_type in enumerate(sorted(coordinates_by_type)):
        xyz = np.asarray(coordinates_by_type[feature_type], dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError(f"Invalid coordinates for {feature_type} points in state {state_id!r}")
        if len(xyz) == 1:
            points.append(PharmPoint(tuple(map(float, xyz[0])), feature_type, 1.0))
            continue

        candidates = []
        # Keep model selection bounded; repeated generations are evidence for
        # a small set of modes, not justification for one point per sample.
        maximum_components = min(len(xyz), max(1, int(np.ceil(np.sqrt(len(xyz))))), 8)
        for component_count in range(1, maximum_components + 1):
            mixture = GaussianMixture(
                n_components=component_count,
                covariance_type="full",
                random_state=seed + type_index,
                reg_covar=1e-6,
            ).fit(xyz)
            candidates.append((mixture.bic(xyz), mixture))
        mixture = min(candidates, key=lambda item: item[0])[1]
        points.extend(
            PharmPoint(tuple(map(float, center)), feature_type, float(weight))
            for center, weight in zip(mixture.means_, mixture.weights_)
        )

    return PharmModel(f"pharmflow_{state_id}", points, "per-state", [state_id])


@dataclass
class CMDFlowMatchingAdapter:
    """Run the real checkpoint in its repository environment and pool its samples."""

    repository_dir: str | Path
    checkpoint_path: str | Path
    samples_per_state: int
    timesteps: int
    python_executable: str = sys.executable
    version: str = "cmd-fork-diffphar-dev-flow-matching"

    def __post_init__(self) -> None:
        self.repository_dir = Path(self.repository_dir).resolve()
        self.checkpoint_path = Path(self.checkpoint_path).resolve()
        if self.samples_per_state <= 0:
            raise ValueError("pharmflow_samples_per_state must be positive")
        if not 10 <= self.timesteps <= 20:
            raise ValueError("pharmflow_timesteps must be between 10 and 20 per the agreed plan")

    def predict(self, state: PocketState, seed: int) -> PharmModel:
        return self.predict_many([state], seed)[0]

    def predict_many(self, states: list[PocketState], seed: int) -> list[PharmModel]:
        if not states:
            return []
        required_repo_files = ("lightning_modules.py", "flow_matching.py", "generate_phars.py")
        missing_repo_files = [name for name in required_repo_files if not (self.repository_dir / name).is_file()]
        if missing_repo_files:
            raise FileNotFoundError(
                f"CMD-fork repository is missing {missing_repo_files}: {self.repository_dir}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"PharmFlow checkpoint does not exist: {self.checkpoint_path}")

        state_specs = []
        for state in states:
            pdb_path = Path(state.coordinate_path).resolve()
            if not pdb_path.is_file():
                raise FileNotFoundError(f"Pocket-state PDB does not exist: {pdb_path}")
            state_specs.append({
                "state_id": state.state_id,
                "pdb_path": str(pdb_path),
                "pocket_residues": _pocket_residue_ids(pdb_path),
            })

        worker = Path(__file__).with_name("cmd_flow_worker.py").resolve()
        with tempfile.TemporaryDirectory(prefix="pharmpipe-pharmflow-") as temp_dir:
            temp = Path(temp_dir)
            request_path, output_path = temp / "request.json", temp / "predictions.json"
            request_path.write_text(json.dumps({
                "repository_dir": str(self.repository_dir),
                "checkpoint_path": str(self.checkpoint_path),
                "states": state_specs,
                "samples_per_state": self.samples_per_state,
                "timesteps": self.timesteps,
                "seed": int(seed),
            }), encoding="utf-8")
            completed = subprocess.run(
                [self.python_executable, str(worker), str(request_path), str(output_path)],
                cwd=self.repository_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "CMD-fork PharmFlow inference failed. Confirm that the selected Python "
                    "environment has the repository dependencies installed.\n"
                    f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
                )
            if not output_path.is_file():
                raise RuntimeError("CMD-fork inference ended without writing its predictions file")
            result = json.loads(output_path.read_text(encoding="utf-8"))

        predictions = result.get("states", {})
        models = []
        for index, state in enumerate(states):
            raw_samples = predictions.get(state.state_id)
            if not isinstance(raw_samples, dict) or not raw_samples:
                raise ValueError(f"CMD-fork returned no pharmacophore samples for {state.state_id!r}")
            models.append(_cluster_samples(state.state_id, raw_samples, seed + index))
        return models
