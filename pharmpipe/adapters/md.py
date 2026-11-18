from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MockMDAdapter:
    version: str = "mock-md-1"

    def analyze_pose(self, compound_id: str, protein_id: str, replicas: int, seed: int) -> dict:
        if replicas <= 0:
            raise ValueError("replicas must be positive")
        rng = np.random.default_rng(seed)
        # TODO: run trajectory analysis after the user selects an MD engine and inputs.
        return {"replicas": replicas, "ligand_rmsd_A": rng.uniform(0.5, 3.0, replicas).tolist(), "contact_persistence": rng.uniform(0.3, 0.95, replicas).tolist()}
