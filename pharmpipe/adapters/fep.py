from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MockFEPAdapter:
    version: str = "mock-fep-1"

    def calculate(self, smiles: str, protein_id: str, replicas: int, seed: int) -> dict:
        if not smiles or replicas <= 0:
            raise ValueError("SMILES and positive replica count are required")
        rng = np.random.default_rng(seed)
        estimates = rng.normal(-7, 0.5, replicas)
        # TODO: call a user-selected FEP engine; mock values are not evidence.
        return {"replica_values": estimates.tolist(), "mean": float(estimates.mean()), "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))], "converged": False}
