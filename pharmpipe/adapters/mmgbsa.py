from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MockMMGBSAAdapter:
    version: str = "mock-mmgbsa-1"

    def rescore(self, smiles: str, pose_id: str, seed: int) -> tuple[float, float]:
        if not smiles:
            raise ValueError("SMILES must be non-empty")
        rng = np.random.default_rng(seed)
        # TODO: call a selected MM-GBSA engine; uncertainty is explicit.
        return float(-20 + rng.normal()), float(abs(rng.normal(1, 0.2)))
