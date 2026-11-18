from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.models import PharmModel, PharmPoint, PocketState


@dataclass
class MockPharmFlowAdapter:
    version: str = "mock-pharmflow-1"

    def predict(self, state: PocketState, seed: int) -> PharmModel:
        if not state.pocket_atoms:
            raise ValueError("mock PharmFlow requires pocket atoms")
        rng = np.random.default_rng(seed)
        center = np.mean([atom.xyz for atom in state.pocket_atoms], axis=0)
        feature_types = ("AROM", "HYBL", "POSC", "NEGI", "HACC", "HDON", "LHYBL", "OTHERS")
        points = [PharmPoint(tuple(map(float, center + rng.normal(0, 1, 3))), feature_types[index % len(feature_types)]) for index in range(4)]
        # TODO: replace with trained PharmFlow inference after model specification and weights exist.
        return PharmModel(f"mock_{state.state_id}", points, "per-state", [state.state_id], [1] * len(points))
