from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MockASAMTAdapter:
    version: str = "mock-asamt-1"

    def sample(self, coordinates, n_samples: int, temperature: float, seed: int) -> list[np.ndarray]:
        """Deterministic coordinate perturbations for plumbing tests only."""
        base = np.asarray(coordinates, dtype=float)
        if base.ndim != 2 or base.shape[1] != 3 or n_samples < 0:
            raise ValueError("coordinates must be N x 3 and n_samples non-negative")
        rng = np.random.default_rng(seed)
        # TODO: call the user-selected aSAMt package after its interface is confirmed.
        return [base + rng.normal(0, 0.01 * temperature / 300, base.shape) for _ in range(n_samples)]
