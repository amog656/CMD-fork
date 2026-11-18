from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from ..data.models import PharmModel


@dataclass
class MockGCPGAdapter:
    version: str = "mock-gcpg-1"
    valid_templates: tuple[str, ...] = (
        "CCO", "CCN", "c1ccccc1", "CC(=O)N", "CCOC", "c1ccncc1",
        "CC(C)O", "NCCO", "CC(=O)O", "COc1ccccc1", "CCS", "C1CCCCC1",
    )

    def generate(self, model: PharmModel, count: int, property_gates: dict, seed: int) -> list[str]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if not model.points:
            raise ValueError("mock GCPG requires a non-empty pharmacophore model")
        rng = np.random.default_rng(seed)
        # TODO: read the exact repository preprocessing and invoke frozen CMD-GEN GCPG.
        choices = rng.integers(0, len(self.valid_templates), size=count)
        return [Chem.MolToSmiles(Chem.MolFromSmiles(self.valid_templates[int(i)]), isomericSmiles=True) for i in choices]
