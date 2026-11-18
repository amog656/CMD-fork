from __future__ import annotations

from dataclasses import dataclass

from ..data.models import PocketAtom


@dataclass
class MockPocketDetectorAdapter:
    version: str = "mock-pocket-detector-1"

    def detect(self, atoms: list[PocketAtom], seed: int) -> list[PocketAtom]:
        if not atoms:
            return []
        # Deterministic mock: preserve input atom order and return its central subset.
        # TODO: call the selected pocket detector once its interface is approved.
        midpoint = len(atoms) // 2
        return atoms[: midpoint + 1]
