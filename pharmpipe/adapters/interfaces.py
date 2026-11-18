"""External tool boundaries; production implementations remain intentionally unspecified."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..data.models import PharmModel, PocketAtom, PocketState


class ASAMTAdapter(Protocol):
    version: str
    def sample(self, coordinates, n_samples: int, temperature: float, seed: int) -> list[np.ndarray]: ...


class PharmFlowAdapter(Protocol):
    version: str
    def predict(self, state: PocketState, seed: int) -> PharmModel: ...


class GCPGAdapter(Protocol):
    version: str
    def generate(self, model: PharmModel, count: int, property_gates: dict, seed: int) -> list[str]: ...


class DockingAdapter(Protocol):
    version: str
    def score(self, smiles: str, state_id: str, seed: int) -> float: ...


class MMGBSAAdapter(Protocol):
    version: str
    def rescore(self, smiles: str, pose_id: str, seed: int) -> tuple[float, float]: ...


class FEPAdapter(Protocol):
    version: str
    def calculate(self, smiles: str, protein_id: str, replicas: int, seed: int) -> dict: ...


class MDAdapter(Protocol):
    version: str
    def analyze_pose(self, compound_id: str, protein_id: str, replicas: int, seed: int) -> dict: ...


class PocketDetectorAdapter(Protocol):
    version: str
    def detect(self, atoms: list[PocketAtom], seed: int) -> list[PocketAtom]: ...
