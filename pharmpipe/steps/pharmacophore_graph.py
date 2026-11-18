"""3D PharmModel to complete pharmacophore-graph conversion."""

from __future__ import annotations

import numpy as np

from ..data.models import PharmModel


FEATURE_TYPE_MAPPING = {
    "Aromatic": "AROM",
    "Hydrophobe": "HYBL",
    "PosIonizable": "POSC",
    "NegIonizable": "NEGI",
    "Acceptor": "HACC",
    "Donor": "HDON",
    "LumpedHydrophobe": "LHYBL",
}


def pharmacophore_graph(model: PharmModel, *, edge_distance_mode: str) -> dict:
    if edge_distance_mode != "euclidean_angstrom":
        raise ValueError("real GCPG repository edge encoding is unverified; only explicit euclidean_angstrom mock mode is available")
    coordinates = np.asarray([point.xyz for point in model.points], dtype=float)
    if not len(coordinates):
        raise ValueError("pharmacophore graph cannot be empty")
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    return {"model_id": model.model_id, "types": [point.type for point in model.points], "coordinates_A": coordinates.tolist(), "edge_distances_A": distances.tolist(), "edge_distance_mode": edge_distance_mode}
