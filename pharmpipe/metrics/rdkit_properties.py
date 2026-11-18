"""Molecular property calculations used for steering and reporting."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer


def compute_properties(smiles: str) -> dict[str, float | int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "QED": float(QED.qed(mol)),
        "SAS": float(sascorer.calculateScore(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
    }
