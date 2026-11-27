"""Load a protein-ligand complex and prepare the protein plus pocket center."""

from __future__ import annotations

import copy
from io import StringIO
from typing import Sequence

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
KNOWN_LIGANDS = {
    "1FKB": "RAP", "1M17": "AQ4", "1W84": "L12", "29IA": "A1J2Q",
    "29ID": "A1J2O", "31FJ": "A1J9A", "31FK": "A1J9B", "31FL": "A1J88",
    "31FM": "A1J89", "31HW": "A1J9R", "31HZ": "A1J9S", "31JL": "A1J9T",
    "35XD": "A1DIJ", "35XH": "A1DIM", "35XL": "A1DIN", "9T9A": "A1JUL",
    "9T9B": "A1JUK", "9TWE": "6U9", "9XQP": "A1E09", "9XRK": "O5F",
}


class _ProteinChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " " and residue.resname.strip().upper() in STANDARD_AA

    def accept_atom(self, atom):
        element = (atom.element or "").strip().upper()
        return element not in {"H", "D"}


def _heavy_atoms(residue):
    by_name = {}
    for atom in residue.get_unpacked_list():
        element = (atom.element or "").strip().upper()
        if element in {"H", "D"}:
            continue
        by_name.setdefault(atom.get_name(), []).append(atom)
    selected = []
    for atoms in by_name.values():
        atoms.sort(key=lambda atom: (
            atom.get_altloc() in {" ", "A", "1"},
            float(atom.get_occupancy() or 0.0),
        ), reverse=True)
        selected.append(atoms[0])
    return selected


def extract_complex(file_bytes: bytes, filename: str) -> tuple[bytes, tuple[float, float, float], str]:
    """Return receptor-only PDB bytes, ligand heavy-atom centroid, and ligand ID.

    For an unannotated complex, the selected pocket ligand is the largest
    non-water, non-polymer residue with at least eight heavy atoms.
    """
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in {"pdb", "ent", "cif", "mmcif"}:
        raise ValueError("Upload a complete PDB or mmCIF protein-ligand complex.")
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True) if suffix in {"cif", "mmcif"} else PDBParser(QUIET=True)
        structure = parser.get_structure("uploaded_complex", StringIO(text))
        model = next(structure.get_models())
    except Exception as exc:
        raise ValueError(f"Could not read `{filename}` as a protein-ligand complex: {exc}") from exc

    protein_chains = []
    for chain in model:
        residues = [
            residue for residue in chain
            if residue.id[0] == " " and residue.resname.strip().upper() in STANDARD_AA
        ]
        if residues:
            protein_chains.append((len(residues), chain, residues))
    if not protein_chains:
        raise ValueError("This file contains no standard protein chain.")
    # Prefer chain A when chains have the same length; otherwise choose the
    # chain with the most standard amino-acid residues for a single-chain model.
    protein_chains.sort(key=lambda item: (item[0], item[1].id == "A"), reverse=True)
    _, protein_chain, protein_residues = protein_chains[0]

    water_names = {"HOH", "WAT", "DOD", "H2O"}
    ligand_candidates = []
    for chain in model:
        for residue in chain:
            if residue.id[0] == " " or residue.resname.strip().upper() in water_names:
                continue
            atoms = _heavy_atoms(residue)
            if len(atoms) >= 8 and any((atom.element or "").strip().upper() == "C" for atom in atoms):
                ligand_candidates.append((len(atoms), residue, atoms))
    if not ligand_candidates:
        raise ValueError("No bound organic ligand with at least eight heavy atoms was found in this complex.")
    expected_component = KNOWN_LIGANDS.get(filename.rsplit(".", 1)[0].upper())
    expected_candidates = [
        item for item in ligand_candidates
        if item[1].resname.strip().upper() == expected_component
    ] if expected_component else []
    ligand_candidates.sort(key=lambda item: item[0], reverse=True)
    _, ligand, ligand_atoms = (expected_candidates or ligand_candidates)[0]
    center_array = np.mean([atom.coord for atom in ligand_atoms], axis=0)
    center = tuple(float(value) for value in center_array)

    # PDB chain IDs are one character; some mmCIF author IDs are longer.
    # Copy only the selected protein chain and normalize its ID for aSAM.
    single_chain_structure = Structure("protein")
    single_chain_model = Model(0)
    normalized_chain = copy.deepcopy(protein_chain)
    normalized_chain.id = "A"
    single_chain_model.add(normalized_chain)
    single_chain_structure.add(single_chain_model)
    writer = PDBIO()
    writer.set_structure(single_chain_structure)
    output = StringIO()
    writer.save(output, select=_ProteinChainSelect("A"))
    protein_lines = [line for line in output.getvalue().splitlines() if line.startswith("ATOM  ")]
    # aSAM expects the selected chain to be named A.
    protein_lines = [line[:21] + "A" + line[22:] for line in protein_lines]
    if not protein_lines:
        raise ValueError("Could not extract protein atoms from the selected chain.")
    receptor_bytes = ("\n".join([*protein_lines, "TER", "END", ""])).encode("ascii", errors="replace")
    ligand_id = f"{ligand.resname.strip()} chain {ligand.get_parent().id}"
    return receptor_bytes, center, ligand_id
