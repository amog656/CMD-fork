import os
import numpy as np
import mdtraj
from sam.data.sequences import (aa_one_letter,
                                aa_three_letters,
                                aa_one_to_three_dict)


def get_ca_topology(sequence: str):
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for res in sequence:
        res_obj = topology.add_residue(aa_one_to_three_dict[res], chain)
        topology.add_atom("CA", mdtraj.core.topology.elem.carbon, res_obj)
    return topology


def get_seq_from_top(top: str) -> str:
    if isinstance(top, str):
        top_traj = mdtraj.load(top)
        topology = top_traj.topology
    else:
        topology = top
    seq = [r.code for r in topology.residues if \
           r.name in aa_three_letters]
    return "".join(seq)


def _check_ca_atom(a, standard=True) -> bool:
    if standard:
        return a.name == "CA" and a.residue.name in aa_three_letters  # a.residue.code in aa_one_letter
    else:
        return a.name == "CA"

# def _check_cg_atom(a):
#     return a.name in ("CG", "CG2") and a.residue.code in aa_one_letter

def slice_ca_traj(traj, standard=True):
    ca_ids = [a.index for a in traj.topology.atoms \
              if _check_ca_atom(a, standard)]
    traj = traj.atom_slice(ca_ids)
    return traj

def slice_traj_to_com(traj, get_xyz=True):
    ha_ids = [a.index for a in traj.topology.atoms if \
              a.residue.name in aa_three_letters and \
              a.element.symbol != "H"]
    ha_traj = traj.atom_slice(ha_ids)
    residues = list(ha_traj.topology.residues)
    com_xyz = np.zeros((ha_traj.xyz.shape[0], len(residues), 3))
    for i, residue_i in enumerate(residues):
        ha_ids_i = [a.index for a in residue_i.atoms]
        masses_i = np.array([a.element.mass for a in residue_i.atoms])
        masses_i = masses_i[None,:,None]
        tot_mass_i = masses_i.sum()
        com_xyz_i = np.sum(ha_traj.xyz[:,ha_ids_i,:]*masses_i, axis=1)/tot_mass_i
        com_xyz[:,i,:] = com_xyz_i
    if get_xyz:
        return com_xyz
    else:
        return mdtraj.Trajectory(
            xyz=com_xyz,
            topology=get_ca_topology(
                sequence="".join([r.code for r in ha_traj.topology.residues])
            ))


residue_atoms = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "N",
        "NE1",
        "O",
    ],
    "TYR": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "N",
        "O",
        "OH",
    ],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}

atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

def get_atom14_sam_data(traj):
    topology = traj.topology
    n_frames = len(traj)
    n_residues = topology.n_residues

    atom14_xyz = np.zeros((n_frames, n_residues, 14, 3))
    atom14_mask = np.zeros((n_frames, n_residues, 14))
    for res_idx, res in enumerate(topology.residues):
        if res.name not in residue_atoms:
            raise KeyError(res.name)
        for atom in res.atoms:
            if atom.name in residue_atoms[res.name]:
                atom14_mask[:,res_idx,residue_atoms[res.name].index(atom.name)] = 1
                atom14_xyz[:, res_idx,residue_atoms[res.name].index(atom.name)] = traj.xyz[:,atom.index]
            else:
                pass
    for res_idx in range(n_residues):
        if atom14_mask[:,res_idx,1].sum() != n_frames:
            raise ValueError(f"No Ca atoms for residue id {res_idx}")
    return {"xyz": atom14_xyz, "top": atom14_mask}