from typing import Any, Sequence, Mapping, Optional
import string

import numpy as np
import mdtraj
import torch

from sam.openfold.np.residue_constants import (
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_name_to_atom14_names,
    restypes,
    restype_1to3
)
from sam.openfold.np.protein import Protein, from_pdb_string
from sam.openfold.np import residue_constants
# from sam.openfold.data.data_pipeline import make_pdb_features
from sam.openfold.data import data_transforms

from sam.data.sequences import aa_one_to_three_dict, aa_list, aa_three_letters


#
# mdtraj trajectories.
#

def _check_aa_atom(a, standard=True) -> bool:
    if standard:
        return a.element.symbol != "H" and a.residue.name in aa_three_letters
    else:
        return a.element.symbol != "H"

def slice_aa_traj(traj, standard=True):
    aa_ids = [a.index for a in traj.topology.atoms \
              if _check_aa_atom(a, standard)]
    traj = traj.atom_slice(aa_ids)
    return traj


#
# Load OpenFold data.
#

def get_po_from_pdb_string(
        pdb_string: str,
        unsqueeze: bool = False
    ):

    _po = from_pdb_string(pdb_string)

    po = {}
    po["aatype"] = torch.tensor(_po.aatype)
    po["all_atom_mask"] = torch.tensor(_po.atom_mask)  # _po.aatype
    po["all_atom_positions"] = torch.tensor(_po.atom_positions)
    if unsqueeze:
        po["aatype"] = po["aatype"].unsqueeze(0)
        po["all_atom_mask"] = po["all_atom_mask"].unsqueeze(0)
        po["all_atom_positions"] = po["all_atom_positions"].unsqueeze(0)
    return po


def get_po_from_mdtraj(
        topology: mdtraj.Topology,
        xyz: np.ndarray,
        chain_idx: int = None,
        unsqueeze: bool = False
    ) -> dict:

    _po = from_mdtraj(
        topology=topology,
        xyz=xyz,
        chain_idx=chain_idx,
        unsqueeze=unsqueeze
    )
    po = {}
    po["aatype"] = torch.tensor(_po.aatype)
    po["all_atom_mask"] = torch.tensor(_po.atom_mask)
    po["all_atom_positions"] = torch.tensor(_po.atom_positions)
    if unsqueeze:
        po["aatype"] = po["aatype"].unsqueeze(0)
        po["all_atom_mask"] = po["all_atom_mask"].unsqueeze(0)
        po["all_atom_positions"] = po["all_atom_positions"].unsqueeze(0)
    return po


def from_mdtraj(
        topology: mdtraj.Topology,
        xyz: np.ndarray,
        chain_idx: int = None,
        unsqueeze: bool = False
    ) -> Protein:
    """Takes a ... and constructs a Protein object.

    Adapted from the openfold.np.protein.from_pdb_string function.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    if len(xyz.shape) != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "Not a 3D structure, expected a shape of (*, 3)"
            " received: {}".format(xyz.shape)
        )
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in topology.chains:
        if(chain_idx is not None and chain.index != chain_idx):
            continue

        for res in chain.residues:
            res_shortname = res.code
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res.atoms:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = xyz[atom.index]
                mask[residue_constants.atom_order[atom.name]] = 1.0
                # res_b_factors[
                #     residue_constants.atom_order[atom.name]
                # ] = 0.1  # atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue

            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.resSeq)
            chain_ids.append(chain.index)
            b_factors.append(res_b_factors)

    parents = None
    parents_chain_index = None
    # if("PARENT" in pdb_str):
    #     parents = []
    #     parents_chain_index = []
    #     chain_id = 0
    #     for l in pdb_str.split("\n"):
    #         if("PARENT" in l):
    #             if(not "N/A" in l):
    #                 parent_names = l.split()[1:]
    #                 parents.extend(parent_names)
    #                 parents_chain_index.extend([
    #                     chain_id for _ in parent_names
    #                 ])
    #             chain_id += 1

    # unique_chain_ids = np.unique(chain_ids)
    # chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    chain_index = np.array(chain_ids)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        parents=parents,
        parents_chain_index=parents_chain_index,
    )


def get_frames_from_po(po: dict):

    # ['aatype', 'all_atom_mask', 'all_atom_positions', 'rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_alt_gt_frames']

    ## po = data_transforms.atom37_to_frames(po)
    ## po = data_transforms.make_atom14_masks(po)
    ## po = data_transforms.get_backbone_frames(po)

    data_transforms.make_seq_mask(po)
    data_transforms.make_atom14_masks(po)  ##
    data_transforms.make_atom14_positions(po)  ## 
    data_transforms.atom37_to_frames(po)
    data_transforms._atom37_to_torsion_angles(po, "")  ##
    data_transforms._make_pseudo_beta(po, "")  ##
    data_transforms.get_backbone_frames(po)
    data_transforms.get_chi_angles(po)  ##

    return po


#
# Iterface with OpenFold.
#

# restypes = [
#     "A",
#     "R",
#     "N",
#     "D",
#     "C",
#     "Q",
#     "E",
#     "G",
#     "H",
#     "I",
#     "L",
#     "K",
#     "M",
#     "F",
#     "P",
#     "S",
#     "T",
#     "W",
#     "Y",
#     "V",
# ]

# Converts SAM residue numbering to the OpenFold one.
if len(restypes) != len(aa_list):
    raise ValueError()
sam_openfold_aa_map = []
for aa_i in aa_list:
    sam_openfold_aa_map.append(restypes.index(aa_i))
sam_openfold_aa_map = np.array(sam_openfold_aa_map)


def _get_a(structure: dict, a: torch.Tensor = None):
    if "a" in structure:
        _a = structure["a"]
    else:
        _a = a
    if _a is None:
        raise TypeError()
    return _a

# Amino acid pair indices, used for building statistical potentials.
ap_mapping = np.zeros((20, 20))
ap_count = 0
for i, r_i in enumerate(restypes):
    for j, r_j in enumerate(restypes):
        ap_mapping[i, j] = ap_count
        ap_count += 1

"""
def get_traj(
        structure: dict,
        a: torch.Tensor = None
    ):
    _a = _get_a(structure, a)

    mask = restype_atom14_mask[_a]

    # Get first element of column and compare to all remaining ones.
    are_same = np.all(_a == _a[0, :], axis=0)
    
    if not np.all(are_same):
        raise ValueError()
    
    raise NotImplementedError()
"""

def get_atom14_topology(sequence: str, verbose: bool = False):
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for i, si in enumerate(sequence):
        if verbose:
            print("---")
        res_obj = topology.add_residue(aa_one_to_three_dict[si], chain)
        ai = restype_name_to_atom14_names[aa_one_to_three_dict[si]]
        for j, aj in enumerate(ai):
            if aj:
                if verbose:
                    print(aj)
                if aj.startswith("C"):
                    elem = mdtraj.core.topology.elem.carbon
                elif aj.startswith("O"):
                    elem = mdtraj.core.topology.elem.oxygen
                elif aj.startswith("N"):
                    elem = mdtraj.core.topology.elem.nitrogen
                elif aj.startswith("S"):
                    elem = mdtraj.core.topology.elem.sulfur
                else:
                    raise NotImplementedError()
                topology.add_atom(
                    aj, elem, res_obj)
    return topology


def get_atom14_mask(aa, device=None):
    atom14_ids = []
    for res in restypes:
        vals_i = []
        for a in restype_name_to_atom14_names[restype_1to3[res]]:
            if a:
                vals_i.append(1)
            else:
                vals_i.append(0)
        atom14_ids.append(vals_i)
    atom14_ids = torch.tensor(
        atom14_ids,
        dtype=torch.long,
        device=device if device is not None else aa.device
    )
    atom14_atom_exists = atom14_ids[aa]
    return atom14_atom_exists


def get_traj_list(
        structure: dict,
        a: torch.Tensor = None,
        verbose: bool = False,
        join: bool = False
    ):

    _a = _get_a(structure, a).cpu()
    
    mask = restype_atom14_mask[_a]

    traj_l = []
    for i in range(structure["positions"].shape[0]):
    
        one_letter_seq_i = [restypes[j] for j in _a[i]]

        xyz_atom14_i = structure["positions"][i]
        bool_mask_i = mask[i].astype(bool)
        bool_mask_i = torch.Tensor(bool_mask_i) == 1.0
        # bool_mask_i = mask[i].astype(bool)[...,None]

        # Use the boolean mask to filter data
        # We need to expand the dimensions of bool_mask to match data's dimensions for broadcasting
        xyz_traj_i = xyz_atom14_i[bool_mask_i]
        # Now filtered_data will have shape (x, 3) where x is the number of True in bool_mask

        topology_i = get_atom14_topology(one_letter_seq_i, verbose=verbose)

        traj_i = mdtraj.Trajectory(xyz=xyz_traj_i.detach().cpu().numpy()*0.1,
                                   topology=topology_i)
        if verbose:
            print(traj_i)
        traj_l.append(traj_i)

    if not join:
        return traj_l
    else:
        return mdtraj.join(traj_l)


per_conf_attrs = [
    "x",
    "backbone_rigid_tensor",
    "rigidgroups_gt_frames",
    "rigidgroups_alt_gt_frames",
    "atom14_gt_positions",
    "atom14_alt_gt_positions",
    "chi_angles_sin_cos"
]

compressed_per_conf_attrs = [
    "rigidgroups_gt_frames",
    "rigidgroups_alt_gt_frames",
    "atom14_gt_positions",
    "atom14_alt_gt_positions",
    "chi_angles_sin_cos"
]

per_sys_attrs = [
    "a",
    "r",
    "backbone_rigid_mask",
    "rigidgroups_gt_exists",
    "atom14_atom_is_ambiguous",
    "atom14_gt_exists",
    "atom14_alt_gt_exists",
    "atom14_atom_exists",
    "seq_mask",
    "chi_mask"
]