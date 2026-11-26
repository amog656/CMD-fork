"""
TODO:
    - filter terms by constant from force fields
    - implement L amino acid improper dihedrals
    - build omega from force field parameters
"""

import os
import math
import time
import json
from typing import Tuple
import torch
import numpy as np
from sam.data.sequences import (
    ofo_restypes, ofo_restype_name_to_atom14_names, aa_one_to_three_dict,
    chi_angles_atoms
)
from sam.data.aa_topology import get_atom14_mask
from sam.utils import print_msg


######################
# Define parameters. #
######################

# PHI_ATOMS = ["-C", "N", "CA", "C"]
# PSI_ATOMS = ["N", "CA", "C", "+N"]
# OMEGA_ATOMS = ["CA", "C", "+N", "+CA"]
# CHI1_ATOMS = [
#     ["N", "CA", "CB", "CG"],
#     ["N", "CA", "CB", "CG1"],
#     ["N", "CA", "CB", "SG"],
#     ["N", "CA", "CB", "OG"],
#     ["N", "CA", "CB", "OG1"],
# ]
# CHI2_ATOMS = [
#     ["CA", "CB", "CG", "CD"],
#     ["CA", "CB", "CG", "CD1"],
#     ["CA", "CB", "CG1", "CD1"],
#     ["CA", "CB", "CG", "OD1"],
#     ["CA", "CB", "CG", "ND1"],
#     ["CA", "CB", "CG", "SD"],
# ]
# CHI3_ATOMS = [
#     ["CB", "CG", "CD", "NE"],
#     ["CB", "CG", "CD", "CE"],
#     ["CB", "CG", "CD", "OE1"],
#     ["CB", "CG", "SD", "CE"],
# ]
# CHI4_ATOMS = [
#     ["CG", "CD", "NE", "CZ"],
#     ["CG", "CD", "CE", "NZ"],
# ]
# CHI5_ATOMS = [["CD", "NE", "CZ", "NH1"]]


openstructure_data = """Non-bonded distance     Minimum Dist    Tolerance
C-C                     3.4             1.5
C-N                     3.25            1.5
C-S                     3.5             1.5
C-O                     3.22            1.5
N-N                     3.1             1.5
N-S                     3.35            1.5
N-O                     3.07            1.5
O-S                     3.32            1.5
O-O                     3.04            1.5
S-S                     2.03            1.0"""

os_params = {}
for l in openstructure_data.split("\n")[1:]:
    els, thresh, tol = l.split()
    el_1, el_2 = els.split("-")
    os_params[(el_1, el_2)] = (float(thresh)*0.1, float(tol)*0.1)
    os_params[(el_2, el_1)] = (float(thresh)*0.1, float(tol)*0.1)


###################
# Setup topology. #
###################

msg_tag = "minimization"

def get_topology(
        a: torch.Tensor,
        use_ff_consts: bool = False,
        nb_mode: str = "const",
        nb_const_val: int = 0.33,
        nb_os_tol: float = None,
        device: str = None,
        verbose: bool = False
    ):

    module_dp = os.path.dirname(__file__)
    ff_fn = "mizu.20022025.json"
    params_fp = os.path.join(module_dp, "params", ff_fn)
    with open(params_fp, "r") as i_fh:
        force_field_params = json.load(i_fh)

    if device is None:
        device = a.device

    seq = [ofo_restypes[i.item()] for i in a[0]]

    topology = {
        # Bonds.
        "bonds": {"params": [], "ids": []},
        # Bond angles.
        "angles": {"params": [], "ids": []},
        # Dihedrals from force field.
        "proper_dihedrals": {"params": [], "ids": []},
        "improper_dihedrals": {"params": [], "ids": []},
        # Dihedrals from input structures.
        "phi_psi": {"params": [], "ids": []},
        "chi": {"params": [], "ids": []},
        # Non-bonbed interactions.
        "nb_centers": {"ids": [], "map_ids": {}, "map_params": {}},
        "nb_config": {"use_os": nb_mode == "os"},
        "nb_cache": {"params": [], "ids": []},
        "cabl": {"params": [], "ids": []},
        # Topology.
        "seq": seq,
        "atoms_dict": {}
    }

    atoms_dict = {}
    atom_idx_count = 0
    for m, aa in enumerate(seq):
        for atm in ofo_restype_name_to_atom14_names[aa_one_to_three_dict[aa]]:
            if atm:
                atoms_dict[(m, atm)] = atom_idx_count
                atom_idx_count += 1

    for m, aa in enumerate(seq):
        aa3 = aa_one_to_three_dict[aa]
        if not aa3 in force_field_params:
            raise KeyError(aa3)
        print_msg("", verbose=verbose, tag=msg_tag)
        print_msg(f"# {m} {aa}", verbose=verbose, tag=msg_tag)

        # Add intra-residue bonds.
        for t in force_field_params[aa3]["bonds"]:
            atm_i, atm_j, dist, const = t[0:4]
            idx_i = atoms_dict[(m, atm_i)]
            idx_j = atoms_dict[(m, atm_j)]
            print_msg("INTRA-BOND: {} {} {} {} {} {}".format(
                atm_i, atm_j, dist, const, idx_i, idx_j),
                verbose=verbose, tag=msg_tag
            )
            topology["bonds"]["ids"].append([idx_i, idx_j])
            if use_ff_consts:
                topology["bonds"]["params"].append([dist, const])
            else:
                topology["bonds"]["params"].append(dist)

        # Add intra-residue angles.
        for t in force_field_params[aa3]["angles"]:
            atm_i, atm_j, atm_k, angle, const = t[0:5]
            idx_i = atoms_dict[(m, atm_i)]
            idx_j = atoms_dict[(m, atm_j)]
            idx_k = atoms_dict[(m, atm_k)]
            print_msg("INTRA-ANGLE: {} {} {} {} {} {} {} {}".format(
                atm_i, atm_j, atm_k, angle, const, idx_i, idx_j, idx_k),
                verbose=verbose, tag=msg_tag
            )
            topology["angles"]["ids"].append([idx_i, idx_j, idx_k])
            if use_ff_consts:
                topology["angles"]["params"].append([angle, const])
            else:
                topology["angles"]["params"].append(angle)
        
        # Residue-specific proper dihedrals.
        for t in force_field_params[aa3]["proper_dihedrals"]:
            atm_i, atm_j, atm_k, atm_l, period, phase, const = t[0:7]
            idx_i = atoms_dict[(m, atm_i)]
            idx_j = atoms_dict[(m, atm_j)]
            idx_k = atoms_dict[(m, atm_k)]
            idx_l = atoms_dict[(m, atm_l)]
            '''
            for period_v, phase_v in zip(period, phase):
                print("INTRA-PRDH:", atm_i, atm_j, atm_k, atm_l, period_v, phase_v, idx_i, idx_j, idx_k, idx_l)
                topology["proper_dihedrals"]["ids"].append([idx_i, idx_j, idx_k, idx_l])
                topology["proper_dihedrals"]["params"].append((period_v, phase_v))
            '''
            for period_v, phase_v, const_v in zip(period, phase, const):
                print_msg("INTRA-PRDH: {} {} {} {} {} {} {} {} {} {} {}".format(
                    atm_i, atm_j, atm_k, atm_l, period_v, phase_v, const_v, idx_i, idx_j, idx_k, idx_l),
                    verbose=verbose, tag=msg_tag
                )
                topology["proper_dihedrals"]["ids"].append([idx_i, idx_j, idx_k, idx_l])
                if use_ff_consts:
                    topology["proper_dihedrals"]["params"].append((period_v, phase_v, const_v))
                else:
                    topology["proper_dihedrals"]["params"].append((period_v, phase_v))

        for t in force_field_params[aa3]["improper_dihedrals"]:
            atm_i, atm_j, atm_k, atm_l, period, phase, const = t[0:7]
            idx_i = atoms_dict[(m, atm_i)]
            idx_j = atoms_dict[(m, atm_j)]
            idx_k = atoms_dict[(m, atm_k)]
            idx_l = atoms_dict[(m, atm_l)]
            for period_v, phase_v, const_v in zip(period, phase, const):
                print_msg("INTRA-IMDH: {} {} {} {} {} {} {} {} {} {} {}".format(
                    atm_i, atm_j, atm_k, atm_l, period_v, phase_v, const_v, idx_i, idx_j, idx_k, idx_l),
                    verbose=verbose, tag=msg_tag
                )
                topology["improper_dihedrals"]["ids"].append([idx_i, idx_j, idx_k, idx_l])
                if use_ff_consts:
                    topology["improper_dihedrals"]["params"].append((period_v, phase_v, const_v))
                else:
                    topology["improper_dihedrals"]["params"].append((period_v, phase_v))
        
        # Chi dihedrals.
        for t in chi_angles_atoms[aa3]:
            atm_i, atm_j, atm_k, atm_l = t
            idx_i = atoms_dict[(m, atm_i)]
            idx_j = atoms_dict[(m, atm_j)]
            idx_k = atoms_dict[(m, atm_k)]
            idx_l = atoms_dict[(m, atm_l)]
            print_msg("CHI-PRDH: {} {} {} {} {} {} {} {}".format(
                atm_i, atm_j, atm_k, atm_l, idx_i, idx_j, idx_k, idx_l),
                verbose=verbose, tag=msg_tag
            )
            topology["chi"]["ids"].append([idx_i, idx_j, idx_k, idx_l])
            # topology["proper_dihedrals"]["params"].append((period, phase))

        # Inter-residue energy terms.
        if m > 0:
            
            # Add peptide bond length and angles.
            for t, off in force_field_params["INTER"]["bonds"]:
                atm_i, atm_j, dist, const = t[0:4]
                off_i, off_j = off
                idx_i = atoms_dict[(m+off_i, atm_i)]
                idx_j = atoms_dict[(m+off_j, atm_j)]
                print_msg("INTER-BOND: {} {} {} {} {} {}".format(
                    atm_i, atm_j, dist, const, idx_i, idx_j),
                    verbose=verbose, tag=msg_tag
                )
                topology["bonds"]["ids"].append([idx_i, idx_j])
                if use_ff_consts:
                    topology["bonds"]["params"].append([dist, const])
                else:
                    topology["bonds"]["params"].append(dist)
            
            for t, off in force_field_params["INTER"]["angles"]:
                atm_i, atm_j, atm_k, angle, const = t[0:5]
                off_i, off_j, off_k = off
                idx_i = atoms_dict[(m+off_i, atm_i)]
                idx_j = atoms_dict[(m+off_j, atm_j)]
                idx_k = atoms_dict[(m+off_k, atm_k)]
                print_msg("INTER-ANGLE: {} {} {} {} {} {} {} {}".format(
                    atm_i, atm_j, atm_k, angle, const, idx_i, idx_j, idx_k),
                    verbose=verbose, tag=msg_tag
                )
                topology["angles"]["ids"].append([idx_i, idx_j, idx_k])
                if use_ff_consts:
                    topology["angles"]["params"].append([angle, const])
                else:
                    topology["angles"]["params"].append(angle)

            # Omega torsion angle.
            for t, off in force_field_params["INTER"]["proper_dihedrals"]:
                atm_i, atm_j, atm_k, atm_l, period, phase, const = t[0:7]
                off_i, off_j, off_k, off_l = off
                idx_i = atoms_dict[(m+off_i, atm_i)]
                idx_j = atoms_dict[(m+off_j, atm_j)]
                idx_k = atoms_dict[(m+off_k, atm_k)]
                idx_l = atoms_dict[(m+off_l, atm_l)]
                '''
                print("INTER-PRDH:", atm_i, atm_j, atm_k, atm_l, period, phase, idx_i, idx_j, idx_k, idx_l)
                topology["proper_dihedrals"]["ids"].append([idx_i, idx_j, idx_k, idx_l])
                topology["proper_dihedrals"]["params"].append((period, phase))
                '''
                for period_v, phase_v, const_v in zip(period, phase, const):
                    print_msg("INTER-PRDH: {} {} {} {} {} {} {} {} {} {} {}".format(
                        atm_i, atm_j, atm_k, atm_l, period_v, phase_v, const_v, idx_i, idx_j, idx_k, idx_l),
                        verbose=verbose, tag=msg_tag
                    )
                    topology["proper_dihedrals"]["ids"].append([idx_i, idx_j, idx_k, idx_l])
                    if use_ff_consts:
                        topology["proper_dihedrals"]["params"].append((period_v, phase_v, const_v))
                    else:
                        topology["proper_dihedrals"]["params"].append((period_v, phase_v))
            
            # Add phi and psi restraints.
            for t, off in [
                    (("C", "N", "CA", "C"), (-1, 0, 0, 0)),    # phi
                    (("N", "CA", "C", "N"), (-1, -1, -1, 0)),  # psi
                ]:
                atm_i, atm_j, atm_k, atm_l = t
                off_i, off_j, off_k, off_l = off
                idx_i = atoms_dict[(m+off_i, atm_i)]
                idx_j = atoms_dict[(m+off_j, atm_j)]
                idx_k = atoms_dict[(m+off_k, atm_k)]
                idx_l = atoms_dict[(m+off_l, atm_l)]
                print_msg(
                    "PHIPSI-PRDH: {} {} {} {} {} {} {} {}".format(
                        atm_i, atm_j, atm_k, atm_l, idx_i, idx_j, idx_k, idx_l
                    ), verbose=verbose, tag=msg_tag
                )
                topology["phi_psi"]["ids"].append([idx_i, idx_j, idx_k, idx_l])
        
        # Add Ca-Ca bond restraints.
        if m < len(seq)-1:
            idx_i = atoms_dict[(m, "CA")]
            idx_j = atoms_dict[(m+1, "CA")]
            print_msg(
                "CA-BOND: {} {} {} {}".format("CA", "CA", idx_i, idx_j),
                verbose=verbose, tag=msg_tag
            )
            topology["cabl"]["ids"].append([idx_i, idx_j])
        
    # Add non-bonded centers list.
    if nb_mode == "const":
        topology["nb_cache"]["params"] = nb_const_val
        set_nb_param = lambda p, *args: p
    elif nb_mode == "os":
        if nb_os_tol is None:
            def set_nb_param(params, a_i, a_j):
                params.append(os_params[(a_i, a_j)][0])
        else:
            def set_nb_param(params, a_i, a_j):
                p_ij = os_params[(a_i, a_j)]
                params.append(p_ij[0] + nb_os_tol*p_ij[1])
    else:
        raise KeyErrornb_mode

    for m, aa_m in enumerate(seq):
        for n, aa_n in enumerate(seq):
            if not m+1 < n:
                continue
            cen_idx_m = atoms_dict[(m, "CA")]
            cen_idx_n = atoms_dict[(n, "CA")]
            topology["nb_centers"]["ids"].append([cen_idx_m, cen_idx_n])
            atms_m = _get_residue_atoms(atoms_dict, m, aa_m)
            atms_n = _get_residue_atoms(atoms_dict, n, aa_n)
            mn_pairs = []
            mn_params = []
            for atm_m_u, atm_name_m_u in atms_m:
                for atm_n_v, atm_name_n_v in atms_n:
                    mn_pairs.append([atm_m_u, atm_n_v])
                    set_nb_param(mn_params, atm_name_m_u, atm_name_n_v)
            topology["nb_centers"]["map_ids"][(cen_idx_m, cen_idx_n)] = mn_pairs
            topology["nb_centers"]["map_params"][(cen_idx_m, cen_idx_n)] = mn_params
        
    for feat in (
            "bonds", "angles",
            "proper_dihedrals",
            "improper_dihedrals",
            "phi_psi", "chi",
            "nb_centers",
            "cabl"
        ):
        topology[feat]["ids"] = torch.tensor(
            topology[feat]["ids"], dtype=torch.long
        )
        if feat == "nb_centers":
            topology[feat]["ids"] = topology[feat]["ids"].to(device)
        if feat in (
                "bonds", "angles", "proper_dihedrals", "improper_dihedrals",
            ):
            topology[feat]["params"] = torch.tensor(
                topology[feat]["params"], device=device
            )
    topology["atoms_dict"] = atoms_dict
    
    return topology


def _get_residue_atoms(atoms_dict, idx, aa):
        atms = []
        for atm_l in ofo_restype_name_to_atom14_names[aa_one_to_three_dict[aa]]:
            if atm_l:
                atms.append((atoms_dict[(idx, atm_l)], _get_element(atm_l)))
        if not atms:
            raise ValueError()
        return atms

def _get_element(atm_name):
    return atm_name[0]


###################
# Compute energy. #
###################

def _get_bond_params(params):
    if len(params.shape) == 1:
        return params, 1.0
    elif len(params.shape) == 2:
        return params[:,0], params[:,1]
    else:
        raise ValueError(params.shape)

def _get_angle_params(params):
    if len(params.shape) == 1:
        return params, 1.0
    elif len(params.shape) == 2:
        return params[:,0], params[:,1]
    else:
        raise ValueError(params.shape)

def _get_torsion_params(params):
    if params.shape[1] == 2:
        return params[:,0], params[:,1], 1.0
    elif params.shape[1] == 3:
        return params[:,0], params[:,1], params[:,2]*0.5
    else:
        raise ValueError(params.shape)


def calc_energy(
        positions: torch.Tensor,
        topology: dict,
        bond_const: float = 1e4,  # original: 1e4
        angle_const: float = 1e3,  # original: 1e3
        proper_dihedral_const: float = 1e1,  # original: 1e1
        improper_dihedral_const: float = 1e1,  # original: 1e1
        phi_psi_const: float = 1e2,  # original: 1e1
        chi_const: float = 1e3,  # original: 1e3
        nb_const: float = 1e2,  # original: 1e2
        nb_form: str = "l2",
        early_stopping_hc_score: float = None,
        early_stopping_hc_thresh: float = 0.16,
        cabl_const: float = None,
        eps: float = 1e-12,
        verbose: bool = False
    ):
    """
    "Simulate" the effect of temperature via small perturbation of equilibrium
    values!
    """
    #
    #
    #

    # Total energy.
    tot_energy = 0

    # Bond length energy.
    bond_ids_i = topology["bonds"]["ids"][:,0]
    bond_ids_j = topology["bonds"]["ids"][:,1]
    bond_lens = torch.sqrt(
        torch.sum(
            torch.square(
                positions[:,bond_ids_i,:] - positions[:,bond_ids_j,:]
            ), dim=2
        )
    )
    bond_l0, bond_k = _get_bond_params(topology["bonds"]["params"])
    bond_energy = torch.square(bond_lens - bond_l0)*bond_k
    bond_energy = bond_energy.sum(dim=1)

    # Bond angle energy.
    angle_ids_i = topology["angles"]["ids"][:,0]
    angle_ids_j = topology["angles"]["ids"][:,1]
    angle_ids_k = topology["angles"]["ids"][:,2]
    angles = calc_angles_(
        positions[:,angle_ids_i,:],
        positions[:,angle_ids_j,:],
        positions[:,angle_ids_k,:],
    )
    angle_t0, angle_k = _get_angle_params(topology["angles"]["params"])
    # print("angles:", angles.shape)
    angle_energy = torch.square(angles - angle_t0)*angle_k
    angle_energy = angle_energy.sum(dim=1)

    # Proper dihedral energies (from force field).
    proper_ids_i = topology["proper_dihedrals"]["ids"][:,0]
    proper_ids_j = topology["proper_dihedrals"]["ids"][:,1]
    proper_ids_k = topology["proper_dihedrals"]["ids"][:,2]
    proper_ids_l = topology["proper_dihedrals"]["ids"][:,3]
    p_dihedrals = calc_dihedrals(
        positions[:,proper_ids_i,:],
        positions[:,proper_ids_j,:],
        positions[:,proper_ids_k,:],
        positions[:,proper_ids_l,:],
    )
    prd_pe, prd_ph, prd_k = _get_torsion_params(
        topology["proper_dihedrals"]["params"]
    )
    proper_dihedral_energy = cos_potential(
        angles=p_dihedrals,
        periodicity=prd_pe,  # topology["proper_dihedrals"]["params"][:,0],
        phase=prd_ph,  # topology["proper_dihedrals"]["params"][:,1]
        const=prd_k
    )
    proper_dihedral_energy = proper_dihedral_energy.sum(dim=1)

    # Improper dihedral energies (from force field).
    improper_ids_i = topology["improper_dihedrals"]["ids"][:,0]
    improper_ids_j = topology["improper_dihedrals"]["ids"][:,1]
    improper_ids_k = topology["improper_dihedrals"]["ids"][:,2]
    improper_ids_l = topology["improper_dihedrals"]["ids"][:,3]
    imp_dihedrals = calc_dihedrals(
        positions[:,improper_ids_i,:],
        positions[:,improper_ids_j,:],
        positions[:,improper_ids_k,:],
        positions[:,improper_ids_l,:],
    )
    imd_pe, imd_ph, imd_k = _get_torsion_params(
        topology["improper_dihedrals"]["params"]
    )
    improper_dihedral_energy = cos_potential(
        angles=imp_dihedrals,
        periodicity=imd_pe,  # topology["improper_dihedrals"]["params"][:,0],
        phase=imd_ph,  # topology["improper_dihedrals"]["params"][:,1]
        const=imd_k
    )
    improper_dihedral_energy = improper_dihedral_energy.sum(dim=1)
    
    # Phi-psi angles.
    phi_psi_angles = calc_dihedrals(
        positions[:,topology["phi_psi"]["ids"][:,0],:],
        positions[:,topology["phi_psi"]["ids"][:,1],:],
        positions[:,topology["phi_psi"]["ids"][:,2],:],
        positions[:,topology["phi_psi"]["ids"][:,3],:],
        dim=2
    )
    phi_psi_energy = cos_potential(
        angles=phi_psi_angles,
        periodicity=1,
        phase=topology["phi_psi"]["params"],
        const=-1
    )
    phi_psi_energy = phi_psi_energy.sum(axis=1)
    print_msg(
        "phi_psi: {}".format(phi_psi_energy.mean()), verbose=verbose, tag=msg_tag
    )

    # Chi angles.
    chi_angles = calc_dihedrals(
        positions[:,topology["chi"]["ids"][:,0],:],
        positions[:,topology["chi"]["ids"][:,1],:],
        positions[:,topology["chi"]["ids"][:,2],:],
        positions[:,topology["chi"]["ids"][:,3],:],
        dim=2
    )
    chi_energy = cos_potential(
        angles=chi_angles,
        periodicity=1,
        phase=topology["chi"]["params"],
        const=-1
    )
    chi_energy = chi_energy.sum(axis=1)
    print_msg("chi: {}".format(chi_energy.mean()), verbose=verbose, tag=msg_tag)

    # Ca-Ca bond lengths.
    if cabl_const is None:
        _cabl_const = 0.0
        cabl_energy = 0.0
    else:
        _cabl_const = cabl_const
        cabl_ids_i = topology["cabl"]["ids"][:,0]
        cabl_ids_j = topology["cabl"]["ids"][:,1]
        cabl_vals = calc_bond_length(
            pos_i=positions[:,cabl_ids_i,:], pos_j=positions[:,cabl_ids_j,:]
        )
        cabl_energy = torch.square(cabl_vals - topology["cabl"]["params"])
        if "mask" in topology["cabl"]:
            cabl_energy = cabl_energy*topology["cabl"]["mask"]
        cabl_energy = cabl_energy.sum(dim=1)
        print_msg(
            "cabl: {}".format(cabl_energy.mean()), verbose=verbose, tag=msg_tag
        )

    # Non-bonded interactions (repulsion).
    nb_ids = topology["nb_cache"]["ids"]
    # nb_distances = calc_distances(positions, topology["nb_centers"]["ids"], eps=eps)
    nb_distances = calc_distances(positions, nb_ids, eps=eps)
    print_msg(
        "min_nb_distance: {}".format(nb_distances.min()),
        verbose=verbose, tag=msg_tag
    )
    # print("num_clashes:", torch.sum(nb_distances < 0.35, axis=1).float().mean(0))
    print_msg(
        "num_heavy_clashes: {}".format(torch.sum(nb_distances < (0.35/2), axis=1).float().mean(0)),
        verbose=verbose, tag=msg_tag
    )
    
    # Early stopping if contacts are removed.
    if early_stopping_hc_score is not None:
        hc_score = torch.mean(
            torch.sum(
                nb_distances <= early_stopping_hc_thresh, dim=1
            ).float(),
            dim=0
        )
        if hc_score <= early_stopping_hc_score:
            raise MinimizerEarlyStopping(
                score=hc_score, thresh=early_stopping_hc_score
            )

    nb_params = topology["nb_cache"]["params"]
    if nb_form == "l2":
        nb_func = torch.square
    elif nb_form == "l1":
        nb_func = torch.abs
    else:
        raise KeyError(nb_form)
    nb_energy = nb_func(
        torch.clip(nb_params - nb_distances, min=0)
    ).sum(axis=1)
    
    # Get the total energy.
    tot_energy = (
        bond_energy*bond_const +
        angle_energy*angle_const +
        # NOTE: proper and improper increase Ca-Ca clashes.
        proper_dihedral_energy*proper_dihedral_const +  # 1
        improper_dihedral_energy*improper_dihedral_const +  # 1
        phi_psi_energy*phi_psi_const + 
        chi_energy*chi_const + # 1
        cabl_energy*_cabl_const + 
        nb_energy*nb_const
    )
    return tot_energy.sum(dim=0)


class MinimizerEarlyStopping(Exception):
    def __init__(self, message="", score=None, thresh=None):
        super().__init__(message)  # Call the base Exception class constructor
        self.score = score
        self.thresh = thresh


def calc_angles_(A, B, C, eps=1e-12):
    # ix01 = angle_indices[:, [1, 0]]
    # ix21 = angle_indices[:, [1, 2]]

    u_prime = A - B  # xyz[:,ix01[:,1]]-xyz[:,ix01[:,0]]
    v_prime = C - B  # xyz[:,ix21[:,1]]-xyz[:,ix01[:,0]]
    u_norm = torch.sqrt((u_prime**2).sum(-1) + eps)
    v_norm = torch.sqrt((v_prime**2).sum(-1) + eps)

    # adding a new axis makes sure that broasting rules kick in on the third
    # dimension
    u = u_prime / (u_norm[..., None])
    v = v_prime / (v_norm[..., None])

    return torch.arccos((u * v).sum(-1))

def calc_dihedrals(A, B, C, D, dim=2):
    b0 = -(B - A)
    b1 = C - B
    b2 = D - C
    b0xb1 = torch.cross(b0, b1, dim=dim)  ##
    b1xb2 = torch.cross(b2, b1, dim=dim)  ##
    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2, dim=dim)  ##
    y = torch.sum(b0xb1_x_b1xb2*b1, axis=dim)*(1.0/torch.linalg.norm(b1, dim=dim))
    x = torch.sum(b0xb1*b1xb2, axis=dim)
    angle = torch.atan2(y, x)
    return angle

def calc_distances(positions, pair_ids, eps=1e-12):
    distances = torch.sqrt(
        torch.sum(
            torch.square(
                positions[:,pair_ids[:,0],:] - positions[:,pair_ids[:,1],:]
            ), dim=2
        ) + eps
    )
    return distances

def calc_bond_length(pos_i, pos_j):
    bl = torch.sqrt(torch.sum(torch.square(pos_i - pos_j), dim=2))
    return bl

def cos_potential(angles, periodicity, phase, const=1):
    return const*(1 + torch.cos(periodicity*angles - phase))


#####################################
# Initialize systems and positions. #
#####################################

def initialize(positions, a):
    atom14 = get_atom14_mask(a)[0]
    # print("atom14:", atom14.shape)

    atom14 = atom14 == 1
    atom14 = atom14.ravel()
    # print("atom14:", atom14.shape)
    N, L, _, _ = positions.shape
    positions = positions.view(N, L*14, 3)
    positions = positions[:,atom14,:]
    # print(positions[0])
    # print("positions:", positions.shape)

    positions = torch.autograd.Variable(positions)  # .clone()
    positions.requires_grad = True
    # print(positions.shape)
    # energy = calc_energy(
    #     positions=positions, topology=topology
    # )
    # energy.backward()
    return positions


#############
# Minimize. #
#############

def minimize(
        positions,
        topology,
        opt: str = "lbfgs",
        step_size: float = 1.0,
        steps: int = 20,
        max_iter: int = 10,  # lbfgs only.
        history_size: int = 100,  # lbfgs only.
        beta1: float = 0.9,  # adam only.
        beta2: float = 0.999,  # adam only.
        nb_centers_threshold: float = 1.0,
        nb_update_freq: int = 10,
        cabl_init_range: Tuple[float] = None,
        energy_params: dict = {},
        gradient_clip: float = None,
        gradient_clip_mode: str = "value",
        eps: float = 1e-12,
        return_early_stopping: bool = False,
        verbose: int = 1,
    ):
    
    if isinstance(verbose, bool):
        verbose = int(verbose)

    print_msg("# Starting minimization", verbose=verbose>0, tag=msg_tag)

    if opt == "sgd":
        optimizer = torch.optim.SGD([positions], lr=step_size, momentum=0.9)
    elif opt == "gd":
        optimizer = torch.optim.SGD([positions], lr=step_size, momentum=0.0)
    elif opt == "adam":
        optimizer = torch.optim.Adam(
            [positions],
            betas=(beta1, beta2),
            lr=step_size
        )
    elif opt == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [positions],
            lr=step_size,
            max_iter=max_iter,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=history_size,
            line_search_fn=None
        )
    else:
        raise KeyError(opt)

    # Compute initial phi-psi.
    init_positions = positions.detach()
    phi_psi_init = calc_dihedrals(
        init_positions[:,topology["phi_psi"]["ids"][:,0],:],
        init_positions[:,topology["phi_psi"]["ids"][:,1],:],
        init_positions[:,topology["phi_psi"]["ids"][:,2],:],
        init_positions[:,topology["phi_psi"]["ids"][:,3],:],
        dim=2
    )
    topology["phi_psi"]["params"] = phi_psi_init
    # Compute initial chi angles.
    chi_init = calc_dihedrals(
        init_positions[:,topology["chi"]["ids"][:,0],:],
        init_positions[:,topology["chi"]["ids"][:,1],:],
        init_positions[:,topology["chi"]["ids"][:,2],:],
        init_positions[:,topology["chi"]["ids"][:,3],:],
        dim=2
    )
    topology["chi"]["params"] = chi_init
    # Compute initial Ca-Ca bond lengths.
    if energy_params.get("cabl_const") is not None:
        cabl_init = calc_bond_length(
            pos_i=init_positions[:,topology["cabl"]["ids"][:,0],:],
            pos_j=init_positions[:,topology["cabl"]["ids"][:,1],:],
        )
        topology["cabl"]["params"] = cabl_init
        if cabl_init_range is not  None:
            topology["cabl"]["mask"] = (cabl_init > cabl_init_range[0]) & \
                                       (cabl_init < cabl_init_range[1])
            topology["cabl"]["mask"] = topology["cabl"]["mask"].float()

    def closure():
        optimizer.zero_grad()
        print_msg(f"# opt step {i}", verbose=verbose>1, tag=msg_tag)  ###
        
        energy_t = calc_energy(
            positions=positions,
            topology=topology,
            eps=eps,
            verbose=verbose>1,
            **energy_params,
        )
        print_msg(f"energy: {energy_t.item()}", verbose=verbose>1, tag=msg_tag)  ###

        energy_t.backward()
        if gradient_clip is not None:
            if gradient_clip_mode == "norm":
                torch.nn.utils.clip_grad_norm_(positions, gradient_clip)
            elif gradient_clip_mode == "value":
                torch.nn.utils.clip_grad_value_(positions, gradient_clip)
            else:
                raise KeyError(gradient_clip_mode)
        print_msg(
            "grad: min={}, mean={}, max={}".format(
                positions.grad.min(),
                positions.grad.mean(),
                positions.grad.max()
            ),
            verbose=verbose>1, tag=msg_tag
        )  ###

        return energy_t

    early_stopped = False
    for i in range(steps):
        t0 = time.time()
        ###########
        if i % nb_update_freq == 0:
            print_msg("updating nb list", verbose=verbose>1, tag=msg_tag)
            nbce_ids = topology["nb_centers"]["ids"]
            nbce_dmap = calc_distances(positions, nbce_ids, eps=eps)
            nbce_mask = nbce_dmap < nb_centers_threshold
            if nbce_mask.sum() == 0:
                raise ValueError
            nbce_mask = nbce_mask.sum(dim=0) >= 1
            sel_nbce_ids_i = nbce_ids[nbce_mask,0].cpu().numpy()
            sel_nbce_ids_j = nbce_ids[nbce_mask,1].cpu().numpy()
            nb_ids = []
            if topology["nb_config"]["use_os"]:
                nb_params = []
            for sel_i, sel_j in zip(sel_nbce_ids_i, sel_nbce_ids_j):
                sel_ij = topology["nb_centers"]["map_ids"][(sel_i, sel_j)]
                nb_ids.extend(sel_ij)
                if topology["nb_config"]["use_os"]:
                    par_ij = topology["nb_centers"]["map_params"][(sel_i, sel_j)]
                    nb_params.extend(par_ij)
            nb_ids = torch.tensor(nb_ids, dtype=torch.long)
            topology["nb_cache"]["ids"] = nb_ids
            if topology["nb_config"]["use_os"]:
                nb_params = torch.tensor(nb_params, device=positions.device)
                topology["nb_cache"]["params"] = nb_params
            print_msg(
                f"nb_list: {nb_ids.shape}", verbose=verbose>1, tag=msg_tag
            )
        else:
            if not "ids" in topology["nb_cache"]:
                raise ValueError()
        ###########
        try:
            optimizer.step(closure)
        except MinimizerEarlyStopping as e:
            early_stopped = True
            print_msg("- Early stopping", verbose=verbose>0, tag=msg_tag)
            break
        print_msg(f"It took {time.time()-t0}", verbose=verbose>1, tag=msg_tag)

    print_msg("- Completed", verbose=verbose>0, tag=msg_tag)

    positions = positions.detach()
    if not return_early_stopping:
        return positions
    else:
        return positions, early_stopped


##################################
# Reconstruct atom14 trajectory. #
##################################

def reconstruct_atom14(positions, topology):
    N = positions.shape[0]
    L = len(topology["seq"])
    atom14_rec = torch.zeros(N, L, 14, 3)
    for m, aa in enumerate(topology["seq"]):
        atom14_m = ofo_restype_name_to_atom14_names[aa_one_to_three_dict[aa]]
        for l, atm_l in enumerate(atom14_m):
            if atm_l:
                atm_l_idx = topology["atoms_dict"][(m, atm_l)]
                atom14_rec[:,m,l,:] = positions[:,atm_l_idx,:]
    atom14_rec = atom14_rec*10.0
    return atom14_rec