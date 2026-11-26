import os
import math
import tempfile
import pathlib
from contextlib import contextmanager
import subprocess
import numpy as np
import mdtraj
from sam.data.sequences import chi_angles_atoms
from sam.trajectory import calc_mdtraj_rmsf
from sam.minimizer import openstructure_data


def score_pcc_ca_rmsf(ref_ca_traj, hat_ca_traj, ini_ca_traj):

    ref_rmsf = calc_mdtraj_rmsf(traj=ref_ca_traj, ref_traj=ini_ca_traj)
    hat_rmsf = calc_mdtraj_rmsf(traj=hat_ca_traj, ref_traj=ini_ca_traj)

    return pearson_correlation(ref_rmsf, hat_rmsf)


def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient between two 1D arrays.
    """
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")
    x = np.asarray(x)
    y = np.asarray(y)
    # Compute mean-centered arrays.
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    if denominator == 0:
        return np.nan  # Return NaN if division by zero occurs (constant arrays).
    return numerator / denominator


def calc_kld_for_jsd(x, m):
    non_zero = x > 0
    if non_zero.sum() == 0:
        raise ValueError()
    return np.sum(x[non_zero]*np.log(x[non_zero]/m[non_zero]))

def calc_jsd(p_h, q_h):
    m_h = 0.5*(p_h + q_h)
    kld_pm = calc_kld_for_jsd(p_h, m_h)
    kld_qm = calc_kld_for_jsd(q_h, m_h)
    jsd_pq = 0.5*(kld_pm + kld_qm)
    return jsd_pq


res_list = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL"
]

chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

chi_dict = {}
for n, t in (("chi_12", 2), ("chi_23", 3), ("chi_34", 4)):
    chi_dict[n] = []
    for i, res in enumerate(res_list):
        if sum(chi_angles_mask[i]) >= t:
            chi_dict[n].append(res)


def _get_chi_atoms(ref_res, hat_res, chi_ids):
    ref_atoms_a = [a_j.index for a_i in chi_angles_atoms[ref_res.name][chi_ids[0]] for a_j in ref_res.atoms_by_name(a_i)]
    if len(ref_atoms_a) != 4:
        raise ValueError()
    ref_atoms_b = [a_j.index for a_i in chi_angles_atoms[ref_res.name][chi_ids[1]] for a_j in ref_res.atoms_by_name(a_i)]
    if len(ref_atoms_b) != 4:
        raise ValueError()
    hat_atoms_a = [a_j.index for a_i in chi_angles_atoms[hat_res.name][chi_ids[0]] for a_j in hat_res.atoms_by_name(a_i)]
    if len(hat_atoms_a) != 4:
        raise ValueError()
    hat_atoms_b = [a_j.index for a_i in chi_angles_atoms[hat_res.name][chi_ids[1]] for a_j in hat_res.atoms_by_name(a_i)]
    if len(hat_atoms_b) != 4:
        raise ValueError()
    return {
        "ref": [ref_atoms_a, ref_atoms_b], "hat": [hat_atoms_a, hat_atoms_b]
    }
    
def _process_chijsd_residue(
        ref_traj, hat_traj, sel_atoms, verbose=False
    ):
    ref_chi_2d = mdtraj.compute_dihedrals(ref_traj, sel_atoms["ref"])
    hat_chi_2d = mdtraj.compute_dihedrals(hat_traj, sel_atoms["hat"])

    shift = np.deg2rad(120)
    ref_chi_shift = (ref_chi_2d + shift) % (2 * np.pi)
    hat_chi_shift = (hat_chi_2d + shift) % (2 * np.pi)
    chi_ticks = [np.deg2rad(a) for a in [0, 120, 240, 360]]
    ref_hist = np.histogram2d(
        ref_chi_shift[:,0], ref_chi_shift[:,1], bins=[chi_ticks, chi_ticks]
    )[0]
    hat_hist = np.histogram2d(
        hat_chi_shift[:,0], hat_chi_shift[:,1], bins=[chi_ticks, chi_ticks]
    )[0]
    ref_p = ref_hist.ravel()/ref_hist.sum()
    hat_p = hat_hist.ravel()/hat_hist.sum()
    
    jsd = calc_jsd(ref_p, hat_p)
    # for r, h in zip(ref_p, hat_p):
    #     print(f"{r}-{h}", end="; ")
    # print("")
    # print(jsd)

    return jsd
    

def score_chiJSD(ref_traj, hat_traj, get_res_ids=False):
    ref_seq = "".join([r.code for r in ref_traj.topology.residues])
    hat_seq = "".join([r.code for r in hat_traj.topology.residues])
    if ref_seq != hat_seq:
        raise ValueError()
    scores_list = []
    res_idx_list = []
    res_idx = 0
    for ref_res, hat_res in zip(ref_traj.topology.residues, hat_traj.topology.residues):
        if ref_res.name in chi_dict["chi_12"]:
            sel_atoms = _get_chi_atoms(ref_res, hat_res, (0, 1))
            score_k = _process_chijsd_residue(ref_traj, hat_traj, sel_atoms)
            scores_list.append(score_k)
            res_idx_list.append(res_idx)
        if ref_res.name in chi_dict["chi_23"]:
            sel_atoms = _get_chi_atoms(ref_res, hat_res, (1, 2))
            score_k = _process_chijsd_residue(ref_traj, hat_traj, sel_atoms)
            scores_list.append(score_k)
            res_idx_list.append(res_idx)
        if ref_res.name in chi_dict["chi_34"]:
            sel_atoms = _get_chi_atoms(ref_res, hat_res, (2, 3))
            score_k = _process_chijsd_residue(ref_traj, hat_traj, sel_atoms)
            scores_list.append(score_k)
            res_idx_list.append(res_idx)
        res_idx +=1
    if not scores_list:
        raise ValueError()
    if not get_res_ids:
        return np.mean(scores_list)
    else:
        return scores_list, res_idx_list


elements = ["C", "N", "O", "S"]
# ATLAS.
cn_params = {
    "mean": 1.34811,  # new: 0.133038
    "std":  0.02875,  # new: 0.000989
}

c_n_ca_params = {
    "mean": 2.16150,
    "std": 0.05703,
}

o_c_n_params = {
    "mean": 2.11054,
    "std": 0.04895,
}

ca_c_n_params = {
    "mean": 2.05753,
    "std": 0.05144,
}

os_params = np.zeros((5, 5, 2))
for l in openstructure_data.split("\n")[1:]:
    els, thresh, tol = l.split()
    el_1, el_2 = els.split("-")
    os_params[elements.index(el_1)+1, elements.index(el_2)+1, 0] = float(thresh)
    os_params[elements.index(el_2)+1, elements.index(el_1)+1, 0] = float(thresh)
    os_params[elements.index(el_1)+1, elements.index(el_2)+1, 1] = float(tol)
    os_params[elements.index(el_2)+1, elements.index(el_1)+1, 1] = float(tol)


bb_atoms = ("CA", "N", "O", "C", "OXT")

def mstats_stereo(
        traj,
        res_offset=2,
        n_sigma_bond=3.0,
        n_sigma_angle=3.0
        # func="l2",
        # sel="aa"
    ):
    """
    Use it to score mdtraj trajectories.
    """

    ##################
    # Score clashes. #
    ##################

    sel_atoms = []
    is_ca = []
    is_bb = []
    ele = []
    res = []

    for a in traj.topology.atoms:
        if a.element.symbol in elements and a.residue.is_protein:
            sel_atoms.append(a.index)
            ele.append(elements.index(a.element.symbol))
            res.append(a.residue.index)
            is_ca.append(a.name == "CA")
            is_bb.append(a.name in bb_atoms)
    if not sel_atoms:
        raise ValueError()
    sel_atoms = np.array(sel_atoms)
    ele = np.array(ele)+1
    is_ca = np.array(is_ca)
    is_bb = np.array(is_bb)

    pairs = []
    for i in range(len(sel_atoms)):
        for j in range(i+1, len(sel_atoms)):
            if res[i] + res_offset <= res[j]:
                pairs.append([i, j])
    pairs = np.array(pairs)

    sel_pairs = sel_atoms[pairs]
    is_ca_pairs = is_ca[pairs].sum(axis=1) == 2
    is_bb_pairs = is_bb[pairs].sum(axis=1) == 2

    ha_params = os_params[ele[pairs[:,0]], ele[pairs[:,1]], 0]
    bb_params = ha_params[is_bb_pairs]
    ca_params = 3.9

    scores = {
        "clash_ca": [], "clash_bb": [], "clash_ha": [],
        "heavy_clash_bb": [], "heavy_clash_ha": [],
    }
    for k in range(len(traj)):
        dist = mdtraj.compute_distances(traj[k], sel_pairs)*10.0
        dist = dist[0]
        for sel in ("ca", "bb", "ha"):
            if sel == "ca":
                score = dist[is_ca_pairs] < ca_params
                scores["clash_ca"].append(score.sum())
            elif sel == "bb":
                score = dist[is_bb_pairs] < bb_params
                scores["clash_bb"].append(score.sum())
                score = dist[is_bb_pairs] < bb_params*0.5
                scores["heavy_clash_bb"].append(score.sum())
            elif sel == "ha":
                score = dist < ha_params
                scores["clash_ha"].append(score.sum())
                score = dist < ha_params*0.5
                scores["heavy_clash_ha"].append(score.sum())
            else:
                raise KeyError(func)
    
    ####################
    # Score C-N bonds. #
    ####################

    cn_params
    res_data = []
    for res in traj.topology.residues:
        if not res.is_protein:
            raise ValueError()
        try:
            res_data.append({
                "C": res.atom("C").index,
                "N": res.atom("N").index,
                "CA": res.atom("CA").index,
                "O": res.atom("O").index,
            })
        except KeyError:
            raise ValueError()
    
    c_n_ids = [(r_im1["C"], r_i["N"]) for (r_i, r_im1) \
               in zip(res_data[1:], res_data[:-1])]
    c_n = mdtraj.compute_distances(traj, c_n_ids)
    scores["viol_c_n"] = score_violations(
        c_n*10.0, cn_params["mean"], cn_params["std"], n_sigma_bond
    )

    #######################################
    # Score C-N-CA and O-C-N bond angles. #
    #######################################

    c_n_ca_ids = []
    # o_c_n_ids = []
    ca_c_n_ids = []
    for r_i, r_im1 in zip(res_data[1:], res_data[:-1]):
        c_n_ca_ids.append([r_im1["C"], r_i["N"], r_i["CA"]])
        # o_c_n_ids.append([r_im1["O"], r_im1["C"], r_i["N"]])
        ca_c_n_ids.append([r_im1["CA"], r_im1["C"], r_i["N"]])
    c_n_ca = mdtraj.compute_angles(traj, c_n_ca_ids)
    scores["viol_c_n_ca"] = score_violations(
        c_n_ca, c_n_ca_params["mean"], c_n_ca_params["std"], n_sigma_angle
    )
    # o_c_n = mdtraj.compute_angles(traj, o_c_n_ids)
    ca_c_n = mdtraj.compute_angles(traj, ca_c_n_ids)
    scores["viol_ca_c_n"] = score_violations(
        ca_c_n, ca_c_n_params["mean"], ca_c_n_params["std"], n_sigma_angle
    )

    return scores


def score_violations(
        data: np.ndarray, mean: float, std: float, n_sigma: float
    ):
    norm = abs(data - mean)/std
    return np.sum(norm >= n_sigma, axis=1)