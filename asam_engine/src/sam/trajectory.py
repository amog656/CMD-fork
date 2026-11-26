"""
Analyze different types of features of mdtraj trajectories.
"""

import numpy as np
import mdtraj
from sam.data.topology import slice_ca_traj
from sam.data.sequences import ofo_restype_name_to_atom14_names


def calc_mdtraj_rmsf(
        traj: mdtraj.Trajectory,
        ref_traj: mdtraj.Trajectory,
        ref_index: int = 0
    ) -> np.ndarray:
    traj_c = mdtraj.Trajectory(traj.xyz, topology=traj.topology)
    ref_traj = mdtraj.Trajectory(ref_traj.xyz, topology=traj.topology)
    rmsf = mdtraj.rmsf(traj_c, ref_traj, ref_index)
    return rmsf


std_atoms = set()
for k in ofo_restype_name_to_atom14_names:
    for a in ofo_restype_name_to_atom14_names[k]:
        if a:
            std_atoms.add(a)


def calc_q_values(
        traj,
        native_traj,
        beta=50.0,
        lambda_=1.2,
        delta=0.0,
        threshold=1.0  # in nanometers.
    ):

    if len(native_traj) != 1:
        raise NotImplementedError()
    
    dist = []
    top_atoms_dict = [[a for a in r.atoms if a.name in std_atoms] for r in native_traj.topology.residues]
    
    top_ids = []
    traj_ids = []
    top_residues = list(native_traj.topology.residues)  #
    top_atoms = list(native_traj.topology.atoms)        #
    traj_residues = list(traj.topology.residues)
    traj_atoms = list(traj.topology.atoms)           #
    for i, r_i in enumerate(native_traj.topology.residues):
        for j, r_j in enumerate(native_traj.topology.residues):
            if r_j.index - r_i.index > 3:
                a_k_ids = [a_k.index for a_k in top_atoms_dict[i]]
                a_k_atoms = top_atoms_dict[i]
                a_l_ids = [a_l.index for a_l in top_atoms_dict[j]]
                a_l_atoms = top_atoms_dict[j]
                dist_ij = np.sqrt(
                    np.sum(
                        np.square(
                            native_traj.xyz[:,a_k_ids,None,:] - native_traj.xyz[:,None,a_l_ids,:]
                        ),
                    axis=-1
                    )
                )[0]
                a_k_pos, a_l_pos = np.unravel_index(dist_ij.argmax(), dist_ij.shape)
                if threshold is not None:
                    if dist_ij[a_k_pos, a_l_pos] > threshold:
                        continue
                top_ids.append((a_k_ids[a_k_pos], a_l_ids[a_l_pos]))
                traj_ids.append(
                    (traj_residues[i].atom(a_k_atoms[a_k_pos].name).index,
                     traj_residues[j].atom(a_l_atoms[a_l_pos].name).index)
                )
                
    if not traj_ids:
        raise ValueError()
    ref_dist = mdtraj.compute_distances(native_traj, top_ids)
    traj_dist = mdtraj.compute_distances(traj, traj_ids)
    n_contacts = ref_dist.shape[1]
    q_x = np.sum(1/(1+np.exp(beta*(traj_dist - lambda_*(ref_dist + delta)))), axis=1)/n_contacts
    return q_x


def calc_rmsd(hat_traj, ref_traj, ref_idx=0, prealigned=True, get_tm=False):
    ref_traj = ref_traj[ref_idx:ref_idx+1]
    if not prealigned:
        raise NotImplementedError()
    ref_xyz = ref_traj.xyz * 10.0
    hat_xyz = hat_traj.xyz * 10.0
    sq_dev = np.sum(np.square(ref_xyz - hat_xyz), axis=-1)
    rsq_dev = np.sqrt(sq_dev)
    n_res = ref_xyz.shape[1]
    rmsd = np.sqrt(sq_dev.sum(axis=1)/n_res)
    tm_score = (1 / n_res) * np.sum(1 / (1 + (rsq_dev/d_0(n_res))**2), axis=1)
    if not get_tm:
        return rmsd*0.1
    else:
        return tm_score, rmsd*0.1

def d_0(n_res):
    return 1.24*(n_res - 15)**(1/3) - 1.8
    
def calc_initrmsd(traj, init_traj, is_ca=False, get_tm=False):
    if not is_ca:
        init_traj = slice_ca_traj(init_traj)
        traj = slice_ca_traj(traj)
    traj.superpose(init_traj)
    scores = calc_rmsd(
        traj, init_traj, ref_idx=0, prealigned=True, get_tm=get_tm
    )
    return scores

def calc_ssep(traj, native_traj):
    if len(native_traj) > 1:
        raise ValueError()
    ref_dssp = mdtraj.compute_dssp(native_traj)
    traj_dssp = mdtraj.compute_dssp(traj)
    match = ref_dssp == traj_dssp
    match = match.astype(int)
    score = match.mean(axis=1)
    he_mask = np.isin(ref_dssp, ['H', 'E']).astype(int)
    match_mask = match * he_mask
    score_mask = match_mask.sum(axis=1)/he_mask.sum()
    return score_mask

def _calc_distances(xyz, eps=1e-9):
    d = np.sqrt(
        np.sum(np.square(xyz[:,:,None,:] - xyz[:,None,:,:]), axis=3)+eps
    )
    return d