import torch
import numpy as np


def calc_dmap(xyz, epsilon=1e-12, backend="torch"):
    if backend == "torch":
        B = torch
    elif backend == "numpy":
        B = np
    else:
        raise KeyError(backend)
    if len(xyz.shape) == 2:
        if xyz.shape[1] != 3:
            raise ValueError(xyz.shape)
    elif len(xyz.shape) == 3:
        if xyz.shape[2] != 3:
            raise ValueError(xyz.shape)
    else:
        raise ValueError(xyz.shape)
    if len(xyz.shape) == 3:
        dmap = B.sqrt(
                 B.sum(
                   B.square(xyz[:,None,:,:] - xyz[:,:,None,:]),
                 axis=3) + epsilon)
        exp_dim = 1
    else:
        dmap = B.sqrt(
                 B.sum(
                   B.square(xyz[None,:,:] - xyz[:,None,:]),
                 axis=2) + epsilon)
        exp_dim = 0
    if backend == "torch":
        return dmap.unsqueeze(exp_dim)
    elif backend == "numpy":
        return np.expand_dims(dmap, exp_dim)
    else:
        raise KeyError(backend)


def calc_dmap_triu(input_data, offset=1, epsilon=1e-12, backend="torch"):
    # Check the shape.
    if len(input_data.shape) == 2:
        if input_data.shape[1] != 3:
            raise ValueError(input_data.shape)
        dmap = calc_dmap(input_data, epsilon, backend)
    elif len(input_data.shape) == 3:
        if input_data.shape[2] != 3:
            raise ValueError(input_data.shape)
        dmap = calc_dmap(input_data, epsilon, backend)
    elif len(input_data.shape) == 4:
        if input_data.shape[1] != 1:
            raise ValueError(input_data.shape)
        if input_data.shape[2] != input_data.shape[3]:
            raise ValueError(input_data.shape)
        dmap = input_data
    else:
        raise ValueError(input_data.shape)
    # Get the triu ids.
    l = dmap.shape[2]
    if backend == "torch":
        triu_ids = torch.triu_indices(l, l, offset=offset)
    elif backend == "numpy":
        triu_ids = np.triu_indices(l, k=offset)
    else:
        raise KeyError(backend)
    # Returns the values.
    if len(input_data.shape) != 2:
        return dmap[:,0,triu_ids[0],triu_ids[1]]
    else:
        return dmap[0,triu_ids[0],triu_ids[1]]


def torch_chain_dihedrals(xyz, backend="torch"):
    if backend == "torch":
        r_sel = xyz
    elif backend == "numpy":
        r_sel = torch.tensor(xyz)
    else:
        raise KeyError(backend)
    b0 = -(r_sel[:,1:-2,:] - r_sel[:,0:-3,:])
    b1 = r_sel[:,2:-1,:] - r_sel[:,1:-2,:]
    b2 = r_sel[:,3:,:] - r_sel[:,2:-1,:]
    b0xb1 = torch.cross(b0, b1, dim=2)  ###
    b1xb2 = torch.cross(b2, b1, dim=2)  ###
    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2, dim=2)  ###
    y = torch.sum(b0xb1_x_b1xb2*b1, axis=2)*(1.0/torch.linalg.norm(b1, dim=2))
    x = torch.sum(b0xb1*b1xb2, axis=2)
    dh_vals = torch.atan2(y, x)
    return dh_vals


def calc_chain_bond_angles(xyz, backend="numpy"):
    ids = np.array([[i, i+1, i+2] for i in range(xyz.shape[1]-2)])
    return calc_angles(xyz, ids, backend=backend)


def calc_angles(xyz, angle_indices, backend="numpy"):
    if backend == "numpy":
        B = np
    elif backend == "torch":
        B = torch
    else:
        raise KeyError(backend)

    ix01 = angle_indices[:, [1, 0]]
    ix21 = angle_indices[:, [1, 2]]

    u_prime = xyz[:,ix01[:,1]]-xyz[:,ix01[:,0]]
    v_prime = xyz[:,ix21[:,1]]-xyz[:,ix01[:,0]]
    u_norm = B.sqrt((u_prime**2).sum(-1))
    v_norm = B.sqrt((v_prime**2).sum(-1))

    # adding a new axis makes sure that broasting rules kick in on the third
    # dimension
    u = u_prime / (u_norm[..., None])
    v = v_prime / (v_norm[..., None])

    return B.arccos((u * v).sum(-1))


def calc_torsion(A, B, C, D, dim=2):
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


def compute_rg(xyz):
    """
    Adapted from the mdtraj library: https://github.com/mdtraj/mdtraj.
    """
    num_atoms = xyz.shape[1]
    masses = np.ones(num_atoms)
    weights = masses / masses.sum()
    mu = xyz.mean(1)
    centered = (xyz.transpose((1, 0, 2)) - mu).transpose((1, 0, 2))
    squared_dists = (centered ** 2).sum(2)
    Rg = (squared_dists * weights).sum(1) ** 0.5
    return Rg


def sample_data(data, n_samples, backend="numpy"):
    if backend in ("numpy", "torch"):
        if n_samples is not None:
            ids = np.random.choice(data.shape[0],
                                   n_samples,
                                   replace=data.shape[0] < n_samples)
            return data[ids]
        else:
            return data
    else:
        raise KeyError(backend)


def get_edge_dmap(xyz, batch, epsilon=1e-12):
    row, col = batch.nr_edge_index
    dmap = torch.sqrt(
             torch.sum(
               torch.square(xyz[row] - xyz[col]),
             axis=1) + epsilon)
    return dmap


def calc_bond_len(x, eps=1e-6):
    return torch.sqrt(torch.sum(torch.square(x[:,:-1,:] - x[:,1:,:]), dim=2) + eps)


def calc_com_traj(positions, atom14_gt_exists, mult=0.1):
    """
    TODO: legacy function, its name is wrong.
    """
    return calc_scen_pos(positions, atom14_gt_exists, mult)

def calc_scen_pos(positions, atom14_gt_exists, mult=0.1):
    """
    Calculate side chain centroid positions from OpenFold data.
    """
    # Get positions.
    pos = positions[:,:,4:,:]*mult  # Side-chain atoms positions.
    ca_pos = positions[:,:,1,:]*mult  # CA positions.
    mask = atom14_gt_exists[:,:,4:].unsqueeze(-1)  # Check if side-chain atoms exists.
    gly_mask = 1-atom14_gt_exists[:,:,4].unsqueeze(-1)  # Check if CB atoms exist.
    mask = mask.to(dtype=pos.dtype)
    gly_mask = gly_mask.to(dtype=pos.dtype)
    # Compute centroid.
    com_pos = (pos.sum(dim=2) + ca_pos*gly_mask)/(mask.sum(dim=2) + gly_mask)
    return com_pos


def calc_aa_cen_traj(positions, atom14_gt_exists, mult=0.1):
    # Get every atom positions.
    pos = positions*mult
    mask = atom14_gt_exists.unsqueeze(-1)
    # Calculate centroid.
    cen_pos = pos.sum(dim=2) / mask.sum(dim=2)
    return cen_pos


def calc_bb_cen_traj(positions, atom14_gt_exists, mult=0.1):
    # Get backbone positions.
    pos = positions[:,:,0:4,:]*mult
    mask = atom14_gt_exists[:,:,0:4].unsqueeze(-1)
    # Calculate centroid.
    cen_pos = pos.sum(dim=2) / mask.sum(dim=2)
    return cen_pos

def calc_nb_dist(xyz, get_triu_ids=False, offset=3, eps=1e-9):
    triu_ids = torch.triu_indices(xyz.shape[1], xyz.shape[1], offset=offset)
    # dmap_triu = torch.sqrt(
    #     torch.square(
    #         xyz[:,triu_ids[0]] - xyz[:,triu_ids[1]]
    #     ).sum(axis=-1) + eps
    # )
    dmap_triu = torch.cdist(xyz, xyz, p=2.0)
    dmap_triu = dmap_triu[:,triu_ids[0],triu_ids[1]]
    if get_triu_ids:
        return dmap_triu, triu_ids
    else:
        return dmap_triu