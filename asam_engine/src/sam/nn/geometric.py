"""
PyTorch modules for performing some geometrical operations in neural networks.
"""

import math
from typing import Union, Callable
import torch
import torch.nn as nn

from sam.coords import torch_chain_dihedrals, calc_dmap, calc_torsion
from sam.nn.common import get_act_fn, FeedForward


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        delta = offset[1] - offset[0]
        coeff = -0.5 / delta.item()**2
        if not trainable:
            self.coeff = coeff
            self.register_buffer('offset', offset)
            self.offset = self.offset.view(1, -1, 1, 1)
        else:
            offset = offset.view(1, -1, 1, 1)
            self.register_parameter("offset", nn.Parameter(offset))
            coeff = torch.full((offset.shape[0], ), coeff).view(1, -1, 1, 1)
            self.register_parameter("coeff", nn.Parameter(coeff))

    def forward(self, dist):
        diff = dist - self.offset
        # print("offset:", self.offset.shape, self.coeff)
        # print("diff:", diff.shape)
        return torch.exp(self.coeff * torch.pow(diff, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        r = self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
        return r.transpose(1, 4).squeeze(4)
    

class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class SinusoidalDistanceEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Adapted from diffusion transformer: https://github.com/facebookresearch/DiT
    """
    def __init__(self,
            hidden_size,
            frequency_embedding_size=256,
            activation="silu",
            max_period=10000,
            mult=50
        ):
        super().__init__()
        self.mlp = FeedForward(
            in_dim=frequency_embedding_size,
            hidden_dim=hidden_size,
            out_dim=hidden_size,
            activation=activation,
            linear_bias=True
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        # max_period*mult should be approximately equal to the maximum
        # distance in the training set.
        self.mult = mult

    @staticmethod
    def timestep_embedding(d, dim, mult=50, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param d: a (B, 1, L, L) Tensor of of float values.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (B, D, L, L) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=d.device)
        args = d * freqs.view(1, -1, 1, 1) * mult
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if dim % 2:
            # embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            raise NotImplementedError()
        return embedding

    def forward(self, d):
        """
        d: (B, 1, L, L)
        out: (B, L, L, E)
        """
        d_freq = self.timestep_embedding(
            d, self.frequency_embedding_size,
            mult=self.mult, max_period=self.max_period
        )
        d_freq = d_freq.transpose(1, 3).transpose(1, 2)
        d_emb = self.mlp(d_freq)
        return d_emb


default_msde_bins = [
    [0.0, 0.5],
    [0.5, 1.0],
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0],
    [6.0, 7.0],
    [7.0, 8.0],
    [8.0, 9.0],
    [9.0, 10.0],
    [10.0, 11.0],
    [11.0, 12.0],
    [12.0, None],
]

class MostSimpleDistanceEmbedder(nn.Module):
    
    def __init__(self, bins=None, trainable=False):
        super().__init__()
        if bins is None:
            bins = default_msde_bins
        _bins = []
        for b_il, b_ih in bins:
            if b_il is None:
                raise ValueError()
            if b_ih is not None:
                _b_il = b_il
                _b_ih = b_ih
            else:
                _b_il = 0.0
                _b_ih = b_il
            _bins.append([_b_il, _b_ih])
        # (N, 2)
        _bins = torch.tensor(_bins)
        self.n_bins = _bins.shape[0]
        # (1, 1, 1, N)
        bins_l = _bins[:,0].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        bins_h = _bins[:,1].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if trainable:
            self.register_parameter("bins_l", nn.Parameter(bins_l))
            self.register_parameter("bins_h", nn.Parameter(bins_h))
        else:
            self.register_buffer('bins_l', bins_l)
            self.register_buffer('bins_h', bins_h)
        self._trainable = trainable
    
    def forward(self, d):
        """
        d: (B, 1, L, L)
        e_d: (B, L, L, E)
        """
        d = d.squeeze(1).unsqueeze(-1)  # (B, L, L, 1)
        low = self.bins_l
        high = self.bins_h
        diff = high - low
        d_p = nn.functional.relu(d - low)
        d_norm = d_p / diff
        if not self._trainable:
            d_norm[...,:-1] = torch.clamp(d_norm[...,:-1], max=1.0)
        else:
            d_norm = torch.clamp(d_norm, max=1.0)
        # (B, L, L, E)
        return d_norm


class BeadDistanceEmbedding(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            edge_dim: int,
            embed_type: str,
            activation: Union[Callable, str],
            params: dict,
            use_project: bool = True,
            use_mlp: bool = False,
            input_type: str = "xyz"
        ):
        """Arguments: TODO."""

        super().__init__()

        # Embed distances.
        if embed_type == "rbf":
            self.dmap_expansion = GaussianSmearing(
                start=params["start"],
                stop=params["stop"],
                num_gaussians=params["num_gaussians"],
                trainable=params.get("trainable", False)
            )
            dmap_dim = params["num_gaussians"]
        elif embed_type == "expnorm":
            self.dmap_expansion = ExpNormalSmearing(
                cutoff_lower=params["cutoff_lower"],
                cutoff_upper=params["cutoff_upper"],
                num_rbf=params["num_rbf"],
                trainable=params["trainable"]
            )
            dmap_dim = params["num_rbf"]
        elif embed_type == "sinusoidal":
            self.dmap_expansion = SinusoidalDistanceEmbedder(
                hidden_size=params["hidden_size"],
                frequency_embedding_size=params["frequency_embedding_size"],
                activation=activation,
                max_period=10000,
                mult=params["mult"]
            )
            dmap_dim = params["hidden_size"]
        elif embed_type == "msde":
            self.dmap_expansion = MostSimpleDistanceEmbedder(
                bins=params["bins"],
                trainable=params["trainable"]
            )
            dmap_dim = self.dmap_expansion.n_bins
        else:
            raise KeyError(embed_type)
        self.embed_type = embed_type
        self.dmap_dim = dmap_dim

        if not use_mlp:
            if use_project:
                self.project_dmap = nn.Sequential(nn.Linear(dmap_dim, edge_dim))
            else:
                self.project_dmap = nn.Identity()
        else:
            if not use_project:
                raise ValueError()
            self.project_dmap = FeedForward(
                in_dim=dmap_dim,
                hidden_dim=edge_dim,
                out_dim=edge_dim,
                activation=activation,
                linear_bias=True,
            )
        
        if not input_type in ("xyz", "dmap"):
            raise KeyError(input_type)
        self.input_type = input_type

    def forward(self, x, get_dmap=False):
        if self.input_type == "xyz":
            dmap = calc_dmap(x)
        elif self.input_type == "dmap":
            dmap = x
        else:
            raise KeyError(input_type)
        if not len(dmap.shape) == 4:
            raise ValueError(dmap.shape)
        if not dmap.shape[1] == 1:
            raise ValueError(dmap.shape)
        if not dmap.shape[2] == dmap.shape[3]:
            raise ValueError(dmap.shape)
        expansion = self.dmap_expansion(dmap)
        if self.embed_type in ("rbf", "expnorm"):
            expansion = expansion.transpose(1, 3)
        z = self.project_dmap(expansion)
        if get_dmap:
            return z, dmap
        else:
            return z


def get_chain_torsion_features(x):
    t = torch_chain_dihedrals(x).unsqueeze(2)
    t_f = torch.cat([torch.cos(t), torch.sin(t), torch.ones_like(t)], dim=2)
    t_f = nn.functional.pad(t_f, (0, 0, 1, 2))
    return t_f


def featurize_angle(angle, pad, bins=None, mask_class="empty"):
    if bins is None:
        angle = angle.unsqueeze(2)
        angle_f = torch.cat(
            [torch.cos(angle), torch.sin(angle), torch.ones_like(angle)],
            dim=2
        )
    else:
        # Step 1: Define the bins
        bin_edges = torch.linspace(
            -math.pi, math.pi, bins+1, device=angle.device
        )
        # Step 2: Digitize the float values
        bin_indices = torch.clamp(
            torch.bucketize(angle, bin_edges, right=True) - 1, min=0, max=bins-1
        )
        # Step 3: One-hot encode the bin indices
        one_hot_encoded = torch.nn.functional.one_hot(
            bin_indices, num_classes=bins
        )
        if mask_class == "empty":
            angle_f = one_hot_encoded
        elif mask_class == "extra":
            angle_f = torch.cat(
                [one_hot_encoded, torch.ones_like(angle.unsqueeze(2))], dim=2
            )
        else:
            raise KeyError(mask_class)
    angle_f = nn.functional.pad(angle_f, (0, 0, pad[0], pad[1]))
    return angle_f


######################
# 2d angular features.
#

class TrAnglesEmbedding(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            edge_dim: int,
            activation: Union[Callable, str] = None,
            use_theta: bool = True,
            use_phi: bool = True,
            use_omega: bool = True,
            use_project: bool = True,
            use_mlp: bool = False,
            linear_bias: bool = True
        ):
        """Arguments: TODO."""
        super().__init__()

        self.use_theta = use_theta
        self.use_phi = use_phi
        self.use_omega = use_omega

        if not any((use_theta, use_phi, use_omega)):
            raise ValueError()

        amap_dim = 0
        if use_theta:
            amap_dim += 4
        if use_phi:
            amap_dim += 4
        if use_omega:
            amap_dim += 2
        self.amap_dim = amap_dim
    
        if use_project:
            if use_mlp:
                self.project_amap = FeedForward(
                    in_dim=amap_dim,
                    hidden_dim=edge_dim,
                    out_dim=edge_dim,
                    activation=activation,
                    linear_bias=linear_bias,
                )
            else:
                self.project_amap = nn.Linear(
                    amap_dim, edge_dim, bias=linear_bias
                )
        else:
            self.project_amap = nn.Identity()


    def forward(self, batch):
        expansion = calc_tr_angles(
            batch=batch,
            theta=self.use_theta,
            phi=self.use_phi,
            omega=self.use_omega
        )
        z = self.project_amap(expansion)
        return z


def calc_tr_angles(
        batch, theta=True, phi=True, omega=True,
        featurize=True
    ):

    if not any([theta, phi, omega]):
        raise ValueError()

    # Calculate pseudo Cb atoms positions.
    ca_pos = batch.atom14_gt_positions[:,:,1,:]
    n_pos = batch.atom14_gt_positions[:,:,0,:]
    c_pos = batch.atom14_gt_positions[:,:,2,:]
    # cb_pos = batch.atom14_gt_positions[:,:,4,:]

    # recreate Cb given N,Ca,C
    b = ca_pos - n_pos
    c = c_pos - ca_pos
    a = torch.cross(b, c, dim=-1)
    ma = 0.58273431
    mb = 0.56802827
    mc = 0.54067466
    pcb_pos = -ma*a + mb*b - mc*c + ca_pos
    # d = torch.square(cb_pos - pcb_pos).sum(axis=-1).sqrt()
    cb_pos = pcb_pos

    # Prepare for 2d feature calculation.
    ca_pos_i = ca_pos.unsqueeze(2).repeat(1, 1, ca_pos.shape[1], 1)
    ca_pos_j = ca_pos.unsqueeze(1).repeat(1, ca_pos.shape[1], 1, 1)
    n_pos_i = n_pos.unsqueeze(2).repeat(1, 1, n_pos.shape[1], 1)
    n_pos_j = n_pos.unsqueeze(1).repeat(1, n_pos.shape[1], 1, 1)
    cb_pos_i = cb_pos.unsqueeze(2).repeat(1, 1, cb_pos.shape[1], 1)
    cb_pos_j = cb_pos.unsqueeze(1).repeat(1, cb_pos.shape[1], 1, 1)

    diag_ids = torch.arange(0, ca_pos.shape[1], device=ca_pos.device)

    # Prepare output.
    if featurize:
        out = []
    else:
        # out = {}
        raise NotImplementedError()

    # Calculate theta torsion (asymmetric).
    if theta:
        tr_theta_12 = calc_torsion(
            A=n_pos_i, B=ca_pos_i, C=cb_pos_i, D=cb_pos_j, dim=-1
        )
        tr_theta_21 = calc_torsion(
            A=n_pos_j, B=ca_pos_j, C=cb_pos_j, D=cb_pos_i, dim=-1
        )
        if featurize:
            tr_theta = torch.cat([
                torch.cos(tr_theta_12.unsqueeze(-1)),
                torch.sin(tr_theta_12.unsqueeze(-1)),
                torch.cos(tr_theta_21.unsqueeze(-1)),
                torch.sin(tr_theta_21.unsqueeze(-1))
            ], dim=-1)
            tr_theta[:,diag_ids,diag_ids,:] = 0
            out.append(tr_theta)
        else:
            raise NotImplementedError()
    
    # Calculate phi angle (asymmetric).
    if phi:
        tr_phi_12 = calc_angles_(
            A=ca_pos_i, B=cb_pos_i, C=cb_pos_j
        )
        tr_phi_21 = calc_angles_(
            A=ca_pos_j, B=cb_pos_j, C=cb_pos_i
        )
        if featurize:
            tr_phi = torch.cat([
                torch.cos(tr_phi_12.unsqueeze(-1)),
                torch.sin(tr_phi_12.unsqueeze(-1)),
                torch.cos(tr_phi_21.unsqueeze(-1)),
                torch.sin(tr_phi_21.unsqueeze(-1))
            ], dim=-1)
            tr_phi[:,diag_ids,diag_ids,:] = 0
            out.append(tr_phi)
        else:
            raise NotImplementedError()

    # Calculate omega torsion (symmetric).
    if omega:
        tr_omega = calc_torsion(
            A=ca_pos_i, B=cb_pos_i, C=cb_pos_j, D=ca_pos_j, dim=-1
        )
        if featurize:
            tr_omega = torch.cat([
                torch.cos(tr_omega.unsqueeze(-1)),
                torch.sin(tr_omega.unsqueeze(-1))
            ], dim=-1)
            tr_omega[:,diag_ids,diag_ids,:] = 0
            out.append(tr_omega)
        else:
            raise NotImplementedError()
    
    # Return results.
    if featurize:
        out = torch.cat(out, axis=-1)
    else:
        raise NotImplementedError()
    
    return out


#-------
# Alpha.

def duplicate(x, pos):
    # x: Tensor of shape (batch_size, n_atoms, 3)
    # Roll the tensor to shift all elements to the right by one position along the atom axis
    if pos == "i":
        x_im1 = torch.roll(x, shifts=1, dims=1)
        x_im1[:, 0, :] = 0.0
        # Stack the original and shifted tensors along a new dimension
        xc = torch.stack([x_im1, x], dim=2)  # New shape will be (batch_size, n_atoms, 2, 3)
    elif pos == "j":
        x_ip1 = torch.roll(x, shifts=-1, dims=1)
        x_ip1[:, -1, :] = 0.0
        xc = torch.stack([x, x_ip1], dim=2)  # New shape will be (batch_size, n_atoms, 2, 3)
    else:
        raise ValueError(pos)
    return xc

def make_alpha_2d_flag(tensor):
    n_beads = tensor.shape[1]
    diag = torch.arange(0, n_beads)
    tensor[:,diag,diag] = 0
    tensor[:,diag[0:-1],diag[1:]] = 0
    tensor[:,diag[1:],diag[:-1]] = 0
    tensor[:,diag[0:-2],diag[2:]] = 0
    tensor[:,diag[2:],diag[:-2]] = 0
    tensor[:,0,:] = 0
    tensor[:,-1,:] = 0
    tensor[:,:,0] = 0
    tensor[:,:,-1] = 0
    return tensor

def get_alpha_2d(x):
    xc_i = duplicate(x, pos="i")
    xc_j = duplicate(x, pos="j")

    n_beads = x.shape[1]
    xc_i = xc_i.unsqueeze(2).repeat(1, 1, n_beads, 1, 1)
    xc_j = xc_j.unsqueeze(1).repeat(1, n_beads, 1, 1, 1)

    alpha_ij = calc_torsion(
        xc_i[:,:,:,0,:], xc_i[:,:,:,1,:], xc_j[:,:,:,0,:], xc_j[:,:,:,1,:],
        dim=-1
    )
    alpha_ij = make_alpha_2d_flag(alpha_ij)
    return alpha_ij

def get_alpha_2d_features(x):
    alpha_ij = get_alpha_2d(x)
    flag_ij = make_alpha_2d_flag(torch.ones_like(alpha_ij))
    feat_alpha_ij = torch.cat([
        torch.cos(alpha_ij).unsqueeze(-1),
        torch.sin(alpha_ij).unsqueeze(-1),
        flag_ij.unsqueeze(-1)
    ], dim=-1)
    return feat_alpha_ij


def get_coord_diff_2d(x):
    x_diff = x[:,:,None,:] - x[:,None,:,:]
    return x_diff


#------
# Beta.

def calc_angles_(A, B, C):
    # ix01 = angle_indices[:, [1, 0]]
    # ix21 = angle_indices[:, [1, 2]]

    u_prime = A - B  # xyz[:,ix01[:,1]]-xyz[:,ix01[:,0]]
    v_prime = C - B  # xyz[:,ix21[:,1]]-xyz[:,ix01[:,0]]
    u_norm = torch.sqrt((u_prime**2).sum(-1))
    v_norm = torch.sqrt((v_prime**2).sum(-1))

    # adding a new axis makes sure that broasting rules kick in on the third
    # dimension
    u = u_prime / (u_norm[..., None])
    v = v_prime / (v_norm[..., None])

    return torch.arccos((u * v).sum(-1))

def make_beta_2d_flag(tensor):
    n_beads = tensor.shape[1]
    diag = torch.arange(0, n_beads)
    tensor[:,diag,diag] = 0
    # tensor[:,diag[0:-1],diag[1:]] = 0
    tensor[:,diag[1:],diag[:-1]] = 0
    # tensor[:,diag[0:-2],diag[2:]] = 0
    # tensor[:,diag[2:],diag[:-2]] = 0
    tensor[:,0,:] = 0
    # tensor[:,-1,:] = 0
    # tensor[:,:,0] = 0
    # tensor[:,:,-1] = 0
    return tensor

def get_beta_2d(x):
    xc_i = duplicate(x, pos="i")
    xc_j = x.unsqueeze(2)
    n_beads = x.shape[1]
    xc_i = xc_i.unsqueeze(2).repeat(1, 1, n_beads, 1, 1)
    xc_j = xc_j.unsqueeze(1).repeat(1, n_beads, 1, 1, 1)

    beta_ij = calc_angles_(
        xc_i[:,:,:,0,:], xc_i[:,:,:,1,:], xc_j[:,:,:,0,:]
    )
    beta_ij = make_beta_2d_flag(beta_ij)
    return beta_ij

def get_beta_2d_features(x):
    beta_ij = get_beta_2d(x)
    flag_ij = make_beta_2d_flag(torch.ones_like(beta_ij))
    feat_beta_ij = torch.cat([
        torch.cos(beta_ij).unsqueeze(-1),
        torch.sin(beta_ij).unsqueeze(-1),
        flag_ij.unsqueeze(-1)
    ], dim=-1)
    return feat_beta_ij


#-----------
# Distances.

def bucketize(x, bins):
    """
    x: A scalar tensor (positive as per the use-case)
    bins: A tensor of bin edges
    
    Note: bins should be 1D and sorted in ascending order
    """
    # Since the bins are the edges, the output index will be where x should go between the edges
    index = torch.bucketize(x, bins, right=True)
    # Adjust index to be zero-based and to avoid out-of-bound index for values exactly at the upper edge
    index = index - 1
    # Number of bins is bins.size(0) - 1 since bins define the edges
    num_bins = bins.size(0)  # - 1
    # Create one-hot encoding
    one_hot = torch.nn.functional.one_hot(index, num_classes=num_bins)
    return one_hot

def calc_distogram_features(d, max_d, num_bins):
    bins = torch.linspace(0, max_d, num_bins).to(d.device)  # Create 12 bins
    # Testing with a specific scalar value
    b_ij = bucketize(d, bins)
    n_beads = d.shape[1]
    diag = torch.arange(0, n_beads)
    b_ij[:,diag,diag,:] = 0
    b_i = b_ij[:,diag[0:-1],diag[1:]].unsqueeze(2).repeat(1, 1, n_beads, 1)
    b_i = torch.nn.functional.pad(b_i, (0, 0, 0, 0, 1, 0))
    b_j = b_ij[:,diag[1:],diag[0:-1]].unsqueeze(1).repeat(1, n_beads, 1, 1)
    b_j = torch.nn.functional.pad(b_j, (0, 0, 1, 0, 0, 0))
    return torch.cat([b_ij, b_i, b_j], axis=-1)