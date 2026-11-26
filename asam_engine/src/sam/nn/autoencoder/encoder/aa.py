import math
from typing import Union, Callable

import numpy as np

import torch
import torch.nn as nn

from sam.coords import (
    calc_dmap, calc_torsion, calc_chain_bond_angles, torch_chain_dihedrals
)
from sam.nn.common import get_act_fn, AF2_PositionalEmbedding
from sam.nn.geometric import featurize_angle
from sam.nn.af3 import Transition
try:
    from sam.openfold.utils.rigid_utils import Rigid
except ImportError:
    Rigid = None


################################################################################
# Modules of the encoder network.                                              #
################################################################################

class SimpleAllAtomEmbedder(nn.Module):
    
    def __init__(self,
            mode: str,
            num_beads: int = 20,
            bead_embed_dim: int = 32,
            label_terminus: bool = False,
            local_pos: bool = False,
            local_pos_span: int = 2,
        ):
        super().__init__()

        self.mode = mode
        if mode == "cg":
            num_feats = 8
        elif mode == "aa":
            num_feats = 29
        else:
            raise KeyError(mode)
        if label_terminus:
            num_feats += 1
        self.label_terminus = label_terminus
        if local_pos:
            num_feats += (3*2+1)*(local_pos_span*2+1)
        self.local_pos = local_pos
        self.local_pos_span = local_pos_span

        self.bead_embedder = nn.Embedding(num_beads, bead_embed_dim)
        self.out_dim = num_feats + bead_embed_dim


    def forward(self, batch):

        node_f = []

        ## Embed amino acid types.
        bead_emb = self.bead_embedder(batch.a)

        ## Get Ca coordinates.
        if self.mode == "cg":
            # ca_xyz = batch.x
            raise NotImplementedError()
        elif self.mode == "aa":
            ca_xyz = batch.atom14_gt_positions[:,:,1,:]
        else:
            raise KeyError(self.mode)

        ## Alpha angles.
        alpha = torch_chain_dihedrals(ca_xyz)
        alpha_f = featurize_angle(alpha, (1, 2))
        node_f.append(alpha_f)

        ## Beta angles.
        beta = calc_chain_bond_angles(ca_xyz, backend="torch")
        beta_f = featurize_angle(beta, (0, 2))
        node_f.append(beta_f)

        ## Bond lengths.
        bond_length = torch.sqrt(
            torch.square(ca_xyz[:,:-1,:] - ca_xyz[:,1:,:]).sum(2)
        ) * 0.1
        bond_length = bond_length.unsqueeze(2)
        bond_length_f = torch.cat(
            [bond_length, torch.ones_like(bond_length)],
            dim=2
        )
        bond_length_f = nn.functional.pad(bond_length_f, (0, 0, 0, 1))
        node_f.append(bond_length_f)
        
        if self.mode == "aa":

            ## Chi angles.
            chi_f = torch.cat(
                [batch.chi_mask.unsqueeze(3), batch.chi_angles_sin_cos],
                dim=3
            )
            chi_f = chi_f.view(chi_f.shape[0], chi_f.shape[1], -1)
            node_f.append(chi_f)

            ## Phi/psi/omega angles.
            # PHI_ATOMS = ["-C", "N", "CA", "C"]
            phi = calc_torsion(batch.atom14_gt_positions[:,0:-1,2,:],
                               batch.atom14_gt_positions[:,1:,  0,:],
                               batch.atom14_gt_positions[:,1:,  1,:],
                               batch.atom14_gt_positions[:,1:,  2,:])
            phi_f = featurize_angle(phi, (1, 0))
            node_f.append(phi_f)

            # PSI_ATOMS = ["N", "CA", "C", "+N"]
            psi = calc_torsion(batch.atom14_gt_positions[:,0:-1,0,:],
                               batch.atom14_gt_positions[:,0:-1,1,:],
                               batch.atom14_gt_positions[:,0:-1,2,:],
                               batch.atom14_gt_positions[:,1:,  0,:])
            psi_f = featurize_angle(psi, (0, 1))
            node_f.append(psi_f)

            # OMEGA_ATOMS = ["CA", "C", "+N", "+CA"]
            omega = calc_torsion(batch.atom14_gt_positions[:,0:-1,1,:],
                                 batch.atom14_gt_positions[:,0:-1,2,:],
                                 batch.atom14_gt_positions[:,1:,  0,:],
                                 batch.atom14_gt_positions[:,1:,  1,:])
            omega_f = featurize_angle(omega, (0, 1))
            node_f.append(omega_f)

            if self.local_pos:
                if Rigid is None:
                    raise ImportError("Openfold is not installed")
                # Get the backbone frames.
                gt_aff = Rigid.from_tensor_4x4(batch.backbone_rigid_tensor)
                # Extract sliding windows for CA and COM coordinates.
                u_ca_xyz, u_mask = local_coords_unfold(
                    ca_xyz, S=self.local_pos_span, get_mask=True
                )
                com_xyz = calc_com_traj(batch, mult=1.0)
                u_com_xyz = local_coords_unfold(com_xyz, S=self.local_pos_span)
                # Transform the coordinates using the backbone frames. We have
                # L residues (and sliding windows) as well as L backbone frames.
                u_ca_xyz = gt_aff.invert()[...,None].apply(u_ca_xyz)
                u_ca_xyz = u_ca_xyz*u_mask
                u_com_xyz = gt_aff.invert()[...,None].apply(u_com_xyz)
                u_com_xyz = u_com_xyz*u_mask
                # Reshape.
                u_coords = torch.cat([u_ca_xyz, u_com_xyz, u_mask[...,:1]], dim=-1)
                u_coords = u_coords.view(*u_coords.shape[:-2], -1)
                node_f.append(u_coords)
        
        ## Label N- and C-term.
        if self.label_terminus:
            term_label = torch.zeros(
                batch.x.shape[0],
                batch.x.shape[1],
                device=batch.x.device
            )
            term_label[:,0] = 1
            term_label[:,-1] = 1
            node_f.append(term_label.unsqueeze(-1))
            
        ## Concatenate the features.
        node_f.append(bead_emb)
        
        node_f = torch.cat(node_f, dim=2)

        return node_f


def calc_no_dmap(batch, epsilon=1e-12):
    n = batch.atom14_gt_positions[:,:,0,:]*0.1
    o = batch.atom14_gt_positions[:,:,3,:]*0.1
    dmap_no = torch.cdist(n, o, p=2.0)
    dmap_no = dmap_no.unsqueeze(1)
    return dmap_no

def calc_com_traj(batch, mult=0.1):
    """
    TODO: remove, redundant.
    """
    # Get positions.
    pos = batch.atom14_gt_positions[:,:,4:,:]*mult  # Side-chain atoms positions.
    ca_pos = batch.atom14_gt_positions[:,:,1,:]*mult  # CA positions.
    mask = batch.atom14_gt_exists[:,:,4:].unsqueeze(-1)  # Check if side-chain atoms exists.
    gly_mask = 1-batch.atom14_gt_exists[:,:,4].unsqueeze(-1)  # Check if CB atoms exist.
    mask = mask.to(dtype=pos.dtype)
    gly_mask = gly_mask.to(dtype=pos.dtype)
    # Compute centroid.
    com_pos = (pos.sum(dim=2) + ca_pos*gly_mask)/(mask.sum(dim=2) + gly_mask)
    return com_pos


def local_coords_unfold(x, S, get_mask=False):
    """
    x: coordinates
    S: sliding window size
    """
    B, L, C = x.shape
    if get_mask:
        mask = torch.ones_like(x)

    # Pad the tensor along the length dimension
    padded_x = torch.nn.functional.pad(x, (0, 0, S, S), "constant", 0)
    if get_mask:
        padded_mask = torch.nn.functional.pad(mask, (0, 0, S, S), "constant", 0)
        
    # Use unfold to create sliding windows
    unfolded = padded_x.unfold(1, S * 2 + 1, 1)
    if get_mask:
        unfolded_mask = padded_mask.unfold(1, S * 2 + 1, 1)

    # Reshape and permute to get the desired output shape (B, L, S*2+1, 3)
    output = unfolded.permute(0, 1, 3, 2).contiguous()
    if get_mask:
        output_mask = unfolded_mask.permute(0, 1, 3, 2).contiguous()
        return output, output_mask
    else:
        return output

def slice_unfolded(x):
    # Determine the middle index along the dimension W
    middle_index = x.shape[2] // 2  # Assuming W is odd, e.g., W = 5, middle_index = 2
    # Use tensor slicing to exclude the middle element
    # Concatenate the slices before and after the middle index
    output = torch.cat((x[:, :, :middle_index, :], x[:, :, middle_index+1:, :]), dim=2)
    return output


class DmapProjectionModule(nn.Module):

    def __init__(self, edge_dim, dmap_dim, linear_bias: bool = True):
        super().__init__()
        self.edge_embedder = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, edge_dim, bias=linear_bias),
        )
        self.dmap_embedder = nn.Sequential(
            nn.Linear(dmap_dim, edge_dim, bias=linear_bias),
        )
        self.out_module = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim, bias=linear_bias),
        )
    
    def forward(self, dmap_emb, z):
        v = self.edge_embedder(z) + self.dmap_embedder(dmap_emb)
        return self.out_module(v)


# class AllAtomEncoderBlock(nn.Module):

#     def __init__(
#             self,
#             node_dim,
#             edge_dim,
#             dmap_dim,
#             num_heads: int,
#             linear_bias: bool = True,
#             update_edges: bool = False,
#             node_update_params: dict = {}
#         ):

#         super().__init__()
#         ### Input.
#         self.node_residual_module = nn.Sequential(
#             nn.LayerNorm(node_dim),
#             nn.Linear(node_dim, node_dim, bias=linear_bias)
#         )
#         self.edge_residual_module = nn.Sequential(
#             nn.LayerNorm(edge_dim),
#             nn.Linear(edge_dim, edge_dim, bias=linear_bias)
#         )
#         self.dmap_projection_module = DmapProjectionModule(
#             edge_dim=edge_dim,
#             dmap_dim=dmap_dim,
#             linear_bias=linear_bias
#         )

#         ### Edge update.
#         self.update_edges = update_edges
#         if update_edges:
#             raise NotImplementedError()
        
#         ### Node update.
#         self.node_update_module = AllAtomEncoderNodeUpdater(
#             mode="transformer",
#             node_dim=node_dim,
#             edge_dim=edge_dim,
#             num_heads=num_heads,
#             params=node_update_params
#         )

#         self.node_transition_module = Transition(
#             in_dim=node_dim,
#             n=2,
#             linear_bias=linear_bias
#         )

    
#     def forward(self, h_hat, z_hat, h_init, z_init, dmap_exp):
#         z = z_init + self.edge_residual_module(z_hat)
#         z = z + self.dmap_projection_module(dmap_exp, z)
#         h = h_init + self.node_residual_module(h_hat)

#         '''
#         if self.update_edges:
#             z += self.edge_update_module(z)
#             z += self.edge_transition_module(z)
#         '''

#         h = h + self.node_update_module(h, z)
#         h = h + self.node_transition_module(h)

#         return h, z


# class AllAtomEncoderNodeUpdater(nn.Module):
#     """New module documentation: TODO."""

#     def __init__(self,
#             mode: str, node_dim: int, edge_dim: int, num_heads: int,
#             params: dict
#         ):
#         """Arguments: TODO."""
#         super().__init__()
#         self.mode = mode
#         if self.mode == "transformer":
#             self.layer = DecoderTransformerBlock(
#                 d_model=node_dim,
#                 nhead=num_heads,
#                 dim_feedforward=params.get("mlp_dim", node_dim*4),
#                 dropout=0.0,
#                 activation=params.get("activation", nn.ReLU()),
#                 batch_first=True,
#                 norm_first=True,
#                 edge_dim=edge_dim,
#                 bias_2d=params.get("bias_2d", True),
#             )
#         else:
#             raise NotImplementedError()

#     def forward(self, h, z):
#         return self.layer(h, z)

  
################################################################################
# Encoder networks.                                                            #
################################################################################

# class AllAtomEncoder_0(nn.Module):

#     def __init__(
#             self,
#             encoding_dim: int = 16,
#             num_blocks: int = 8,
#             # attention_type: str = "transformer",
#             node_dim: int = 256,
#             edge_dim: int = 128,
#             # d_model: int = None,
#             num_heads: int = 16,
#             mlp_dim: int = None,
#             # dropout: int = None,
#             # norm_eps: float = 1e-5,
#             # norm_pos: str = "pre",
#             activation: Union[str, Callable] = "relu",
#             out_mode: str = "idpgan",
#             bead_embed_dim: int = 32,
#             num_beads: int = 20,
#             # pos_embed_dim: int = 64,
#             # use_bias_2d: bool = True,
#             pos_embed_r: int = 32,
#             use_res_ids: bool = False,
#             input_embed_mode: str = "fc",
#             input_embed_params: dict = {"mode": "cg"},
#             # embed_2d_inject_mode: str = "concat",
#             dmap_ca_min: float = 0.0,
#             dmap_ca_cutoff: float = 10.0,
#             dmap_ca_num_gaussians: int = 128,
#             dmap_embed_type: str = "rbf",
#             dmap_embed_trainable: bool = False,
#             use_dmap_mlp: bool = False,  # legacy

#             dmap_sin_dim: int = 256,
#             dmap_sin_freq: int = 256,
#             dmap_sin_mult: float = 50.0,

#             use_2d_angular: bool = False,
#             # max_2d_dist_bin: int = 1.4,
#             # num_2d_dist_bins: int = 11,

#             linear_bias: bool = True,

#             node_update_params: dict = {}
#         ):
#         """
#         `encoding_dim`: dimension of the structural encoding vectors.
#         `dmap_ca_min`: min distance in the radial basis function (RBF) embedding
#             for Ca-Ca distances. Values are in Amstrong.
#         `dmap_ca_cutoff`: maximum distance in the RBF ebbedding for Ca-Ca
#             distances.
#         """

#         super().__init__()

#         ### Check and store the attributes.
#         self.node_dim = node_dim
#         self.use_res_ids = use_res_ids
#         if node_update_params is None:
#             node_update_params = {}

#         ### Shared functions.
#         if isinstance(activation, str):
#             act_cls = get_act_fn(activation_name=activation)
#         else:
#             raise TypeError(activation.__class__)
#         self.act_cls = act_cls

#         ### Process input.
#         # Embed node features.
#         if input_embed_mode == "fc":
#             self.node_input_embedder = SimpleAllAtomEmbedder(
#                 **input_embed_params
#             )
#         elif input_embed_mode == "atom_embedder":
#             raise NotImplementedError()
#         else:
#             raise KeyError(input_embed_mode)

#         self.node_init_linear = nn.Linear(
#             self.node_input_embedder.out_dim, node_dim, bias=linear_bias
#         )

#         # Embed edge features.
#         self.edge_init_linear_i = nn.Linear(
#             self.node_input_embedder.out_dim, edge_dim, bias=linear_bias
#         )
#         self.edge_init_linear_j = nn.Linear(
#             self.node_input_embedder.out_dim, edge_dim, bias=linear_bias
#         )

#         # Relative positional encodings.
#         self.position_embedder = AF2_PositionalEmbedding(
#             pos_embed_dim=edge_dim,
#             pos_embed_r=pos_embed_r,
#             dim_order="trajectory"
#         )

#         # Embed Ca-Ca distances.
#         if dmap_embed_type == "rbf":
#             self.dmap_ca_expansion = GaussianSmearing(
#                 start=dmap_ca_min,
#                 stop=dmap_ca_cutoff,
#                 num_gaussians=dmap_ca_num_gaussians)
#             dmap_dim = dmap_ca_num_gaussians
#         elif dmap_embed_type == "expnorm":
#             self.dmap_ca_expansion = ExpNormalSmearing(
#                 cutoff_lower=dmap_ca_min,
#                 cutoff_upper=dmap_ca_cutoff,
#                 num_rbf=dmap_ca_num_gaussians,
#                 trainable=dmap_embed_trainable)
#             dmap_dim = dmap_ca_num_gaussians
#         elif dmap_embed_type == "sinusoidal":
#             self.dmap_ca_expansion = SinusoidalDistanceEmbedder(
#                 hidden_size=dmap_sin_dim,
#                 frequency_embedding_size=dmap_sin_freq,
#                 activation=act_cls,
#                 max_period=10000,
#                 mult=dmap_sin_mult
#             )
#             dmap_dim = dmap_sin_dim
#         else:
#             raise KeyError(dmap_embed_type)
#         self.dmap_embed_type = dmap_embed_type
        
#         # if not use_dmap_mlp:
#         #     self.dmap_ca_embedder = nn.Linear(dmap_ca_num_gaussians, edge_dim)
#         # else:
#         #     self.dmap_ca_embedder = nn.Sequential(
#         #         nn.Linear(dmap_ca_num_gaussians,edge_dim),
#         #         act_cls(),
#         #         nn.Linear(edge_dim, edge_dim)
#         #     )

#         if use_2d_angular:
#             self.angular_2d_embedder = nn.Sequential(
#                 nn.Linear(3+3, edge_dim),
#                 act_cls(),
#                 nn.Linear(edge_dim, edge_dim),
#             )
#             # self.max_2d_dist_bin = max_2d_dist_bin
#             # self.num_2d_dist_bins = num_2d_dist_bins
#         self.use_2d_angular = use_2d_angular

        
#         ### Transformer layers.
#         self.num_blocks = num_blocks
#         self.blocks = []

#         for l in range(self.num_blocks):
#             block_l = AllAtomEncoderBlock(
#                 node_dim=node_dim,
#                 edge_dim=edge_dim,
#                 dmap_dim=dmap_dim,
#                 num_heads=num_heads,
#                 linear_bias=linear_bias,
#                 node_update_params=node_update_params
#             )
#             self.blocks.append(block_l)
            
#         self.blocks = nn.ModuleList(self.blocks)

#         ### Output module.
#         self.out_module = nn.Sequential(
#             nn.LayerNorm(node_dim),
#             nn.Linear(node_dim, encoding_dim)
#         )

#     # def get_embed_dim(self):
#     #     return self.embed_dim

#     def forward(self, batch):
#         """
#         TODO.
#         """

#         ### Embed input and initialize.

#         # Node embeddings.
#         h_input = self.node_input_embedder(batch)

#         h_init = self.node_init_linear(h_input)

#         # Edge embeddings. Init with outer sum from node input.
#         z_init = self.edge_init_linear_i(h_input)[:,None,:,:] + \
#                  self.edge_init_linear_j(h_input)[:,:,None,:]
        
#         z_init += self.position_embedder(batch.x, r=batch.r)

#         # Get the Ca-Ca distance matrix (2d features).
#         dmap_ca = calc_dmap(batch.x)
#         dmap_ca_exp = self.dmap_ca_expansion(dmap_ca)
#         if self.dmap_embed_type in ("rbf", "expnorm"):
#             dmap_ca_exp = dmap_ca_exp.transpose(1, 3)
#         # dmap_ca_emb = self.dmap_ca_embedder(dmap_ca_exp)
#         if self.use_2d_angular:
#             _af = get_alpha_2d_features(batch.x)
#             _bf = get_beta_2d_features(batch.x)
#             # _df = calc_distogram_features(
#             #     dmap_ca[:,0,...],
#             #     max_d=self.max_2d_dist_bin,
#             #     num_bins=self.num_2d_dist_bins
#             # )
#             _az = self.angular_2d_embedder(torch.cat([_af, _bf], axis=3))
#             z_init = z_init + _az

#         ### Go through all the transformer blocks.
#         h_hat = torch.zeros_like(h_init)
#         z_hat = torch.zeros_like(z_init)
#         for block_idx, block_l in enumerate(self.blocks):
            
#             h, z = block_l(
#                 h_hat=h_hat,
#                 z_hat=z_hat,
#                 h_init=h_init,
#                 z_init=z_init,
#                 dmap_exp=dmap_ca_exp
#             )

#             h_hat = h
#             z_hat = z

#         ### Output module.
#         enc = self.out_module(h)

#         return enc

    
#     def nn_forward(self, batch, x=None, num_nodes=None):
#         if num_nodes is not None:
#             raise NotImplementedError()
#         enc = self.forward(batch=batch)
#         return enc


# class AllAtomEncoder_1(CA_TransformerEncoder):

#     def _new_layers(self):
#         self.project_input_0 = SimpleAllAtomEmbedder(
#             mode="cg",
#             num_beads=20,
#             bead_embed_dim=self.beads_embed.embedding_dim
#         )
#         self.project_input_1 = nn.Sequential(
#             nn.Linear(self.project_input_0.out_dim, self.embed_dim),
#             self.act_cls(),
#             nn.Linear(self.embed_dim, self.embed_dim),
#         )


#     def forward(self, batch):
#         """
#         x: input tensor with shape (B, L, 3).
#         a: amino acid tensor with shape (B, L).
#         """
#         ### Input.
#         # Get the Ca-Ca distance matrix (2d features).
#         dmap_ca = calc_dmap(batch.x)
#         rbf_ca = self.dmap_ca_expansion(dmap_ca).transpose(1, 3)
#         z = self.project_dmap(rbf_ca)

#         # Amino acid encodings.
#         if self.embed_inject_mode is not None:
#             a_e = self.beads_embed(batch.a).transpose(0, 1)
#         else:
#             a_e = None
        
#         # Torsion angles as 1d input embeddings.
#         s = self.project_input_0(batch)
#         s = self.project_input_1(s)
#         s = s.transpose(0, 1)  # (B, L, E)

#         ### Positional embeddings.
#         p = self.embed_pos(batch.x, r=batch.r)

#         ### Go through all the transformer blocks.
#         s_0 = None
#         for layer_idx, layer in enumerate(self.layers):
#             s, attn = layer(x=s, a=a_e, p=p, z=z, x_0=s_0)

#         ### Output module.
#         s = self.out_module(s)
#         enc = s.transpose(0, 1)  # (L, B, E) => (B, L, E)
#         # if self.out_conv_layer is not None:
#         #     enc = self.out_conv_layer(enc.transpose(1, 2)).transpose(2, 1)

#         return enc

    
#     def nn_forward(self, batch, x=None, num_nodes=None):
#         if num_nodes is not None:
#             raise NotImplementedError()
#         enc = self.forward(batch)
#         return enc


################################################################################
# New encoder network v0.1.                                                    #
################################################################################

from sam.nn.geometric import BeadDistanceEmbedding, TrAnglesEmbedding
from sam.nn.common import FeedForward, SwiGLU, TransformerBlock_v01


class DmapProjectionModule_v01(nn.Module):

    def __init__(self,
            edge_dim: int,
            dmap_dim: int,
            linear_bias: bool = True,
            out_mode: str = "shallow",
            activation: str = "silu"
        ):
        super().__init__()
        self.edge_embedder = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, edge_dim, bias=linear_bias),
        )
        self.dmap_embedder = nn.Sequential(
            nn.Linear(dmap_dim, edge_dim, bias=linear_bias),
        )
        if out_mode == "shallow":
            self.out_module = nn.Sequential(
                nn.LayerNorm(edge_dim),
                get_act_fn(activation)(),
                nn.Linear(edge_dim, edge_dim, bias=linear_bias),
            )
        elif out_mode == "transition":
            # self.out_module = nn.Sequential(
            #     nn.LayerNorm(edge_dim),
            #     SwiGLU(
            #         in_dim=edge_dim,
            #         hidden_dim=edge_dim,
            #         out_dim=edge_dim,
            #         linear_bias=linear_bias
            #     )
            # )
            raise NotImplementedError()
        else:
            raise KeyError(out_mode)
    
    def forward(self, dmap_emb, z):
        v = self.edge_embedder(z) + self.dmap_embedder(dmap_emb)
        return self.out_module(v)


class AllAtomEncoderBlock_v01(nn.Module):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            dmap_dim: int,
            num_heads: int,
            mlp_dim: int = None,
            activation: Union[Callable, str] = "silu",
            linear_bias: bool = True,
            add_bias_2d: bool = True,
            edge_residual: bool = True,
            node_residual: bool = True,
            update_edges: bool = False,
            node_update_addition: bool = True,
            edge_transition: bool = False,
            node_transition: bool = True,

            # Development.
            sum_z_to_dmap: bool = False,
            normalize_residual: bool = False,
            dmap_out_mode: str = "shallow",
            accessory_activation: Union[Callable, str] = "silu",
        ):

        super().__init__()

        ### Input.
        self.node_residual = node_residual
        if self.node_residual:
            self.node_residual_module = nn.Sequential(
                nn.LayerNorm(node_dim),
                nn.Linear(node_dim, node_dim, bias=linear_bias)
            )
            if normalize_residual:
                # self.node_residual_normalize = nn.LayerNorm(node_dim)
                raise NotImplementedError()
            else:
                self.node_residual_normalize = nn.Identity()

        self.edge_residual = edge_residual
        if self.edge_residual:
            # self.edge_residual_module = nn.Sequential(
            #     nn.LayerNorm(edge_dim),
            #     nn.Linear(edge_dim, edge_dim, bias=linear_bias)
            # )
            raise NotImplementedError()

        self.dmap_projection_module = DmapProjectionModule_v01(
            edge_dim=edge_dim,
            dmap_dim=dmap_dim,
            linear_bias=linear_bias,
            out_mode=dmap_out_mode,
            activation=accessory_activation,
        )
        self.sum_z_to_dmap = sum_z_to_dmap

        ### Edge update.
        self.update_edges = update_edges
        if update_edges:
            raise NotImplementedError()
        
        ### Node update.
        self.node_update_module = TransformerBlock_v01(
            embed_dim=node_dim,
            edge_dim=edge_dim,
            mlp_dim=mlp_dim if mlp_dim is not None else node_dim*2,  # node_dim*4
            num_heads=num_heads,
            activation=activation,
            linear_bias=linear_bias,
            add_bias_2d=add_bias_2d,
        )
        self.node_update_addition = node_update_addition

        if node_transition:
            self.node_transition_module = Transition(
                in_dim=node_dim,
                n=2,
                linear_bias=linear_bias
            )
        self.node_transition = node_transition

    
    def forward(self, h_hat, z_hat, h_init, z_init, dmap_exp):
        # Edge residual.
        if self.edge_residual:  # TODO: it should be used only when updating.
            z = z_init + self.edge_residual_module(z_hat)
        else:
            if self.update_edges:
                # z = z_hat
                raise NotImplementedError()
            else:
                z = z_init
        # Project dmap.
        if self.sum_z_to_dmap:
            # z = z + self.dmap_projection_module(dmap_exp, z)
            raise NotImplementedError()
        else:
            z = self.dmap_projection_module(dmap_exp, z)
        # Node residual.
        if self.node_residual:
            h = h_init + self.node_residual_module(h_hat)
            h = self.node_residual_normalize(h)
        else:
            # h = h_hat
            raise NotImplementedError()

        # Update edges.
        if self.update_edges:
            # z += self.edge_update_module(z)
            # z += self.edge_transition_module(z)
            raise NotImplementedError()

        # Update nodes.
        h = self._node_update_addition(h, self.node_update_module(h, z))
        if self.node_transition:
            h = h + self.node_transition_module(h)

        return h, z
    
    def _node_update_addition(self, h, h_upt):
        if self.node_update_addition:
            # return h + h_upt
            raise NotImplementedError()
        else:
            return h_upt


class AllAtomEncoder_v01(nn.Module):

    def __init__(
            self,
            encoding_dim: int = 16,
            num_blocks: int = 8,
            # attention_type: str = "transformer",
            node_dim: int = 256,
            edge_dim: int = 128,
            num_heads: int = 16,
            mlp_dim: int = None,
            activation: Union[str, Callable] = "silu",
            out_mode: str = "simple",
            bead_embed_dim: int = 32,
            num_beads: int = 20,
            pos_embed_r: int = 32,
            use_res_ids: bool = False,
            input_embed_params: dict = {},
            node_init_mode: str = "linear",
            edge_init_mode: str = "linear",

            dmap_embed_params: dict = {
                "type": "rbf",
                "start": 0.0, "stop": 10.0, "num_gaussians": 128
            },
            dmap_inject_mode: str = "shallow",
            # embed_2d_inject_mode: str = "concat",
            no_dmap_embed_params: dict = {},
            com_dmap_embed_params: dict = {},
            amap_embed_params: dict = {},
            dmap_merge_mode: str = "cat_shallow",
            dmap_merge_dim: int = 128,

            linear_bias: bool = True,
            add_bias_2d: bool = True,

            edge_residual: bool = True,
            node_residual: bool = True,
            node_update_addition: bool = True,
            recycle_init: str = "zeros",
            accessory_activation: Union[str, Callable] = "silu",
            node_transition: bool = True,
            out_ln: bool = False,
            encode_bl: bool = False  # Legacy.
        ):
        """
        `encoding_dim`: dimension of the structural encoding vectors.
        """

        super().__init__()

        ### Check and store the attributes.
        self.node_dim = node_dim
        self.use_res_ids = use_res_ids

        ### Process input.
        # Embed node features.
        self.node_input_embedder = SimpleAllAtomEmbedder(
            mode="aa",
            bead_embed_dim=bead_embed_dim,
            **input_embed_params
        )

        if node_init_mode == "linear":
            self.node_init_linear = nn.Linear(
                self.node_input_embedder.out_dim, node_dim, bias=linear_bias
            )
        elif node_init_mode == "mlp":
            self.node_init_linear = FeedForward(
                in_dim=self.node_input_embedder.out_dim,
                hidden_dim=node_dim,  # node_dim*2
                out_dim=node_dim,
                activation=accessory_activation,  # activation
                linear_bias=linear_bias
            )
        else:
            raise KeyError(node_init_mode)

        # Embed edge features.
        if edge_init_mode == "linear":
            self.edge_init_linear_i = nn.Linear(
                self.node_input_embedder.out_dim, edge_dim, bias=linear_bias
            )
            self.edge_init_linear_j = nn.Linear(
                self.node_input_embedder.out_dim, edge_dim, bias=linear_bias
            )
        elif edge_init_mode == "mlp":
            self.edge_init_linear_i = FeedForward(
                in_dim=self.node_input_embedder.out_dim,
                hidden_dim=edge_dim,
                out_dim=edge_dim,
                activation=accessory_activation,
                linear_bias=linear_bias
            )
            self.edge_init_linear_j = FeedForward(
                in_dim=self.node_input_embedder.out_dim,
                hidden_dim=edge_dim,
                out_dim=edge_dim,
                activation=accessory_activation,
                linear_bias=linear_bias
            )
        elif edge_init_mode == "nl_from_node":
            self.edge_init_linear_i = nn.Sequential(
                get_act_fn(accessory_activation)(),
                nn.Linear(node_dim, edge_dim, bias=linear_bias)
            )
            self.edge_init_linear_j = nn.Sequential(
                get_act_fn(accessory_activation)(),
                nn.Linear(node_dim, edge_dim, bias=linear_bias)
            )
        else:
            raise KeyError(edge_init_mode)
        self.edge_init_mode = edge_init_mode


        # Relative positional encodings.
        self.position_embedder = AF2_PositionalEmbedding(
            pos_embed_dim=edge_dim,
            pos_embed_r=pos_embed_r,
            dim_order="trajectory"
        )

        # Embed Ca-Ca distances.
        self.dmap_embedder = BeadDistanceEmbedding(
            edge_dim=edge_dim,
            embed_type=dmap_embed_params["type"],
            activation=activation,  # accessory_activation
            params=dmap_embed_params,
            use_project=False,
            use_mlp=False,
            input_type="xyz"
        )

        # Embed N-O distances.
        if no_dmap_embed_params:
            self.no_dmap_embedder = BeadDistanceEmbedding(
                edge_dim=edge_dim,
                embed_type=no_dmap_embed_params["type"],
                activation=activation,  # accessory_activation
                params=no_dmap_embed_params,
                use_project=False,
                use_mlp=False,
                input_type="dmap"
            )
        else:
            self.no_dmap_embedder = None
        
        # Embed COM-COM distances.
        if com_dmap_embed_params:
            self.com_dmap_embedder = BeadDistanceEmbedding(
                edge_dim=edge_dim,
                embed_type=com_dmap_embed_params["type"],
                activation=activation,  # accessory_activation
                params=com_dmap_embed_params,
                use_project=False,
                use_mlp=False,
                input_type="xyz"
            )
        else:
            self.com_dmap_embedder = None
        
        # trRosetta angles 2d map.
        if amap_embed_params:
            self.amap_embedder = TrAnglesEmbedding(
                edge_dim=edge_dim,
                activation=accessory_activation,
                use_theta=amap_embed_params["theta"],
                use_phi=amap_embed_params["phi"],
                use_omega=amap_embed_params["omega"],
                use_project=False,
                use_mlp=False
            )
        else:
            self.amap_embedder = None
        
        # Module to merge the distance maps of various beads.
        dmap_modules = (
            self.no_dmap_embedder, self.com_dmap_embedder, self.amap_embedder
        )
        # if self.no_dmap_embedder is not None or self.com_dmap_embedder is not None:
        if any(mod is not None for mod in dmap_modules):
            if dmap_merge_mode == "cat_shallow":
                # Concatenate.
                dmap_merge_in = self.dmap_embedder.dmap_dim
                if self.no_dmap_embedder is not None:
                    dmap_merge_in += self.no_dmap_embedder.dmap_dim
                if self.com_dmap_embedder is not None:
                    dmap_merge_in += self.com_dmap_embedder.dmap_dim
                if self.amap_embedder is not None:
                    dmap_merge_in += self.amap_embedder.amap_dim
                self.dmap_merger = nn.Sequential(
                    nn.Linear(dmap_merge_in, dmap_merge_dim, bias=linear_bias),
                    get_act_fn(accessory_activation)()
                )
            else:
                raise KeyError(dmap_merge_mode)
            dmap_exp_dim = dmap_merge_dim
        else:
            self.dmap_merger = None
            dmap_exp_dim = self.dmap_embedder.dmap_dim
        self.dmap_merge_mode = dmap_merge_mode
        
        ### Transformer layers.
        self.num_blocks = num_blocks
        self.blocks = []

        for l in range(self.num_blocks):
            block_l = AllAtomEncoderBlock_v01(
                node_dim=node_dim,
                edge_dim=edge_dim,
                dmap_dim=dmap_exp_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                node_transition=node_transition,
                linear_bias=linear_bias,
                add_bias_2d=add_bias_2d,
                edge_residual=edge_residual,
                node_residual=node_residual,
                node_update_addition=node_update_addition,
                activation=activation,
                dmap_out_mode=dmap_inject_mode,
                accessory_activation=accessory_activation
            )
            self.blocks.append(block_l)
        self.recycle_init = recycle_init
        self.node_residual = node_residual
        self.edge_residual = edge_residual
            
        self.blocks = nn.ModuleList(self.blocks)

        ### Output module.
        if out_mode == "simple":
            self.out_module = nn.Sequential(
                nn.LayerNorm(node_dim),
                nn.Linear(
                    node_dim, encoding_dim,
                    # linear_bias=linear_bias
                )
            )
        elif out_mode == "mlp":
            self.out_module = nn.Sequential(
                nn.LayerNorm(node_dim),
                FeedForward(
                    in_dim=node_dim,
                    hidden_dim=node_dim,
                    out_dim=encoding_dim,
                    activation=accessory_activation,
                    linear_bias=linear_bias
                )
            )
        elif out_mode == "transition":
            self.out_module = nn.Sequential(
                nn.LayerNorm(node_dim),
                SwiGLU(
                    in_dim=node_dim,
                    hidden_dim=node_dim*2,
                    out_dim=encoding_dim,
                    linear_bias=linear_bias
                )
            )
        else:
            raise KeyError(out_mode)
        
        if out_ln:
            self.out_module = nn.Sequential(
                self.out_module,
                nn.LayerNorm(encoding_dim, elementwise_affine=False)
            )


    def forward(self, batch):
        """
        TODO.
        """

        ### Embed input and initialize.

        # Node embeddings.
        h_input = self.node_input_embedder(batch)

        h_init = self.node_init_linear(h_input)

        # Edge embeddings. Init with outer sum from node features.
        if self.edge_init_mode == "nl_from_node":
            # Input are embedded node features.
            # edge_input = h_init
            raise NotImplementedError()
        else:
            # Input are raw node features.
            edge_input = h_input
        z_init = self.edge_init_linear_i(edge_input)[:,None,:,:] + \
                 self.edge_init_linear_j(edge_input)[:,:,None,:]
        
        z_init += self.position_embedder(batch.x, r=batch.r)

        # Get an initial embedding (or expansion) of distance matrices.
        dmap_exp = self._embed_dmap_forward(batch)
        
        ### Go through all the transformer blocks.
        if self.recycle_init == "init":
            h_hat = h_init
            z_hat = z_init
        elif self.recycle_init == "zeros":
            h_hat = torch.zeros_like(h_init) if self.node_residual else h_init
            z_hat = torch.zeros_like(z_init) if self.edge_residual else z_init
        else:
            raise KeyError(self.recycle_init)

        for block_idx, block_l in enumerate(self.blocks):
            
            h, z = block_l(
                h_hat=h_hat,
                z_hat=z_hat,
                h_init=h_init,
                z_init=z_init,
                dmap_exp=dmap_exp
            )

            h_hat = h
            z_hat = z

        ### Output module.
        enc = self.out_module(h)

        return enc


    def _embed_dmap_forward(self, batch):
        # Get the Ca-Ca distance matrix (2d features).
        dmap_ca_exp = self.dmap_embedder(batch.x)
        # Get other distance matrices.
        if self.dmap_merger is not None:
            # N-O distance matrix.
            if self.no_dmap_embedder is not None:
                dmap_no = calc_no_dmap(batch)
                dmap_no_exp = self.no_dmap_embedder(dmap_no)
            # COM-COM distance matrix.
            if self.com_dmap_embedder is not None:
                com = calc_com_traj(batch)
                dmap_com_exp = self.com_dmap_embedder(com)
            # trRosetta angles matrix.
            if self.amap_embedder is not None:
                amap_exp = self.amap_embedder(batch)
            # Embed via concatenation.
            if self.dmap_merge_mode == "cat_shallow":
                dmap_merger_in = [dmap_ca_exp]
                if self.no_dmap_embedder is not None:
                    dmap_merger_in.append(dmap_no_exp)
                if self.com_dmap_embedder is not None:
                    dmap_merger_in.append(dmap_com_exp)
                if self.amap_embedder is not None:
                    dmap_merger_in.append(amap_exp)
                dmap_exp = self.dmap_merger(
                    torch.cat(dmap_merger_in, dim=-1)
                )
            else:
                raise KeyError(self.dmap_merge_mode)
        else:
            dmap_exp = dmap_ca_exp
        return dmap_exp

    
    def nn_forward(self, batch, num_nodes=None):
        if num_nodes is not None:
            raise NotImplementedError()
        enc = self.forward(batch=batch)
        return enc


if __name__ == "__main__":
    
    from sam.data.aa_protein import AllAtomProteinDataset

    torch.manual_seed(2024)
    np.random.seed(2024)

    device = "cpu"
    use_dec = True

    name = "6c62_C"  # "6c62_C", "5ij2_B"
    dataset = AllAtomProteinDataset(
        input=[
            {"name": name,
             "topology": f"/feig/s2/giacomo/sam/datasets/new/atlas/ha/{name}/top.pdb",
             "trajectories": [
                    f"/feig/s2/giacomo/sam/datasets/new/atlas/ha/{name}/traj_0.xtc",
                    f"/feig/s2/giacomo/sam/datasets/new/atlas/ha/{name}/traj_1.xtc"
                    f"/feig/s2/giacomo/sam/datasets/new/atlas/ha/{name}/traj_2.xtc"
                ]
            }
        ],
        n_trajs=1,
        n_frames=1000,
        frames_mode="ensemble",
        proteins=None,
        per_protein_frames=None,
        re_filter=None,
        res_ids_mode=None,
        bead_type="ca",
        alphabet="standard",
        xyz_sigma=None,
        xyz_perturb=None,
        verbose=True,
        random_seed=None
    )

    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=dataset, batch_size=6, shuffle=False
    )

    net = AllAtomEncoder_v01(
        encoding_dim=32,
        accessory_activation="silu",
        activation="silu",
        add_bias_2d=True,
        bead_embed_dim=32,
        com_dmap_embed_params={
            "cutoff_lower": 0.0,
            "cutoff_upper": 7.0,
            "num_rbf": 64,
            "trainable": True,
            "type": "expnorm"
        },
        dmap_embed_params={
            "cutoff_lower": 0.0,
            "cutoff_upper": 10.0,
            "num_rbf": 128,
            "trainable": True,
            "type": "expnorm",
        },
        dmap_inject_mode="shallow",
        dmap_merge_dim=192,
        dmap_merge_mode="cat_shallow",
        edge_dim=128,
        edge_residual=False,
        input_embed_params={
            "label_terminus": True,
            "local_pos": True,
            "local_pos_span": 3,
        },
        linear_bias=True,
        mlp_dim=512,
        no_dmap_embed_params={
            "cutoff_lower": 0.0,
            "cutoff_upper": 3.0,
            "num_rbf": 32,
            "trainable": True,
            "type": "expnorm",
        },
        node_dim=256,
        node_init_mode="mlp",
        node_residual=True,
        node_update_addition=False,
        num_blocks=4,
        num_heads=16,
        out_ln=False,
        out_mode="simple",
        pos_embed_r=32,
    )

    net = net.to(device)


    #
    #
    #
    if use_dec:
        
        from sam.training.autoencoder.aa_losses import (
            compute_bl_loss, compute_e_vdw, compute_e_bond, compute_nbsr_loss
        )
        from sam.nn.autoencoder.decoder.aa import AllAtomDecoder_v01

        dec = AllAtomDecoder_v01(
            encoding_dim=32,

            num_blocks=1,
            node_dim=256,
            edge_dim=128,
            num_heads=16,
            mlp_dim=None,
            activation="swiglu",
            out_mode="simple",
            embed_inject_mode="outer_sum",
            bead_embed_dim=32,
            num_beads=20,
            pos_embed_r=32,
            use_res_ids=False,
            node_init_mode="linear",
            linear_bias=True,
            add_bias_2d=True,
            accessory_activation="silu",
            node_transition=False,
            noise_sigma=None,
            
            sm_c_ipa=16,
            sm_c_resnet=128,
            sm_no_heads_ipa=12,
            sm_no_qk_points=4,
            sm_no_v_points=8,
            sm_dropout_rate=0.1,
            sm_no_blocks=8,
            sm_no_transition_layers=1,
            sm_no_resnet_blocks=2,
            sm_no_angles=7,
            sm_trans_scale_factor=10,
            sm_epsilon=1e-8,
            sm_inf=1e5,

            sm_swiglu_transition=False,
            sm_swiglu_transition_hr=2,
            sm_share=True
        )

        dec = dec.to(device)
    #
    #
    #

    for batch in dataloader:
        batch = batch.to(device)
        enc = net(batch)
        print("enc:", enc.shape)
        if use_dec:
            out = dec.nn_forward(enc, batch)
            
            loss = compute_bl_loss(
                sm_hat=out, batch=batch,
                func="l1"
            )
            print("bl_loss:", loss.item())

            loss = compute_e_vdw(
                sm_hat=out, batch=batch,
                # func="l1"
            )
            print("e_vdw:", loss.item())
            
            loss = compute_e_bond(
                sm_hat=out, batch=batch,
                # func="l1"
            )
            print("e_bond:", loss.item())

            loss = compute_nbsr_loss(
                sm_hat=out, batch=batch,
                # func="l1"
            )
            print("nbsr_loss:", loss.item())

        break