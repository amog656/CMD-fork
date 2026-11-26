import math
from typing import Tuple, Union, Callable

import numpy as np

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

from sam.nn.common import get_act_fn, AF2_PositionalEmbedding, FeedForward
from sam.nn.common import FeedForward, SwiGLU, TransformerBlock_v01
from sam.data.sequences import get_num_beads
from sam.openfold.model.structure_module import StructureModule
from sam.data.aa_topology import get_traj_list, sam_openfold_aa_map


################################################################################
# New decoder network.                                                         #
################################################################################

class AllAtomDecoder_v01(nn.Module):

    def __init__(
        self,
        encoding_dim: int = 16,
        num_blocks: int = 4,
        # attention_type: str = "transformer",
        node_dim: int = 256,
        edge_dim: int = 128,
        num_heads: int = 16,
        mlp_dim: int = None,
        activation: Union[str, Callable] = "swiglu",
        out_mode: str = "simple",
        embed_inject_mode: str = None,
        bead_embed_dim: int = 32,
        num_beads: int = 20,
        pos_embed_r: int = 32,
        use_res_ids: bool = False,
        node_init_mode: str = "linear",
        linear_bias: bool = True,
        add_bias_2d: bool = True,
        accessory_activation: Union[str, Callable] = "silu",
        node_transition: bool = False,
        noise_sigma: float = None,
        block_transition: str = None,
        
        sm_c_ipa: int = 16,
        sm_c_resnet: int = 128,
        sm_no_heads_ipa: int = 12,
        sm_no_qk_points: int = 4,
        sm_no_v_points: int = 8,
        sm_dropout_rate: float = 0.1,
        sm_no_blocks: int = 8,
        sm_no_transition_layers: int = 1,
        sm_no_resnet_blocks: int = 2,
        sm_no_angles: int = 7,
        sm_trans_scale_factor: int = 10,
        sm_epsilon: float = 1e-8,
        sm_inf: float = 1e5,

        sm_custom: bool = False,
        sm_transition_mode: str = "original",
        sm_transition_activation: str = None,
        sm_swiglu_transition: bool = False,
        sm_swiglu_transition_hr: int = 2,
        sm_agle_resnet_activation: str = None,
        sm_share: bool = True,
        sm_self_attention: bool = False,
        sm_no_heads_sa: int = 16,

        output_dim: int = None  # Legacy, actually not used.
        ):
        """
        `encoding_dim`: dimension of the structural encoding vectors.
        """

        super().__init__()

        ### Check and store the attributes.
        self.node_dim = node_dim
        self.use_res_ids = use_res_ids
        self.noise_sigma = noise_sigma
        
        ### Process input.
        if node_init_mode == "linear":
            self.node_init_project = nn.Linear(
                encoding_dim, node_dim, bias=linear_bias
            )
        elif node_init_mode == "mlp":
            self.node_init_project = FeedForward(
                in_dim=encoding_dim,
                hidden_dim=node_dim,  # node_dim*2
                out_dim=node_dim,
                activation=accessory_activation,
                linear_bias=linear_bias
            )
        else:
            raise KeyError(node_init_mode)

        ### Amino acid embedding.
        if embed_inject_mode is not None:
            self.beads_embedder = nn.Embedding(num_beads, bead_embed_dim)
            if embed_inject_mode == "outer_sum":
                self.edge_init_linear_i = nn.Linear(
                    bead_embed_dim, edge_dim, bias=linear_bias
                )
                self.edge_init_linear_j = nn.Linear(
                    bead_embed_dim, edge_dim, bias=linear_bias
                )
                self.pair_ln = nn.LayerNorm(edge_dim)
            else:
                raise KeyError(embed_inject_mode)
        else:
            self.beads_embedder = None
        self.embed_inject_mode = embed_inject_mode

        self.register_buffer(
            "sam_openfold_aa_map", 
            torch.tensor(sam_openfold_aa_map)
        )

        ### Positional embeddings.
        self.embed_pos = AF2_PositionalEmbedding(
            pos_embed_dim=edge_dim,
            pos_embed_r=pos_embed_r,
            dim_order="trajectory"
        )
        
        ### Transformer layers.
        self.layers = []
        for l in range(num_blocks):
            # TODO: replace with a module similar to the encoder one (AllAtomEncoderBlock_v01)?
            layer_l = TransformerBlock_v01(
                embed_dim=node_dim,
                edge_dim=edge_dim,
                mlp_dim=mlp_dim if mlp_dim is not None else node_dim*2,  # node_dim*4
                num_heads=num_heads,
                activation=activation,
                linear_bias=linear_bias,
                add_bias_2d=add_bias_2d
            )
            self.layers.append(layer_l)
        self.layers = nn.ModuleList(self.layers)

        ### Output module of the transformer trunk.
        if block_transition is None:
            self.transition_module = nn.Identity()
        elif block_transition == "mlp":
            self.transition_module = nn.Sequential(
                nn.LayerNorm(node_dim),
                FeedForward(
                    in_dim=encoding_dim,
                    hidden_dim=node_dim,  # node_dim*2
                    out_dim=node_dim,
                    activation=accessory_activation,
                    linear_bias=linear_bias
                )
            )
        else:
            raise KeyError(block_transition)
        
        ### Structure module.
        if StructureModule is None:
            raise ImportError("Openfold can't be imported")

        if sm_custom:
            structure_module_cls = SAM_StructureModule
            sm_kwargs = {
                "transition_mode": sm_transition_mode,
                "transition_activation": sm_transition_activation,
                "swiglu_transition_hr": sm_swiglu_transition_hr,
                "agle_resnet_activation": sm_agle_resnet_activation,
                "self_attention": sm_self_attention,
                "no_heads_sa": sm_no_heads_sa,
                "share": sm_share,
            }
            if sm_swiglu_transition:
                sm_kwargs["transition_mode"] = "swiglu"
        else:
            structure_module_cls = StructureModule
            sm_kwargs = {}
            
        self.structure_module = structure_module_cls(
            c_s=node_dim,
            c_z=edge_dim,
            c_ipa=sm_c_ipa,
            c_resnet=sm_c_resnet,
            no_heads_ipa=sm_no_heads_ipa,
            no_qk_points=sm_no_qk_points,
            no_v_points=sm_no_v_points,
            dropout_rate=sm_dropout_rate,
            no_blocks=sm_no_blocks,
            no_transition_layers=sm_no_transition_layers,
            no_resnet_blocks=sm_no_resnet_blocks,
            no_angles=sm_no_angles,
            trans_scale_factor=sm_trans_scale_factor,
            epsilon=sm_epsilon,
            inf=sm_inf,
            **sm_kwargs
        )


    def get_embed_dim(self):
        return self.embed_dim


    def forward(self, z, a=None, r=None):
        """
        z: input tensor with shape (B, L, E).
        a: amino acid tensor with shape (B, L). It should be set to 'None' if
            'embed_inject_mode' is also 'None'.
        """

        ### Input.
        h_node = self.node_init_project(z)
        if self.noise_sigma is not None:
            h_node = h_node + self.noise_sigma*torch.randn_like(h)

        ### Positional embeddings.
        p = self.embed_pos(z, r=r)

        ### Bead embeddings.
        if self.embed_inject_mode is not None:
            h_a = self.beads_embedder(a)
            if self.embed_inject_mode == "outer_sum":
                h_a_2d = self.edge_init_linear_i(h_a)[:,None,:,:] + \
                         self.edge_init_linear_j(h_a)[:,:,None,:]
                p = self.pair_ln(p + h_a_2d)
            else:
                raise KeyError(embed_inject_mode)

        ### Go through all the transformer blocks.
        # TODO: inject input?
        for layer_idx, layer in enumerate(self.layers):
            h_node = layer(x=h_node, p=p)

        ### Output module of the transformer trunk.
        h_node = self.transition_module(h_node)

        ### Structure module.
        a_openfold = self.sam_openfold_aa_map[a]
        
        structure = self.structure_module(
            evoformer_output_dict={"single": h_node, "pair": p},
            aatype=a_openfold,
            mask=None,
            inplace_safe=False,
            _offload_inference=False
        )
        structure["a"] = a_openfold

        return structure


    def nn_forward(self, e, batch, num_nodes=None):
        if num_nodes is not None:
            raise NotImplementedError()
        a = batch.a
        r = batch.r if self.use_res_ids else None
        sm_hat = self.forward(z=e, a=a, r=r)
        return sm_hat

###############################################################################


if __name__ == "__main__":
    
    torch.manual_seed(0)

    # Batch size.
    N = 128
    # Number of residues (sequence length).
    L = 32
    # Encoding dimension.
    e_dim = 16

    z = torch.randn(N, L, e_dim)
    a = torch.randint(0, 20, (N, L))

    net = AllAtomDecoder_v01(
        encoding_dim=e_dim,

        num_blocks=4,
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

        sm_custom=True,
        sm_transition_mode="original",
        sm_transition_activation="silu",
        sm_agle_resnet_activation="silu",
        sm_self_attention=True,
        sm_no_heads_sa=16,
        sm_share=True,

    )
    out = net(z=z, a=a)
    out["positions"] = out["positions"][-1]
    data = get_traj_list(out, a=a)
    print(data[0])