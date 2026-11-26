from typing import Tuple, Union, Callable, List
import inspect
import torch
import torch.nn as nn
import numpy as np
from sam.nn.common import (
    AF2_PositionalEmbedding, get_act_fn, FeedForward, get_layer_operation_arch
)
from sam.nn.af3 import Transition, AttentionPairBias, ConditionedTransition
from sam.nn.noise_prediction.embedding import (
    TimestepEmbedderWrapper, TemperatureEmbedderWrapper
)
from sam.nn.noise_prediction.template import (
    TemplateEdgeEmbedder, TemplateNodeEmbedder
)
from sam.nn.noise_prediction.edges import EdgeUpdaterWrapper_v01
from sam.nn.noise_prediction.attention import (
    IdpSAM_AttentionBlock, OriginalIdpSAM_AttentionBlock
)


class DiffusionTransformerBlock_v02(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            token_dim: int,
            node_dim: int,
            edge_dim: int,
            num_heads: int,
            linear_bias: bool = True,
            token_residual: bool = True,
            token_residual_target: str = "hidden",
            edge_residual: bool = True,
            edge_residual_target: str = "hidden",
            attention_mode: str = "idpsam",
            attention_params: dict = {},
            # token_update_addition: bool = True,
            conditioned_transition: bool = True,
            edge_update_mode: str = None,
            edge_update_params: dict = {},
            # edge_update_addition: bool = True,
            edge_transition: bool = True,
            activation: str = "relu"
        ):
        """Arguments: TODO."""
        super().__init__()

        ### Residual modules.
        self.use_token_residual = token_residual
        _check_residual_target(token_residual_target)
        self.token_residual_target = token_residual_target
        if self.use_token_residual:
            # TODO: don't use layernorm if residual target is init.
            self.token_residual_module = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, token_dim, bias=linear_bias)
            )

        self.use_edge_residual = edge_residual
        _check_residual_target(edge_residual_target)
        self.edge_residual_target = edge_residual_target
        if self.use_edge_residual:
            # TODO: don't use layernorm if residual target is init.
            self.edge_residual_module = nn.Sequential(
                nn.LayerNorm(edge_dim),
                nn.Linear(edge_dim, edge_dim, bias=linear_bias)
            )

        ### Edge updater.
        self.use_edge_update = edge_update_mode is not None
        if self.use_edge_update:
            self.edge_updater = EdgeUpdaterWrapper_v01(
                mode=edge_update_mode,
                token_dim=token_dim,
                edge_dim=edge_dim,
                linear_bias=linear_bias,
                activation=activation,
                params=edge_update_params
            )
        # self.edge_update_addition = edge_update_addition

        ### Main attention mechanism.
        self.attention_mode = attention_mode
        if self.attention_mode == "idpsam":
            self.attention_mechanism = OriginalIdpSAM_AttentionBlock(
                token_dim=token_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                linear_bias=linear_bias,
                activation=activation,
                **attention_params
            )
        elif self.attention_mode == "adanorm":
            self.attention_mechanism = IdpSAM_AttentionBlock(
                token_dim=token_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                linear_bias=linear_bias,
                activation=activation,
                **attention_params
            )
        elif self.attention_mode == "apb":
            # self.attention_mechanism = AttentionPairBias(
            #     token_dim=token_dim,
            #     node_dim=node_dim,
            #     edge_dim=edge_dim,
            #     num_heads=num_heads,
            #     linear_bias=linear_bias,
            #     **attention_params
            # )
            raise NotImplementedError()
        else:
            raise KeyError(self.attention_mode)
        # self.token_update_addition = token_update_addition

        ### Transitions.
        # Node transition.
        self.use_conditioned_transition = conditioned_transition
        if self.use_conditioned_transition:
            self.conditioned_transition = ConditionedTransition(
                token_dim=token_dim,
                node_dim=node_dim,
                linear_bias=linear_bias,
                n=2
            )
        # Edge transition.
        self.use_edge_transition = edge_transition
        if self.use_edge_transition:
            self.edge_transition = Transition(
                in_dim=edge_dim, n=2, linear_bias=linear_bias
            )
        
    ### def _edge_update_addition(self, x, x_upt):
    ###     if self.edge_update_addition:
    ###         return x + x_upt
    ###     else:
    ###         return x_upt

    ### def _token_update_addition(self, x, x_upt):
    ###     if self.token_update_addition:
    ###         return x + x_upt
    ###     else:
    ###         return x_upt

    def forward(self, h, node_s, p, h_init, p_init):
        # Residual connections.
        if self.use_token_residual:
            h = self.run_token_residual(h, h_init)
        if self.use_edge_residual:
            p = self.run_edge_residual(p, p_init)

        # Edge update.
        if self.use_edge_update:
            p = self.edge_updater(p, h)
            ### p = self._edge_update_addition(p, q)
            if self.use_edge_transition:
                p = p + self.edge_transition(p)

        # Main token update via attention mechanism.
        b = self.run_attention_mechanism(h=h, node_s=node_s, p=p)

        # Additional (optional) token update.
        if self.use_conditioned_transition:
            h = b + self.conditioned_transition(h, node_s["s"])
        else:
            h = b  ### self._token_update_addition(h, b)

        return h, p
    
    def run_token_residual(self, h, h_init):
        if self.token_residual_target == "hidden":
            h = h_init + self.token_residual_module(h)
        elif self.token_residual_target == "init":
            h = self.token_residual_module(h_init) + h
        else:
            raise KeyError(self.token_residual_target)
        return h

    def run_edge_residual(self, p, p_init):
        if self.edge_residual_target == "hidden":
            p = p_init + self.edge_residual_module(p)
        elif self.edge_residual_target == "init":
            p = self.edge_residual_module(p_init) + p
        else:
            raise KeyError(self.edge_residual_target)
        return p

    def run_attention_mechanism(self, h, node_s, p):
        if self.attention_mode == "idpsam":
            h = self.attention_mechanism(
                h=h,
                h_init=node_s["h_init"],
                s_a=node_s["s_a"],
                s_t=node_s["s_t"],
                s_tem=node_s["s_tem"],
                p=p
            )
        elif self.attention_mode == "adanorm":
            h = self.attention_mechanism(h=h, s=node_s["s"], p=p)
        else:
            raise KeyError(self.attention_mode)
        return h

def _check_residual_target(target):
    if target not in ("init", "hidden"):
        raise KeyError(target)

        
class LatentEpsNetwork_v02(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            # General and input.
            input_dim: int = 16,
            token_dim: int = 256,
            node_dim: int = 128,
            edge_dim: int = 128,
            # mlp_dim: int = None,
            activation: Union[str, Callable] = "relu",

            # Embed input.
            input_embed_mode: str = "linear",  # "linear"
            # input_embed_params: dict = {},

            # Output.
            out_mode: str = "linear",  # "linear", "linear_bias", "mlp"

            # Node.
            node_embed_mode: str = "idpsam",  # "sum_transition", "cat"
            node_mlp_mult: int = 2,
            bead_embed_dim: int = 32,
            num_beads: int = 20,
            # Position.
            pos_embed_r: int = 32,
            use_res_ids: bool = False,
            # Time.
            time_embed_mode: str = "sinusoidal",
            time_embed_dim: int = 256,
            time_embed_params: dict = {},
            # Temperature.
            temperature_embed_mode: str = None,
            temperature_embed_dim: int = 128,
            temperature_embed_params: dict = {},
            # Amino acid embeddings from a large language model.
            pllm_embed_mode: str = None,
            pllm_input_dim: int = None,
            pllm_embed_dim: int = 256,

            # Template.
            tem_inject_mode: str = "xyz",
            tem_inject_params: dict = {
                "inject_edge_mode": "ln",
                "dmap_embed_params": {
                    "cutoff_lower": 0.0,
                    "cutoff_upper": 10.0,
                    "num_rbf": 128,
                    "trainable": True,
                    "type": "expnorm"
                },
                "edge_mlp_mult": 1,
                # "inject_node_mode": "add",
                "node_angle_bins": 16,
                "node_angle_mask": "extra",
                "node_dim": 256,
                "node_embed_resolution": "aa",
                "node_mlp_depth": 2,
                "node_mlp_mult": 1,
            },

            # Edges.
            edge_embed_mode: str = "idpsam",  # "idpsam", "sum"
            edge_update_mode: str = None,
            edge_update_params: dict = {},
            edge_update_freq: Union[int, List[int]] = 1,

            # Transformer blocks.
            num_heads: int = 16,
            num_blocks: int = 16,

            # Attention.
            attention_mode: str = "idpsam",
            attention_params: dict = {},

            # Updates in attention blocks.
            token_residual: bool = False,
            token_residual_target: str = "hidden",
            token_update_addition: bool = False,  # TODO: remove, legacy.
            conditioned_transition: bool = False,
            edge_residual: bool = False,
            edge_residual_target: str = "hidden",
            edge_update_addition: bool = False,  # TODO: remove, legacy.
            edge_transition: bool = False,

            # Misc.
            linear_bias: bool = True
        ):
        """Arguments: TODO."""

        super().__init__()

        ### Check and store the attributes.
        if "idpsam" in (attention_mode, node_embed_mode) and \
            attention_mode != node_embed_mode:
            raise ValueError()
        if node_embed_mode == "idpsam" and edge_embed_mode != "idpsam":
            raise ValueError()
        if node_embed_mode == "idpsam" and conditioned_transition:
            raise ValueError()
        if num_beads is None:
            raise NotImplementedError()
        if tem_inject_mode != "xyz":
            raise NotImplementedError()
            
        self.node_dim = node_dim
        self.node_embed_mode = node_embed_mode
        self.edge_embed_mode = edge_embed_mode
        self.use_res_ids = use_res_ids

        # Setup aa embedding dimensions.
        effective_bead_embed_dim = self._get_node_embed_dim(bead_embed_dim)
        # Time.
        effective_time_embed_dim = self._get_node_embed_dim(time_embed_dim)
        # Temperature.
        if temperature_embed_mode is not None:
            effective_temperature_embed_dim = self._get_node_embed_dim(
                temperature_embed_dim
            )
        else:
            effective_temperature_embed_dim = None
        # Template.
        effective_tem_embed_dim = self._get_node_embed_dim(
            tem_inject_params["node_dim"]
        )
        # PLLM embeddings.
        if pllm_embed_mode is not None:
            if pllm_input_dim is None:
                raise ValueError()
            if pllm_embed_mode == "identity":
                if not node_embed_mode.startswith("cat"):
                    raise ValueError()
                effective_pllm_embed_dim = pllm_input_dim
            else:
                effective_pllm_embed_dim = self._get_node_embed_dim(
                    pllm_embed_dim
                )
        else:
            effective_pllm_embed_dim = None

        ### Process node input.
        # Bead type (amino acid) embedding.
        self.bead_embedder = nn.Embedding(num_beads, effective_bead_embed_dim)
        self.use_bead_embedding = num_beads is not None

        # Time step embedding.
        self.time_embedder = TimestepEmbedderWrapper(
            mode=time_embed_mode,
            embed_dim=effective_time_embed_dim,
            activation=activation,
            params=time_embed_params,
        )

        # Temperature embedding (optional).
        if temperature_embed_mode is not None:
            self.temperature_embedder = TemperatureEmbedderWrapper(
                mode=temperature_embed_mode,
                embed_dim=effective_temperature_embed_dim,
                activation=activation,
                params=temperature_embed_params,
            )
            self.use_temperature = True
        else:
            self.use_temperature = False
        
        # Amino acid embeddings from a PLLM (optional).
        if pllm_embed_mode is not None:
            if pllm_embed_mode == "linear":
                self.pllm_embedder = nn.Linear(
                    pllm_input_dim, effective_pllm_embed_dim, bias=linear_bias
                )
            elif pllm_embed_mode == "mlp":
                self.pllm_embedder = FeedForward(
                    in_dim=pllm_input_dim,
                    hidden_dim=effective_pllm_embed_dim,
                    out_dim=effective_pllm_embed_dim,
                    activation=activation,
                    linear_bias=linear_bias
                )
            elif pllm_embed_mode == "identity":
                self.pllm_embedder = nn.Identity()
            else:
                raise KeyError(pllm_embed_mode)
            self.use_pllm_embeddings = True
        else:
            self.use_pllm_embeddings = False

        # Node features embedding.
        if self.node_embed_mode == "idpsam":
            pass
        elif self.node_embed_mode == "sum":
            self.node_merger = nn.Identity()
        elif self.node_embed_mode == "sum_mlp":
            self.node_merger = nn.Sequential(
                nn.LayerNorm(node_dim),
                FeedForward(
                    in_dim=node_dim,
                    hidden_dim=node_dim*node_mlp_mult,
                    out_dim=node_dim,
                    activation=activation,
                    linear_bias=linear_bias
                )
            )
        elif self.node_embed_mode == "sum_transition":
            self.node_merger = Transition(
                in_dim=node_dim, n=node_mlp_mult, linear_bias=linear_bias
            )
        elif self.node_embed_mode.startswith("cat"):
            node_cat_dim = self._get_cat_input_dim(
                effective_bead_embed_dim,
                effective_time_embed_dim,
                effective_tem_embed_dim,
                effective_temperature_embed_dim,
                effective_pllm_embed_dim
            )
            if self.node_embed_mode == "cat":
                self.node_merger = nn.Linear(node_cat_dim, node_dim)
            elif self.node_embed_mode == "cat_mlp":
                self.node_merger = FeedForward(
                    in_dim=node_cat_dim,
                    hidden_dim=node_dim*node_mlp_mult,
                    out_dim=node_dim,
                    activation=activation,
                    linear_bias=linear_bias
                )
            else:
                raise KeyError(self.node_embed_mode)
        else:
            raise KeyError(self.node_embed_mode)


        ### Template embedding (optional).
        # Embed pair features distances.
        self.tem_edge_embedder = TemplateEdgeEmbedder(
            edge_dim=edge_dim,
            dmap_embed_params=tem_inject_params["dmap_embed_params"],
            com_dmap_embed_params=tem_inject_params.get(
                "com_dmap_embed_params", {}
            ),
            no_dmap_embed_params=tem_inject_params.get(
                "no_dmap_embed_params", {}
            ),
            pair_update_net_params=tem_inject_params.get(
                "pair_update_net_params", {}
            ),
            activation=activation,
            dmap_merge_dim=edge_dim*tem_inject_params.get("edge_mlp_mult", 1),
            linear_bias=linear_bias
        )
        if tem_inject_params.get("inject_edge_mode", "ln") == "ln":
            # TODO: remove the Sequential object here.
            self.tem_edge_inject = nn.Sequential(
                nn.LayerNorm(edge_dim)
            )
        elif tem_inject_params["inject_edge_mode"] is None:
            self.tem_edge_inject = nn.Identity()
        else:
            raise KeyError(tem_inject_params["inject_edge_mode"])

        # Embed node features.
        self.tem_node_embedder = TemplateNodeEmbedder(
            node_dim=effective_tem_embed_dim,
            mode=tem_inject_params["node_embed_resolution"],
            angle_bins=tem_inject_params["node_angle_bins"],
            mask_class=tem_inject_params["node_angle_mask"],
            mlp_mult=tem_inject_params["node_mlp_mult"],
            mlp_depth=tem_inject_params["node_mlp_depth"],
        )
        self.tem_inject_mode = tem_inject_mode  # TODO: change name? It's redundant.
        self.use_template = tem_inject_mode is not None


        ### Process edge input.
        # Embed edge features.
        if self.edge_embed_mode == "idpsam":
            pass
        elif self.edge_embed_mode.startswith("sum"):
            if self.node_embed_mode == "sum":
                self.node_to_edge_layer = nn.LayerNorm(node_dim)
            else:
                self.node_to_edge_layer = nn.Identity()
            self.edge_init_linear_i = nn.Linear(
                node_dim, edge_dim, bias=linear_bias
            )
            self.edge_init_linear_j = nn.Linear(
                node_dim, edge_dim, bias=linear_bias
            )
            if self.edge_embed_mode == "sum":
                self.edge_merger = nn.Identity()
            else:
                raise KeyError(self.edge_embed_mode)
        else:
            raise KeyError(self.edge_embed_mode)

        # Relative positional encodings.
        self.position_embedder = AF2_PositionalEmbedding(
            pos_embed_dim=edge_dim,
            pos_embed_r=pos_embed_r,
            dim_order="trajectory"
        )


        ### Input embedding.
        if input_embed_mode == "linear":
            self.input_embedder = nn.Linear(
                input_dim, token_dim, bias=linear_bias
            )
        elif input_embed_mode == "mlp":
            self.input_embedder = FeedForward(
                in_dim=input_dim,
                hidden_dim=token_dim,
                out_dim=token_dim,
                activation=activation,
                linear_bias=linear_bias
            )
        else:
            raise KeyError(input_embed_mode)


        ### Transformer layers.
        # Select which layers will have an edge updater.
        edge_update_arch = get_layer_operation_arch(
            num_blocks, edge_update_freq, edge_update_mode
        )

        # Set additional (optional) input for specific attention mechanisms.
        if attention_mode == "idpsam":
            attention_params["bead_dim"] = effective_bead_embed_dim
            attention_params["time_dim"] = effective_time_embed_dim
            attention_params["tem_dim"] = tem_inject_params["node_dim"]

        # Add transformer blocks.
        self.blocks = []
        for l in range(num_blocks):
            block_l = DiffusionTransformerBlock_v02(
                token_dim=token_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                linear_bias=linear_bias,
                token_residual=token_residual,
                token_residual_target=token_residual_target,
                edge_residual=edge_residual,
                edge_residual_target=edge_residual_target,
                attention_mode=attention_mode,
                attention_params=attention_params,
                conditioned_transition=conditioned_transition,
                # token_update_addition=token_update_addition,
                edge_update_mode=edge_update_arch[l],
                edge_update_params=edge_update_params,
                edge_transition=edge_transition,
                # edge_update_addition=edge_update_addition,
                activation=activation
            )
            self.blocks.append(block_l)
        self.blocks = nn.ModuleList(self.blocks)


        ### Output module.
        if out_mode in ("linear", "linear_bias"):
            self.out_module = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(
                    token_dim,
                    input_dim,
                    bias=linear_bias if out_mode == "linear" else True
                )
            )
        elif out_mode == "mlp":
            self.out_module = nn.Sequential(
                nn.LayerNorm(token_dim),
                FeedForward(
                    in_dim=token_dim,
                    hidden_dim=token_dim,
                    out_dim=input_dim,
                    activation=activation,
                    linear_bias=linear_bias
                )
            )
        elif out_mode == "idpgan":
            self.out_module = nn.Sequential(nn.Linear(token_dim, token_dim),
                                            get_act_fn(activation)(),
                                            nn.Linear(token_dim, input_dim))
        else:
            raise KeyError(out_mode)

    def _get_node_embed_dim(self, dim):
        if self.node_embed_mode.startswith("sum"):
            if self.node_dim is None:
                raise ValueError()
            return self.node_dim
        else:
            if dim is None:
                raise ValueError()
            return dim
    
    def _get_cat_input_dim(self,
            bead_embed_dim,
            time_embed_dim,
            effective_tem_embed_dim,
            effective_temperature_embed_dim,
            effective_pllm_embed_dim
        ):
        if self.node_embed_mode.startswith("cat"):
            cat_input_dim = bead_embed_dim + time_embed_dim + effective_tem_embed_dim
            if self.use_temperature:
                cat_input_dim += effective_temperature_embed_dim
            if self.use_pllm_embeddings:
                cat_input_dim += effective_pllm_embed_dim
            return cat_input_dim
        else:
            return ValueError()


    def forward(self,
            z_t, t, a=None, r=None, x_tem=None, x_top=None, T=None, a_pllm=None,
            get_cache=False, cache=None,
        ):
        """
        z_t: [B, L, c]
        t: [B, ]
        a: [B, L]
        r: [B, L]
        x_tem: [B, L, 14, 3]
        x_top: [B, L, 14]
        T: [B, ]
        a_pllm: [B, L, e_llm]
        """

        ### Check the input.
        self._check_input(
            z_t=z_t, t=t, a=a, r=r, z_tem=x_tem, z_top=x_top, T=T, a_pllm=a_pllm
        )

        ### Embed input and initialize.

        # Input encoding embedding.
        h_init = self.input_embedder(z_t)
        h = h_init

        # Template embedding.
        if cache is None:
            s_tem, p_tem = self.embed_template(x_tem=x_tem, x_top=x_top)
            if get_cache:
                cache = {"s_tem": s_tem, "p_tem": p_tem}
        else:
            if get_cache:
                raise ValueError()
            s_tem = cache["s_tem"]
            p_tem = cache["p_tem"]
        
        # Node embeddings (conditional information).
        node_s = self.embed_node_input(
            t=t, a=a, h_init=h_init, s_tem=s_tem, T=T, a_pllm=a_pllm
        )

        # Edge embeddings (conditional information).
        p_init = self.embed_edge_input(node_s=node_s, r=r, p_tem=p_tem)
        p = p_init

        ### Transformer layers.
        for block_idx, block_l in enumerate(self.blocks):
            h, p = block_l(
                h=h, node_s=node_s, p=p, h_init=h_init, p_init=p_init
            )

        ### Output layer.
        out = self.out_module(h)
        if not get_cache:
            return out
        else:
            return out, cache

    def _check_input(self, z_t, t, a, r, z_tem, z_top, T, a_pllm):
        if self.use_template and z_tem is None:
            raise ValueError()
        if self.use_temperature and T is None:
            raise ValueError()
        if self.use_pllm_embeddings and a_pllm is None:
            raise ValueError()

    def embed_template(self, x_tem, x_top):
        """
        Embed the 3D structure of the template.
        """
        if self.tem_inject_mode == "xyz":
            p_tem = self.tem_edge_embedder(x_tem, x_top)
            # NOTE: h_tem = self.tem_node_embedder(x_tem, x_top, p_tem)
            h_tem = self.tem_node_embedder(x_tem, x_top)
        else:
            raise KeyError(self.tem_inject_mode)
        return h_tem, p_tem
    
    def embed_node_input(self,
            t, a, h_init, s_tem=None, T=None, a_pllm=None
        ) -> dict:
        """
        Embed node input features.
        """
        s_a = self.bead_embedder(a)
        s_t = self.time_embedder(t)
        s_t = s_t.unsqueeze(1).repeat(1, a.shape[1], 1)

        if self.use_temperature:
            s_T = self.temperature_embedder(T)
            s_T = s_T.unsqueeze(1).repeat(1, a.shape[1], 1)
        if self.use_pllm_embeddings:
            s_pllm = self.pllm_embedder(a_pllm)

        if self.node_embed_mode == "idpsam":
            node_s = {
                "s_a": s_a, "s_t": s_t, "s_tem": s_tem,
                "h_init": h_init  # Only for compatibility with the old idpSAM attention mechanism code.
            }
            if self.use_temperature:
                node_s["s_T"] = s_T
            if self.use_pllm_embeddings:
                raise NotImplementedError()
        elif self.node_embed_mode.startswith("sum"):
            s = s_a + s_t + s_tem
            if self.use_temperature:
                s = s + s_T
            if self.use_pllm_embeddings:
                s = s + s_pllm
            s = self.node_merger(s)
            node_s = {
                "s": s,
                "h_init": h_init  # Only for compatibility, see above.
            }
        elif self.node_embed_mode.startswith("cat"):
            node_args = [s_a, s_t, s_tem]
            if self.use_temperature:
                node_args.append(s_T)
            if self.use_pllm_embeddings:
                node_args.append(s_pllm)
            s_init = torch.cat(node_args, dim=2)
            s = self.node_merger(s_init)
            node_s = {
                "s": s,
                "s_init": s_init,
                "h_init": h_init  # Only for compatibility, see above.
            }
        else:
            raise KeyError(self.node_embed_mode)
        
        return node_s

    def embed_edge_input(self, node_s, r=None, p_tem=None) -> torch.Tensor:
        """
        Embed edge features.
        """
        if self.edge_embed_mode == "idpsam":
            p = self.position_embedder(node_s["h_init"], r=r)
            # NOTE: tem_inject_params["inject_edge_mode"] should be None here.
            p = self.tem_edge_inject(p + p_tem)
        elif self.edge_embed_mode.startswith("sum"):
            s = self.node_to_edge_layer(node_s["s"])
            # Init with outer sum from node input.
            p = self.edge_init_linear_i(s)[:,None,:,:] + \
                self.edge_init_linear_j(s)[:,:,None,:]
            p = p + self.position_embedder(node_s["s"], r=r)
            # NOTE: tem_inject_params["inject_edge_mode"] should be None here.
            p = self.tem_edge_inject(p + p_tem)
            p = self.edge_merger(p)
        else:
            raise KeyError(self.edge_embed_mode)
        return p


class SAM_LatentEpsNetwork_v02(nn.Module):
    """Wrapper for training and inference experiments."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = LatentEpsNetwork_v02(*args, **kwargs)

    def forward(self, xt, t, batch, get_cache=False, cache=None, num_nodes=None):
        eps = self.net.forward(
            z_t=xt,
            t=t,
            a=batch.a,
            r=batch.r,
            x_tem=batch.z_t,
            x_top=batch.z_top,
            T=batch.temperature,
            a_pllm=batch.a_e,
            get_cache=get_cache,
            cache=cache
        )
        return eps


if __name__ == "__main__":

    torch.manual_seed(0)

    # Batch size.
    N = 8
    # Number of residues (sequence length).
    L = 12
    # Encoding dimension.
    e_dim = 32

    # Encoding sequence.
    z = torch.randn(N, L, e_dim)
    x_tem = torch.randn(N, L, 14, 3)

    # Timestep integer values.
    t = torch.randint(0, 1000, (N, ))
    # One-hot encoding for amino acid.
    a = torch.randint(0, 20, (N, L))

    # Large language model embeddings.
    pllm_input_dim = 1280
    a_pllm = torch.randn(N, L, pllm_input_dim)

    # Initialize the network.
    net = LatentEpsNetwork_v02(
        # General and input.
        input_dim=e_dim,
        token_dim=256,
        node_dim=256,
        edge_dim=128,
        activation="silu",
        # Embed input.
        input_embed_mode="mlp",
        # Output.
        out_mode="mlp",
        # Node.
        node_embed_mode="sum_mlp",
        bead_embed_dim=32,
        num_beads=20,
        # Position.
        pos_embed_r=32,
        use_res_ids=True,
        # Time.
        time_embed_mode="sinusoidal",
        time_embed_dim=256,
        time_embed_params={"time_freq_dim": 256},
        # Protein large language model represenations.
        pllm_embed_mode="mlp" if a_pllm is not None else None,
        pllm_input_dim=pllm_input_dim,
        pllm_embed_dim=256,
        # Template.
        tem_inject_mode="xyz",
        tem_inject_params={
            "inject_edge_mode": "ln",
            "dmap_embed_params": {
                "cutoff_lower": 0.0,
                "cutoff_upper": 10.0,
                "num_rbf": 128,
                "trainable": True,
                "type": "expnorm"
            },
            # "inject_node_mode": "add",
            # "pair_update_net_params": {
            #     "what": "cool"
            # },

            "node_angle_bins": 16,
            "node_angle_mask": "extra",
            "node_dim": 256,
            "node_embed_resolution": "aa",
            "node_mlp_depth": 2,
            "node_mlp_mult": 1,
        },
        # Edges.
        edge_embed_mode="idpsam",
        edge_update_mode="esm",
        edge_update_params={
            ## framediff
            # "outer_operation": "sum",
            # "use_in_ln": False,
            # "use_out_ln": False,
            
            ## sam_1
            # "edge_downsample": 1,
            # "node_module": "mlp",
            # "use_ij_ln": True,

            ## esm
            "pair_transition_n": 2,
            "use_tmu": True,
            "use_ta": True,
            "tmu_hidden_dim": 64,
            "ta_hidden_dim": 16,
            "ta_nhead": 4
        },
        edge_update_freq=[5, 10, 15],
        # Transformer blocks.
        num_heads=16,
        num_blocks=22,
        # Attention.
        attention_mode="adanorm",
        attention_params={
            "use_swiglu": True
        },
        # Updates in attention blocks.
        token_residual=True,
        token_residual_target="init",
        conditioned_transition=True,
        edge_residual=True,
        edge_residual_target="init",
        edge_transition=True,
        # Misc.
        linear_bias=True
    )

    out = net(z_t=z, t=t, a=a, x_tem=x_tem, a_pllm=a_pllm)
    print(out.shape)

    out, cache = net(z_t=z, t=t, a=a, x_tem=x_tem, a_pllm=a_pllm, get_cache=True)
    print(out.shape)

    out = net(z_t=z, t=t, a=a, x_tem=x_tem, a_pllm=a_pllm, cache=cache)
    print(out.shape)