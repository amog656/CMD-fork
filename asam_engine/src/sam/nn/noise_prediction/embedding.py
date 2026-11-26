import math
from typing import Callable

import numpy as np

import torch
import torch.nn as nn

from sam.nn.common import get_act_fn
from sam.nn.af3 import FourierEmbedding


###############################################################################
# Adapted from diffusion transformer: https://github.com/facebookresearch/DiT #
###############################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256,
                 activation=nn.SiLU, max_period=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            activation(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        if self.max_period is None:
            t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        else:
            t_freq = self.timestep_embedding(t, self.frequency_embedding_size,
                                             max_period=self.max_period)
        t_emb = self.mlp(t_freq)
        return t_emb


class ReshapeTime(nn.Module):
    def forward(self, t):
        return t.unsqueeze(1)

class TimestepEmbedderWrapper(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            mode: str,
            embed_dim: int,
            activation: str,
            params: dict = {},
        ):
        """Arguments: TODO."""
        super().__init__()
        if mode == "fourier":
            self.embedder = FourierEmbedding(embed_dim)
        elif mode == "mlp":
            hidden_dim = params.get("hidden_dim", embed_dim)
            self.embedder = nn.Sequential(
                ReshapeTime(),
                nn.Linear(1, hidden_dim),
                get_act_fn(activation)(),
                nn.Linear(hidden_dim, embed_dim),
            )
        elif mode == "sinusoidal":
            self.embedder = TimestepEmbedder(
                hidden_size=embed_dim,
                frequency_embedding_size=params.get("time_freq_dim", 256),
                max_period=params.get("time_max_period"),
                activation=get_act_fn(activation),
            )
        else:
            raise KeyError(mode)

    def forward(self, t):
        return self.embedder(t)


class TemperatureEmbedderWrapper(nn.Module):
    """New module documentation: TODO."""

    def __init__(self,
            mode: str,
            embed_dim: int,
            activation: str,
            params: dict
        ):
        """Arguments: TODO."""
        super().__init__()

        if mode in ("scaler", "sinusoidal"):
            self.scaler = {
                "min": params["scaler"]["min"],
                "max": params["scaler"]["max"]
            }
            self.scaler["diff"] = self.scaler["max"] - self.scaler["min"]

        if mode == "scaler":
            self.mlp = nn.Sequential(
                nn.Linear(1, embed_dim),
                get_act_fn(activation)(),
                nn.Linear(embed_dim, embed_dim),
            )
        elif mode == "sinusoidal":
            self.embedder = TimestepEmbedder(
                hidden_size=embed_dim,
                frequency_embedding_size=params.get("freq_dim", 256),
                max_period=params.get("max_period"),
                activation=get_act_fn(activation),
            )
        else:
            raise KeyError(mode)
        self.mode = mode

    def forward(self, T):
        T_scale = (T - self.scaler["min"]) / self.scaler["diff"]
        if self.mode == "scaler":
            s_T = self.mlp(T_scale.unsqueeze(-1))
        elif self.mode == "sinusoidal":
            T_scale = T_scale*1000.0
            s_T = self.embedder(T_scale)
        else:
            raise KeyError(mode)
        return s_T


###########################################################################
# Conditional information injection (timestep and amino acid embeddings). #
###########################################################################

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ConditionalInjectionModule(nn.Module):

    def __init__(self,
                 mode: str,
                 embed_dim: int = 256,
                 bead_embed_dim: int = 256,
                 time_embed_dim: int = 256,
                 activation: Callable = nn.SiLU,
                 # mlp_ratio: float = 4.0,
                 norm_pos: str = "pre"):
        """
        `bead_embed_dim`: if set to 'None', the module will be amino acid
            unconditioned.
        `mlp_ratio`: used for with `adanorm`.
        """

        super().__init__()
        self.mode = mode
        self.bead_embed_dim = bead_embed_dim
        self.norm_pos = norm_pos
        if self.mode in ("adanorm", "adanorm_fix"):
            if bead_embed_dim is not None:
                if bead_embed_dim != time_embed_dim:
                    self.bead_project = nn.Linear(bead_embed_dim, time_embed_dim)
                else:
                    self.bead_project = nn.Identity()
            # self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            # self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            # mlp_hidden_dim = int(embed_dim * mlp_ratio)
            # approx_gelu = lambda: nn.GELU(approximate="tanh")
            # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            self.adaLN_modulation = nn.Sequential(
                activation(),
                nn.Linear(time_embed_dim, 6 * embed_dim, bias=True)
            )
            if self.mode == "adanorm_fix":
                self.initialize_weights()

        elif self.mode == "concat": 
            if self.norm_pos == "pre":
                raise NotImplementedError()
        elif self.mode == "add":
            if self.norm_pos == "post":
                raise NotImplementedError()
            if bead_embed_dim is not None:
                if bead_embed_dim != time_embed_dim:
                    self.bead_project = nn.Linear(bead_embed_dim, time_embed_dim)
                else:
                    self.bead_project = nn.Identity()
            self.seq_project = nn.Linear(time_embed_dim, embed_dim)
        else:
            raise KeyError(mode)
    
    def initialize_weights(self):
        if self.mode in ("adanorm", "adanorm_fix"):
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        elif self.mode in ("concat", "add"):
            pass
        else:
            raise KeyError(self.mode)
    
    def forward(self, a, t):
        if self.mode in ("adanorm", "adanorm_fix"):
            # We also experiment with a layer [43] that we refer to as adaptive group
            # normalization (AdaGN), which incorporates the timestep and class embedding
            # into each residual block after a group normalization operation [69],
            # similar to adaptive instance norm [27] and FiLM [48]. We define this
            # layer as AdaGN(h,y)=ys GroupNorm(h)+yb, where h is the intermediate
            # activations of the residual block following the first convolution,
            # and y = [ys , yb ] is obtained from a linear projection of the
            # timestep and class embedding.
            
            # To embed input timesteps, we use a 256-dimensional frequency embedding [9]
            # followed by a two-layer MLP with dimensionality equal to the transformer’s
            # hidden size and SiLU activations.
            # Each adaLN layer feeds the sum of the timestep and class embeddings into
            # a SiLU nonlinearity and a linear layer with output neurons equal to
            # either 4× (adaLN) or 6× (adaLN-Zero) the transformer’s hidden size.
            # We use GELU nonlinearities (approximated with tanh) in the core
            # transformer [16].
            if self.bead_embed_dim is not None:
                c = self.bead_project(a) + t
            else:
                c = t
            adaLN_m_r = self.adaLN_modulation(c).chunk(6, dim=2)
            out = {"shift_msa": adaLN_m_r[0],
                   "scale_msa": adaLN_m_r[1],
                   "gate_msa": adaLN_m_r[2],
                   "shift_mlp": adaLN_m_r[3],
                   "scale_mlp": adaLN_m_r[4],
                   "gate_mlp": adaLN_m_r[5]}
        elif self.mode in ("concat", "add"):
            out = {"a": a, "t": t}
        else:
            raise KeyError(self.mode)
        return out
    
    def inject_1_proto(self, x, inj_out):
        if self.mode in ("adanorm", "adanorm_fix"):
            return x
        elif self.mode == "concat":
            return x
        elif self.mode == "add":
            if self.bead_embed_dim is not None:
                return x + self.seq_project(inj_out["t"] + self.bead_project(inj_out["a"]))
            else:
                return x + self.seq_project(inj_out["t"])
        else:
            raise KeyError(self.mode)

    def inject_1_pre(self, x, inj_out):
        if self.mode in ("adanorm", "adanorm_fix"):
            return modulate(x, inj_out["shift_msa"], inj_out["scale_msa"])
        elif self.mode == "concat":
            return x
        elif self.mode == "add":
            return x
        else:
            raise KeyError(self.mode)

    def inject_1_post(self, x, inj_out):
        if self.mode in ("adanorm", "adanorm_fix"):
            return x * inj_out["gate_msa"]
        elif self.mode == "concat":
            return x
        elif self.mode == "add":
            return x
        else:
            raise KeyError(self.mode)

    def inject_2_pre(self, x, inj_out):
        if self.mode in ("adanorm", "adanorm_fix"):
            return modulate(x, inj_out["shift_mlp"], inj_out["scale_mlp"])
        elif self.mode == "concat":
            if self.bead_embed_dim is not None:
                return torch.cat([x, inj_out["a"], inj_out["t"]], axis=2)
            else:
                return torch.cat([x, inj_out["t"]], axis=2)
        elif self.mode == "add":
            return x
        else:
            raise KeyError(self.mode)

    def inject_2_post(self, x, inj_out):
        if self.mode in ("adanorm", "adanorm_fix"):
            return x * inj_out["gate_mlp"]
        elif self.mode == "concat":
            return x
        elif self.mode == "add":
            return x
        else:
            raise KeyError(self.mode)


###########################################################################
# Input information injection.                                            #
###########################################################################

class InputInjectionModule(nn.Module):

    def __init__(self,
                 mode: str,
                 input_dim: int = 16,
                 embed_dim: int = 256,
                 # bead_embed_dim: int = 256,
                 time_embed_dim: int = 256,
                 activation: Callable = nn.SiLU,
                 mlp_ratio: float = 4.0,
                 # norm_pos: str = "pre"
                 ):
        """
        `mlp_ratio`: used for with `adanorm`.
        """

        super().__init__()
        self.mode = mode

        if self.mode == "add":
            self.input_project = nn.Linear(input_dim, embed_dim)
            self.input_inject_norm = nn.LayerNorm(embed_dim)

        elif self.mode in ("adanorm", "adanorm_fix"):
            self.input_project = nn.Linear(input_dim, embed_dim)
            self.time_project = nn.Linear(time_embed_dim, embed_dim)

            self.norm = nn.LayerNorm(embed_dim,
                                      elementwise_affine=False,
                                      #eps=1e-6
                                    )
            mlp_dim = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_dim),
                                     activation(),
                                     nn.Linear(mlp_dim, embed_dim))
            self.adaLN_modulation = nn.Sequential(
                activation(),
                nn.Linear(embed_dim, 3 * embed_dim, bias=True)
            )
            if self.mode == "adanorm_fix":
                self.initialize_weights()
        else:
            raise KeyError(mode)
    
    def initialize_weights(self):
        if self.mode == "add":
            pass
        elif self.mode in ("adanorm", "adanorm_fix"):
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            raise KeyError(self.mode)
    
    def forward(self, x, x_0, t):
        if self.mode == "add":
            x = x + self.input_project(x_0)
            x = self.input_inject_norm(x)
            return x
        elif self.mode in ("adanorm", "adanorm_fix"):
            residual = x
            c = self.input_project(x_0) + self.time_project(t)
            adaLN_m_r = self.adaLN_modulation(c).chunk(3, dim=2)
            shift_mlp = adaLN_m_r[0]
            scale_mlp = adaLN_m_r[1]
            gate_mlp = adaLN_m_r[2]
            x = self.norm(x)
            x = modulate(x, shift_mlp, scale_mlp)
            x = self.mlp(x)
            x = x * gate_mlp
            return residual + x
        else:
            raise KeyError(self.mode)