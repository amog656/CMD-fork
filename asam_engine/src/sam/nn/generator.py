import os
from typing import Callable
import torch
from sam.nn.noise_prediction.eps import get_eps_network
from sam.diffusion.diffusers_dm import Diffusers


def get_generator_net(model_cfg: dict):
    generator = get_eps_network(model_cfg)
    return generator


def get_generative_model(model_cfg: dict,
                         network: Callable,
                         ema: Callable = None):

    # Diffusion modeling using the Diffusers library.
    if model_cfg["generative_model"]["type"] == "diffusers_dm":
        """
        model = get_diffusion_model(model_cfg=model_cfg, network=network, ema=ema)
        """
        model = Diffusers(
            eps_model=network,
            sched_params=model_cfg["generative_model"]["sched_params"],
            loss=model_cfg["generative_model"].get("loss", "l2"),
            extra_loss=model_cfg["generative_model"].get("extra_loss", {}),
            ema=ema,
            sc_params=None
        )
    else:
        raise NotImplementedError()
    return model