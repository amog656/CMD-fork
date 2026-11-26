from typing import Callable
try:
    from torch_ema import ExponentialMovingAverage as EMA
    has_ema = True
except ImportError:
    has_ema = False


def get_ema(network: Callable, model_cfg: dict, network_key: str = "generator"):
    if model_cfg[network_key].get("ema"):
        if not has_ema:
            raise ImportError("torch_ema is not installed")
        ema = EMA(
            network.parameters(), decay=model_cfg[network_key]["ema"]["beta"]
        )
    else:
        ema = None
    return ema