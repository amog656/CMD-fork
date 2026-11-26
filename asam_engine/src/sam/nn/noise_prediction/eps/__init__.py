import inspect
from sam.data.sequences import get_num_beads
from sam.nn.noise_prediction.eps.eps_simple import (
    LatentEpsNetwork_v02, SAM_LatentEpsNetwork_v02
)

def get_eps_network(model_cfg):
    # Get the class for the noise prediction network.
    if model_cfg["generator"]["arch"] == "eps_v02":
        model_cls = LatentEpsNetwork_v02
        wrapper_cls = SAM_LatentEpsNetwork_v02
    else:
        raise KeyError(model["generator"]["arch"])
    # Get the arguments of the eps network class.
    eps_args = list(inspect.signature(model_cls.__init__).parameters.keys())
    eps_args.remove("input_dim")
    # Get from 'model_cfg' the corresponding arguments.
    eps_params = {}
    for eps_arg in eps_args:
        if eps_arg in model_cfg["generator"]:
            eps_params[eps_arg] = model_cfg["generator"][eps_arg]
    # Initialize the network.
    return wrapper_cls(
        input_dim=model_cfg["generative_stack"]["encoding_dim"],
        use_res_ids=model_cfg.get("data", {}).get("res_ids_mode") is not None,
        num_beads=get_num_beads(model_cfg.get("data", {}).get("alphabet")),
        **eps_params
    )
