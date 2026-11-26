import inspect

from sam.nn.autoencoder.encoder.aa import AllAtomEncoder_v01
from sam.data.sequences import get_num_beads


def get_encoder(model_cfg, output_dim=None):
    """
    Returns an object of the encoder class specified in `model_cfg`.
    """

    # Use a coarse-grained representation.
    if model_cfg["encoder"]["arch"] == "enc_aa_v01":
        enc_class = AllAtomEncoder_v01
    else:
        raise KeyError(model_cfg["encoder"]["arch"])

    # Get the arguments of the encoder network class.
    args = list(
        inspect.signature(enc_class.__init__).parameters.keys())
    args.remove("encoding_dim")
    # Get from 'model_cfg' the corresponding arguments.
    params = {}
    for arg in args:
        if arg in model_cfg["encoder"]:
            params[arg] = model_cfg["encoder"][arg]
    # Initialize the network.
    return enc_class(
        encoding_dim=output_dim if output_dim is not None \
                     else model_cfg["generative_stack"]["encoding_dim"],
        use_res_ids=model_cfg.get("data", {}).get("res_ids_mode") is not None,
        num_beads=get_num_beads(model_cfg.get("data", {}).get("alphabet")),
        **params)