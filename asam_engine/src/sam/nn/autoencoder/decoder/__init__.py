import inspect

from sam.nn.autoencoder.decoder.aa import AllAtomDecoder_v01
from sam.data.sequences import get_num_beads


def get_decoder(model_cfg, input_dim=None, output_dim=None):
    """
    Returns an object of the decoder class specified in `model_cfg`.
    """

    # Use a coarse-grained representation.
    if model_cfg["decoder"]["arch"] == "dec_aa_v01":
        dec_class = AllAtomDecoder_v01
    else:
        raise KeyError(model_cfg["decoder"]["arch"])

    # Get the arguments of the decoder network class.
    args = list(
        inspect.signature(dec_class.__init__).parameters.keys())
    args.remove("encoding_dim")
    # Get from 'model_cfg' the corresponding arguments.
    params = {}
    for arg in args:
        if arg in model_cfg["decoder"]:
            params[arg] = model_cfg["decoder"][arg]
    # Initialize the network.
    return dec_class(
        encoding_dim=input_dim if input_dim is not None \
                     else model_cfg["generative_stack"]["encoding_dim"],
        output_dim=output_dim if output_dim is not None else 3,
        use_res_ids=model_cfg.get("data", {}).get("res_ids_mode") is not None,
        num_beads=get_num_beads(model_cfg.get("data", {}).get("alphabet")),
        **params
    )