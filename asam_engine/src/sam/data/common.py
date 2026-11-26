import pathlib
import json
import glob
import re
from typing import Union, List


traj_file_exts = ("dcd", "xtc", )
enc_file_exts = ("pt", )
ofo_file_exts = ("pt", )

def get_input_list(
    input: Union[List[str], List[dict], str],
    re_filter: str = None,
    proteins: Union[list, str] = None,
    data_type: str = "xyz",
    quick_mode: bool = True
    ):
    """
    Returns:
    `input_list`: a list of dictionaries of the form:
        {"name": "name_of_the_protein",
         "topology": "/path/to/topology_file.pdb",
         "trajectories": ["/path/to/trajectory_0.dcd",
                          "/path/to/trajectory_1.dcd",
                          "/path/to/trajectory_*.xtc"]}
    to be used in the dataset classes for collecting data.
    """

    # Initialize.
    if data_type == "xyz":
        default_prefix = "traj"
        file_exts = traj_file_exts
    elif data_type == "enc":
        default_prefix = "enc"
        file_exts = enc_file_exts
    elif data_type == "ofo":
        default_prefix = "traj"
        file_exts = ofo_file_exts
    else:
        raise ValueError(data_type)

    # Get the names of the selected protein systems (will be used later).
    _proteins = get_proteins(proteins)

    # Select auto or manual mode.
    if isinstance(input, str):
        dataset_path = pathlib.Path(input)
        if dataset_path.is_dir():
            # Will look in a single directory for xyz data files organized in a
            # pre-defined structure.
            auto_mode = True
        elif dataset_path.is_file():
            # Each line of the input file should point to a JSON file with data
            # of a protein system.
            auto_mode = False
            _input = []
            with open(dataset_path, "r") as i_fh:
                for line in i_fh:
                    line = line.rstrip()
                    if line and not line.startswith("#"):
                        _input.append(line)
            input = _input
        else:
            raise FileNotFoundError("Not a file or directory: {dataset_path}")
    else:
        auto_mode = False

    #-------------
    # Auto mode. -
    #-------------
    
    if auto_mode:
        # Scans the xyz directory for protein directories.
        if not quick_mode:
            iterable = dataset_path.glob("./*")
        else:
            iterable = []
            if not _proteins:
                # Falls back to searching all protein systems in the dataset
                # directory if 'None' was provided as the 'proteins' argument.
                iterable = dataset_path.glob("./*")
            else:
                for prot_sys in _proteins:
                    iterable.append(dataset_path / prot_sys)
        input_list = []
        for prot_path in iterable:
            # Looks for a topology file named "top.pdb".
            top_path = prot_path / "top.pdb"
            if not top_path.is_file():
                continue
            # Looks for all trajectory files with names starting with
            # "traj". Also looks for optional template files.
            traj_l = []
            tem_l = []
            for file_ext in file_exts:
                auto_pattern = f"./{default_prefix}_*.{file_ext}"
                for traj_path in prot_path.glob(auto_pattern):
                    traj_l.append(str(traj_path))
                ### tem_pattern = f"template_*.{file_ext}"
                ### for tem_path in prot_path.glob(tem_pattern):
                ###     tem_l.append(tem_path)
            if not traj_l:
                continue
            input_i = {
                "name": prot_path.name,
                "topology": str(top_path),
                "trajectories": traj_l
            }
            # Looks for a topology file named "top.pt" for an OpenFold dataset.
            if data_type == "ofo":
                ofo_top_path = prot_path / "top.pt"
                if not ofo_top_path.is_file():
                    continue
                input_i["ofo_topology"] = str(ofo_top_path)
            ### if tem_l:
            ###     tem_l.sort(
            ###         key=lambda p: int(re.search("template_(\d+)", p.name).groups()[0])
            ###     )
            ###     input_i["template"] = [str(p) for p in tem_l]
            if data_type == "enc":
                with open(prot_path / "enc.json", "r") as i_fh:
                    enc_data = json.load(i_fh)
                    input_i["seq"] = enc_data["seq"]
            input_list.append(input_i)
        if not input_list:
            raise ValueError(
                f"No valid protein data directory found in: {dataset_path}"
            )

    #----------------------------------------------------------
    # Manual mode: privide a list of dictionaries or strings. -
    #----------------------------------------------------------

    elif isinstance(input, list):

        # List of dictionaries.
        if all([isinstance(i, dict) for i in input]):
            input_list = input

        # List of strings specifying file paths.
        elif all([isinstance(i, str) for i in input]):
            input_list = []
            for input_i in input:
                # Assume the string is a filepath or a glob expression for
                # multiple filepaths.
                _input_i = _glob(input_i)
                if not _input_i:
                    raise ValueError(f"No files in the expression: {input_i}")
                for _input_fp_ij in _input_i:
                    # Check the input.
                    if _input_fp_ij.endswith(".json"):
                        with open(_input_fp_ij, "r") as i_fh:
                            json_data = json.load(i_fh)
                        input_list.append(json_data)
                    else:
                        raise ValueError(
                            f"File {input_i} doesn't have a .json extension"
                        )
        else:
            raise TypeError()

    else:
        raise TypeError(type(input))


    #---------------------------------
    # Construct the trajectory list. -
    #---------------------------------

    # Filter by protein name.
    if _proteins is not None:
        if auto_mode and quick_mode:
            # Protein systems have already been selected when using quick mode.
            pass
        else:
            input_list = [in_i for in_i in input_list if in_i["name"] in _proteins]
    if not input_list:
        raise ValueError(
            "No trajectories files were found for the selected proteins"
        )
    
    # Get the files from a list that can look like:
    #    ["/home/user/traj_data_0/traj_0.dcd",
    #     "/home/user/traj_data_0/traj_1.dcd",
    #     "/home/user/traj_data_1/traj_*.dcd""]
    for input_i in input_list:
        traj_l = []
        for traj_pattern in input_i["trajectories"]:
            for t in _glob(traj_pattern):
                traj_l.append(t)
        if not traj_l:
            raise ValueError(
                f"No trajectories files in input data for {input_i['name']}"
            )
        # Filter trajectories by some pattern in their file paths.
        if re_filter is not None:
            traj_l = [t for t in traj_l if re.search(re_filter, t) is not None]
        if not traj_l:
            raise ValueError(
                f"No trajectories files after applying a regular expression"
                " filter on file names"
            )
        input_i["trajectories"] = traj_l

    return input_list


def _glob(pattern, use_glob=False):
    if use_glob:
        return glob.glob(pattern)
    else:
        return [pattern]


def get_proteins(proteins):
    # Get the list of selected proteins.
    if proteins is None:
        _proteins = None
    elif isinstance(proteins, (list, tuple)):
        _proteins =  proteins
    elif isinstance(proteins, str):
        _proteins = []
        with open(proteins, "r") as i_fh:
            for l in i_fh:
                if l.startswith("#") or not l.rstrip():
                    continue
                _proteins.append(l.rstrip())
    else:
        raise TypeError(proteins.__class__)
    return _proteins