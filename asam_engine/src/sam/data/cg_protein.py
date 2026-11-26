import os
import pathlib
import json
import pickle
import shutil
from collections import namedtuple
from typing import Callable, Union, List, Dict
import time
import numpy as np
import mdtraj
import torch
from sam.utils import print_msg
from sam.data.sequences import (aa_list,
                                gpcrmd_alphabet_three_letters,
                                aa_three_to_one_dict,
                                check_alphabet,
                                is_standard_alphabet)
from sam.data.common import get_input_list
from sam.data.topology import slice_ca_traj, get_atom14_sam_data


################################################################################
# Code for storing molecular data in the batches built in dataloaders.         #
################################################################################

# Classes derived from these namedtuples will be used to instantiate the objects
# returned by the .get() methods of the dataset classes in this module.
staticdata_xyz_keys = ["x", "a", "a_e", "r", "x_t"]
_StaticData = namedtuple("StaticData",
                         staticdata_xyz_keys,
                         defaults=[0, 0, 0, 0, 0])

staticdata_enc_keys = ["z", "a", "a_e", "r", "z_t", "z_top", "temperature"]
_StaticDataEnc = namedtuple("StaticDataEnc",
                            staticdata_enc_keys,
                            defaults=[0, 0, 0, 0, 0, 0, 0])

class StaticDataMixin:
    """
    Class to emulate torch_geometric batches using data from regular pytorch
    batches.
    """

    def to(self, device):
        return self._replace(
            **{k: getattr(self, k).to(device) for k in self._fields}
        )
    
    @property
    def num_graphs(self):
        return self.sel_val.shape[0]

    @property
    def num_nodes(self):
        return self.sel_val.shape[0]*self.sel_val.shape[1]

    @property
    def batch(self):
        return torch.arange(0, self.sel_val.shape[0],
                            device=self.sel_val.device)
    
    @property
    def device(self):
        return self.sel_val.device

    def select_ids(self, ids):
        return self.__class__(
            **{k: getattr(self, k)[ids] for k in self._fields})

    def _get_str(self, k):
        o = getattr(self, k)
        if hasattr(o, "shape"):
            return tuple(getattr(self, k).shape)
        else:
            return "null"

    def __str__(self):
        cls_name = self.__class__.__name__
        obj_repr = " ".join(["{}={}".format(k, self._get_str(k)) \
                             for k in self._fields])
        return "{}: {}".format(cls_name, obj_repr)

    
class StaticData(_StaticData, StaticDataMixin):
    __slots__ = ()

    @property
    def sel_val(self):
        return self.x

class StaticDataEnc(_StaticDataEnc, StaticDataMixin):
    __slots__ = ()

    @property
    def sel_val(self):
        return self.z


################################################################################
# Common code for molecular and encoded datasets.                              #
################################################################################

class ProteinDataFiles:
    """
    Class for storing the paths of data files of a protein. During each training
    epoch a random subsets of all the conformations stored in these files will
    be selected.
    """
    def __init__(self,
        input_data: dict,
        data_type: str = None):
        # Name and topology file.
        self.name = input_data["name"]
        self.top_fp = input_data["topology"]
        self.seq = input_data.get("seq")
        self.trajectories = input_data["trajectories"]
        self.template = input_data.get("template")
        self.attributes = input_data.get("attrs", {})
        # Only for OpenFold pre-processed dataset.
        self.ofo_top_fp = input_data.get("ofo_topology")


class CG_Protein:
    """
    Class to store data for a protein molecule used in a training epoch. The
    'xyz' and 'enc' attributes will store arrays representing protein
    conformations used in one epoch.
    """

    def __init__(
            self,
            name: str,
            seq: Union[str, List[str]],
            xyz: np.ndarray,
            r: np.ndarray = None,
            xyz_tem: np.ndarray = None,
            aa_emb: np.ndarray = None,
            alphabet: str = None,
            attributes: dict = {}
        ):

        # Main attributes.
        self.name = name
        check_alphabet(alphabet)
        self.alphabet = alphabet

        # Get the amino acid type tensor.
        if is_standard_alphabet(self.alphabet):  # Standard amino acids.
            if isinstance(seq, str):
                seq_one_letter = seq
            elif isinstance(seq, list):
                seq_one_letter = "".join([aa_three_to_one_dict[r] for r in seq])
            else:
                raise TypeError(seq.__class__)
            self.a = get_features_from_seq(seq_one_letter).argmax(0)
        else:  # Modified residue alphabet.
            if isinstance(seq, list):
                if self.alphabet == "gpcrmd":
                    self.a = get_gpcrmd_aa_features(seq)
                else:
                    raise KeyError()
            else:
                raise TypeError(seq.__class__)
            seq_one_letter = "".join(
                [aa_three_to_one_dict.get(r, "X") for r in seq]
            )
        self.seq = seq_one_letter
        self._seq = seq
        
        # Other important attributes.
        self.aa_emb = aa_emb  # Optional, amino acid embeddings from a pre-trained model.
        self.xyz = xyz  # xyz coordinates.
        self.r = r  # Residue positional ids.
        self.xyz_tem = xyz_tem  # Optional, template xyz coordinates.
        self.e = None  # Energies.
        self.enc = None  # Encodings.
        self.enc_tem = None  # Template encodings.

        # Graph structure.
        self.edge_index = None
        self.nr_edge_index = None
        self.chain_torsion_mask = None

        # Other data attributes.
        self.attributes = attributes

    def set_encoding(self, enc, enc_tem=None):
        self.enc = enc
        self.enc_tem = enc_tem


def get_features_from_seq(seq: str):
    n_res = len(seq)
    # Feature tensor: 20 channels for aa.
    n_features = 20
    features = np.zeros((n_features, n_res))
    # Residues one hot encoding.
    for i, aa_i in enumerate(seq):
        features[aa_list.index(aa_i),i] = 1
    return features

def get_gpcrmd_aa_features(seq: List[str]):
    features = np.zeros(len(seq))
    for i, sym_i in enumerate(seq):
        features[i] = gpcrmd_alphabet_three_letters.index(sym_i)
    return features


# def apply_frames_prob_rule(enc, accept_prob_rule):
#     if accept_prob_rule is not None:
#         raise NotImplementedError()
#         # accept_p = accept_prob_rule(enc.shape[1])
#         # accept = np.random.rand(enc.shape[0]) < accept_p
#         # return enc[accept]
#     else:
#         return enc


################################################################################
# Common code for xyz and encoding datasets.                                   #
################################################################################

allowed_tbm_modes = (None, "random", "lag", "single")
allowed_tbm_types = (None, "enc", "xyz")

class CG_ProteinDatasetMixin:
    """Common methods for both the xyz and encoding datasets."""

    def _init_common(self,
                     input: Union[List[str], List[dict], str],
                     n_trajs: int = None,
                     n_frames: int = None,
                     subsample_frames: float = None,
                     frames_mode: str = "ensemble",
                     proteins: Union[list, str] = None,
                     per_protein_frames: Union[dict, str] = None,
                     n_systems: int = None,
                     re_filter: str = None,
                     res_ids_mode: str = None,
                     bead_type: str = "ca",
                     alphabet: str = None,
                     aa_embeddings_dp: str = None,
                     tbm: dict = {},
                     attributes: list = [],
                     verbose: bool = False,
                     random_seed: int = None):

        self._input = input
        self._re_filter = re_filter
        self._proteins = proteins

        # Get the input data files.
        self.input_list = get_input_list(
            input=input,
            re_filter=re_filter,
            proteins=proteins,
            data_type=self.data_type
        )
        # Get the fraction of frames to use for each protein system.
        if per_protein_frames is None:
            self.per_protein_frames = {}
        elif isinstance(per_protein_frames, dict):
            self.per_protein_frames = per_protein_frames
        elif isinstance(per_protein_frames, str):
            with open(per_protein_frames, "r") as i_fh:
                self.per_protein_frames = json.load(i_fh)
        else:
            raise TypeError(per_protein_frames.__class__)
        # Initialize all the data files.
        self._init_protein_data_files()
        
        # Initialize other attributes.
        if not res_ids_mode in (None, "resseq", "index"):
            raise KeyError(res_ids_mode)
        self.res_ids_mode = res_ids_mode
        if not bead_type in ("ca", "com", "cg"):
            raise KeyError(bead_type)
        self.bead_type = bead_type
        check_alphabet(alphabet)
        self.alphabet = alphabet
        
        self.n_trajs = n_trajs
        self.n_frames = n_frames
        self.subsample_frames = subsample_frames
        if not frames_mode in ("trajectory", "ensemble"):
            raise KeyError(frames_mode)
        self.frames_mode = frames_mode
        self.n_systems = n_systems
        
        if tbm.get("mode") not in allowed_tbm_modes:
            raise KeyError(tbm["mode"])
        if tbm.get("type") not in allowed_tbm_types:
            raise KeyError(tmb["type"])
        self.tbm = tbm
        self.attributes = attributes

        self.aa_embeddings_dp = aa_embeddings_dp
        self.use_aa_embeddings = self.aa_embeddings_dp is not None

        self.verbose = verbose
        self._set_random_obj(random_seed)

        self.protein_list = None
        self.frames = None

        
    def _print(self, msg: str, verbose: bool = None):
        print_msg(msg,
                  verbose=self.verbose if verbose is None else verbose,
                  tag="dataset")

    def _init_protein_data_files(self):
        # Get all the files.
        self.protein_data_files = []
        for input_i in self.input_list:
            # Get the protein data file object.
            prot_data_files_i = ProteinDataFiles(
                input_data=input_i,
                # data_type=data_type,
            )
            self.protein_data_files.append(prot_data_files_i)

    def _set_random_obj(self, random_seed: int = None):
        if random_seed is None:
            self.random_obj = np.random
        else:
            self.random_obj = np.random.default_rng(random_seed)
        self.random_seed = random_seed


    def load_data(self,
            # sel_trajectory
        ):
        
        # Where actually the MD data is stored.
        self.protein_list = []
        self.frames = []
        
        # ########################################################################
        # # TODO: remove.
        # if sel_trajectory is not None and len(self.protein_data_files) != 1:
        #     raise ValueError(
        #         "Can only use the 'sel_trajectory' argument in the 'load_data'"
        #         " method when the dataset contains only one protein system"
        #         f" (now you have {len(self.protein_data_files)})")
        # ########################################################################

        # Process each protein.
        protein_data_files = self._sample_systems()
        for prot_data_files_i in protein_data_files:
            self._print("# Loading data for {}".format(prot_data_files_i.name))
            # Parse and get the protein data.
            protein_obj_i = self.load_protein_data(
                prot_data_files=prot_data_files_i,
                # sel_trajectory=sel_trajectory
            )
            self.add_protein_frames(protein_obj_i)
        
        # Store the data attributes with names used in datasets use for
        # other data types.
        if not self.frames:
            raise ValueError("No frames found")

    def _sample_systems(self):
        if self.n_systems is None:
            return self.protein_data_files
        else:
            return self.random_obj.choice(
                self.protein_data_files, self.n_systems, replace=False
            )

    def _sample_traj_files(self, prot_data_files,
                           # sel_trajectory
        ):
        """Get trajectory files."""
        if True:  # sel_trajectory is None:
            if self.n_trajs is not None:  # Randomly select trajectories.
                trajectories = self.random_obj.choice(
                    prot_data_files.trajectories, self.n_trajs)
            else:  # Select all trajectories.
                trajectories = prot_data_files.trajectories
        else:  # Select a specific trajectory.
            # trajectories = [sel_trajectory]
            pass
        return trajectories


    def sample_data(self, data, n_samples, backend="numpy"):
        if backend in ("numpy", ):
            if n_samples is not None:
                n_data_elements = self._get_data_shape(data)
                ids = self.random_obj.choice(
                    n_data_elements,
                    n_samples,
                    replace=n_data_elements < n_samples
                )
                # TODO: use apply_frames_prob_rule(data, self.accept_prob_rule)
                return self._slice_data(data, ids)
            else:
                return data
        else:
            raise KeyError(backend)
    
    def _get_data_shape(self, data):
        return data.shape[0]

    def _slice_data(self, data, ids):
        return data[ids]

    def _get_n_frames(self, protein: str):
        if protein in self.per_protein_frames:
            frames_val = self.per_protein_frames[protein]
            if isinstance(frames_val, float):
                if self.n_frames is not None:
                    n_frames = int(frames_val * self.n_frames)
                else:
                    n_frames = None
            elif isinstance(frames_val, int):
                n_frames = frames_val
            else:
                raise TypeError(frames_val.__class__)
            if n_frames < 1:
                raise ValueError(n_frames)
            return n_frames
        else:
            return self.n_frames


    def add_protein_frames(self, protein_obj):
        """Add the snapshots for this protein to the dataset."""
        n_residues = len(protein_obj.seq)
        protein_count = len(self.protein_list)
        
        ###############################
        if self.use_metaencoder:
            if self.data_type == "ofo":
                n_tot_frames = protein_obj.ofo["x"].shape[0]
                if n_tot_frames % 2 != 0:
                    raise ValueError(n_tot_frames)
                n_frames = n_tot_frames // 2
                protein_obj.meta_ofo = {}
                for k in protein_obj.ofo.keys():
                    data_i = protein_obj.ofo[k][:n_frames]
                    data_j = protein_obj.ofo[k][n_frames:]
                    protein_obj.ofo[k] = data_i
                    protein_obj.meta_ofo[k] = data_j
            else:
                raise KeyError(self.data_type)
        ###############################

        if self.data_type == "xyz":
            protein_data = protein_obj.xyz
        elif self.data_type == "enc":
            protein_data = protein_obj.enc
        elif self.data_type == "ofo":
            protein_data = protein_obj.ofo["x"]
        else:
            raise KeyError(self.data_type)
        for i in range(protein_data.shape[0]):
            self.frames.append([protein_count, n_residues, i])
        self.protein_list.append(protein_obj)
    
    chain_offset = 10000
    def _get_residue_indices(self, top_traj):
        if self.res_ids_mode in ("resseq", "index"):
            if self.res_ids_mode == "resseq":
                attr = "resSeq"
            elif self.res_ids_mode == "index":
                attr = "index"
            else:
                raise KeyError(self.res_ids_mode)
            res_ids = []
            chain_c = 0
            for r in top_traj.topology.residues:
                i = getattr(r, attr)
                res_ids.append(i+r.chain.index*self.chain_offset)
            return res_ids
        else:
            return None


    def load_top_traj(self, top_fp):
        if top_fp.endswith("pdb"):
            if self.bead_type in ("ca", "com"):
                return slice_ca_traj(
                    traj=mdtraj.load(top_fp),
                    standard=is_standard_alphabet(self.alphabet),
                )
            elif self.bead_type == "cg":
                # return slice_cg_traj(mdtraj.load(top_fp))
                raise NotImplementedError()
            else:
                raise KeyError(self.bead_type)
        else:
            raise ValueError("Unknown topology file type: {}".format(top_fp))


    def refresh_dataset(self):
        if self.n_frames is not None or self.n_trajs:
            self.load_data()


    def get_aa_data(self, prot_idx):
        # Convert to tensors.
        a = self.protein_list[prot_idx].a
        a = torch.tensor(a, dtype=torch.long)
        data = {"a": a}
        if self.use_aa_embeddings:
            aa_emb = self.protein_list[prot_idx].aa_emb
            data["a_e"] = aa_emb
        return data
    
    def get_res_ids_data(self, prot_idx):
        if self.protein_list[prot_idx].r is None:
            r = torch.arange(0, len(self.protein_list[prot_idx].seq))
        else:
            r = torch.tensor(self.protein_list[prot_idx].r, dtype=torch.int)
        return {"r": r}
    
    def crop_sequences(self, data, n_residues, idx, use_crops=False):
        if use_crops:
            raise NotImplementedError()
        else:
            crop_data = None
            n_used_residues = n_residues
        return data, crop_data, n_used_residues

    def _get_aa_embeddings(self, prot_name: str):
        if not self.use_aa_embeddings:
            return None
        else:
            aa_emb = torch.load(
                os.path.join(self.aa_embeddings_dp, f"{prot_name}.pt"))[0]
            return aa_emb
    
    def _get_seq_from_traj(self,
            traj: mdtraj.Trajectory,
            three_letters: bool = False
        ):
        seq = []
        seq_three = []
        for r in traj.topology.residues:
            if r.code is not None:
                seq.append(r.code)
            else:
                seq.append("X")
            seq_three.append(r.name)
        if three_letters:
            return "".join(seq), seq_three
        else:
            return "".join(seq)


class Uniform_length_batch_sampler:
    """
    Batch sampler that will sample batches with proteins with the same
    number of residues.
    """

    def __init__(self, dataset, batch_size, drop_last=False, random_seed=None):
        self._dataset = dataset
        self.batch_size = batch_size
        self.random_seed = random_seed
        if self.random_seed is None:
            self.random_obj = np.random
        else:
            self.random_obj = np.random.default_rng(self.random_seed)

    def _get_batches(self):
        ids = []
        effective_batch_size = self.batch_size
        if effective_batch_size > len(self._dataset):
            raise ValueError
        # Get protein lengths for each element in the dataset.
        len_vals = np.array([t[1] for t in self._dataset.frames])
        # Create batches with dataset elements having the same protein length.
        for len_k in np.unique(len_vals):
            # Get all dataset elements with protein length k.
            ids_k = np.where(len_vals == len_k)[0]
            # Randomize their order.
            self.random_obj.shuffle(ids_k)
            # Get chunks of batch_size.
            for k in range(0, len(ids_k), effective_batch_size):
                ids_i = ids_k[k:k+effective_batch_size]
                ids.append(ids_i)
        return ids

    def __iter__(self):
        # Get chunks of batch_size.
        ids = self._get_batches()
        # Shuffle the chunks and return them.
        self.random_obj.shuffle(ids)
        for ids_k in ids:
            yield ids_k

    def __len__(self):
        return len(self._get_batches())
    

class Uniform_length_batch_sampler_DDP(torch.utils.data.BatchSampler,
                                       Uniform_length_batch_sampler):
    """
    Class used only for training via a Distributed Data Parallel strategy
    in PyTorch Lightning.
    """

    sam_debug = False

    def __init__(self, dataset, batch_size, drop_last=False, random_seed=None):
        """
        PyTorch Lightning will wrap the `dataset` object in an object from the
        `DistributedSampler` class. Each distributed process will instantiate
        its own copy of `DistributedSampler` which will have access to a
        different part of the dataset. 
        The `random_seed` argument here actually does not take effect. It will be
        the `random_seed` of the `dataset` that takes affect here.
        """
        super().__init__(sampler=torch.arange(0, len(dataset)),
                         batch_size=batch_size,
                         drop_last=drop_last)
        self._dataset = dataset  # Actually a `DistributedSampler` object.
    
    def _get_batches(self):
        ids = []
        effective_batch_size = self.batch_size
        if effective_batch_size > len(self._dataset):
            raise ValueError
        # Get the indices for accessing dataset elements covered by the local
        # `DistributedSampler` object.
        ids_local = np.array([i for i in self._dataset])
        # Get protein lengths for each element in the local dataset.
        len_vals = np.array([self._dataset.dataset.frames[i][1] \
                             for i in ids_local])
        # Create batches with dataset elements having the same protein length.
        for len_k in np.unique(len_vals):
            # Get all dataset elements with protein length k.
            ids_k = ids_local[len_vals == len_k]
            # Randomize their order.
            self.random_obj.shuffle(ids_k)
            # Get chunks of batch_size.
            for k in range(0, len(ids_k), effective_batch_size):
                ids_i = ids_k[k:k+effective_batch_size]
                ids.append(ids_i)
        return ids

    def __iter__(self):
        # Get chunks of batch_size.
        ids = self._get_batches()
        #---
        if self.sam_debug:
            if self._dataset.dataset.partition == "train":
                print(f"# sampler: rank={self._dataset.rank}, ids={ids}")
        #---
        # Shuffle the chunks and return them.
        self.random_obj.shuffle(ids)
        for k, ids_k in enumerate(ids):
            #---
            if self.sam_debug:
                if self._dataset.dataset.partition == "train":
                    print(f"# batch: batch_k={k}, ids_k={ids_k}, frames={[self._dataset.dataset.frames[j] for j in ids_k]}, rank={self._dataset.rank}")
            #---
            yield ids_k

    @property
    def random_obj(self):
        return self._dataset.dataset.random_obj


################################################################################
# Dataset for xyz cordinates of proteins.                                      #
################################################################################

class ProteinDataset(torch.utils.data.dataset.Dataset,
                     CG_ProteinDatasetMixin):

    data_type = "xyz"

    def __init__(self,
                 input: Union[List[str], List[dict], str],
                 n_trajs: int = None,
                 n_frames: int = None,
                 subsample_frames: float = None,
                 frames_mode: str = "ensemble",
                 proteins: Union[list, str] = None,
                 per_protein_frames: Union[dict, str] = None,
                 n_systems: int = None,
                 re_filter: str = None,
                 res_ids_mode: str = None,
                 bead_type: str = "ca",
                 alphabet: str = None,
                 xyz_sigma: float = None,
                 xyz_perturb: dict = None,
                 use_metaencoder: bool = False,
                 # aa_embeddings_dp=None,
                 tbm: dict = {},
                 load_data: bool = True,
                 verbose: bool = False,
                 random_seed: int = None):
        """
        Args:
        `input`: information on the data files used for the dataset. It can be
            one of the following data structures:
            * List of dictionaries. Each element represents data for a single
            protein and must have the following structure:
                {"name": "name_of_the_protein",
                 "topology": "/path/to/topology_file.pdb",
                 "trajectories": ["/path/to/trajectory_0.dcd",
                                  "/path/to/trajectory_1.dcd",
                                  "/path/to/trajectory_*.xtc"]}
            * List of strings. They can be filepaths or paths with glob syntax
            that point to JSON files each storing a dictionary like the one
            above.
            * String. The path of a directory with the following structure:
                input_directory
                |-- /protein_name
                |   |-- top.pdb     # a single topology file named 'top.pdb'
                |   |-- traj_0.dcd  # trajectory files in dcd or xtc format named: traj_${num}.${format}.
                |   |-- traj_1.dcd
                |   |-- ...
                |-- ...
            where each subdirectory represents a different protein system.
        `n_trajs`: number of trajectory files to randomly select from the
            entire list of files provided with `input_list`. If 'None', all
            files will be used.
        `n_frames`: numer of frames to randomly select from the input data. If
            set to 'None', all frames of the selected trajectories will be used.
        `subsample_frames`: TODO.
        `frames_mode`: determines how to randomly pick frames when `n_frames` is
            not 'None'. Choices: ('trajectory', 'ensemble'). If 'trajectory',
            `n_frames` will be picked from all selected trajectories. If
            'ensemble', `n_frames` will be picked from the total amount of 
            frames (the ensemble) of all trajectories.
        `proteins`: list of proteins names to use. If set to 'None', all proteins
            specified in the `input_list` will be used.
        `per_protein_frames`: TODO.
        `re_filter`: regular expression to select only some files from
            `input_list`.
        `res_ids_mode`: TODO.
        `xyz_sigma`: standard deviation of the Gaussian noise to add to the xyz
            coordinates. If 'None', no noise will be added.
        `tbm`: template based mode. TODO.
            `lag`: TODO. Used only if `mode` is 'lag'.
        `load_data`: load data when initializing the dataset. Data can be loaded
            at a later time via the `.load_data` method.
        `verbose`: use verbose mode.
        `random_seed`: random seed to draw samples.
        """

        self.use_enc_scaler = False
        self.xyz_sigma = xyz_sigma
        self.xyz_perturb = xyz_perturb
        self.use_metaencoder = use_metaencoder

        self.encoder = None

        self._init_common(input=input,
                          n_trajs=n_trajs,
                          n_frames=n_frames,
                          subsample_frames=subsample_frames,
                          frames_mode=frames_mode,
                          proteins=proteins,
                          per_protein_frames=per_protein_frames,
                          n_systems=n_systems,
                          re_filter=re_filter,
                          res_ids_mode=res_ids_mode,
                          bead_type=bead_type,
                          alphabet=alphabet,
                          # aa_embeddings_dp=aa_embeddings_dp,
                          tbm=tbm,
                          verbose=verbose,
                          random_seed=random_seed)

        if load_data:
            self.load_data()
    
    
    #---------------------------------------------------------------------------
    # Methods for loading the data when the dataset is initialized.            -
    #---------------------------------------------------------------------------
    
    def filter_traj(self, traj):
        if self.bead_type == "ca":
            traj = slice_ca_traj(
                traj=traj,
                standard=is_standard_alphabet(self.alphabet)
            )
        elif self.bead_type == "cg":
            # traj = slice_cg_traj(traj)
            raise NotImplementedError()
        elif self.bead_type == "com":
            # traj = slice_traj_to_com(traj, get_xyz=False)
            raise NotImplementedError()
        else:
            raise KeyError(self.bead_type)
        return traj


    def load_protein_data(
            self,
            prot_data_files: list,
            # sel_trajectory: str = None  # TODO: remove.
        ):
        """Load xyz data for a single protein."""
        
        self._print("* Loading xyz data")
        
        # Select trajectory files.
        trajectories = self._sample_traj_files(
            prot_data_files,
            # sel_trajectory
        )
        
        # Load the topology.
        top_traj = self.load_top_traj(prot_data_files.top_fp)
        seq, seq_three = self._get_seq_from_traj(
            traj=top_traj,
            three_letters=True
        )  # seq = "".join([r.code for r in top_traj.topology.residues])
        self._print("+ Sequence: {}".format(seq))

        # Read xyz data from a trajectory file.
        xyz = []

        # Actually parse each trajectory file.
        for traj_fp_i in trajectories:

            self._print("+ Parsing {}".format(traj_fp_i))
            
            # Load the trajectory.
            xyz_i = self._load_xyz_from_traj(
                traj_fp=traj_fp_i,
                top_fp=prot_data_files.top_fp
            )
            if xyz_i.shape[1] == 0:
                raise ValueError("No atoms found in the parsed trajectory")
                
            self._print("- Parsed a trajectory with shape: {}".format(
                tuple(xyz_i.shape)))
            
            # Sample frames with mode "trajectory".
            if self.frames_mode == "trajectory":
                xyz_i = self.sample_data(
                    data=xyz_i,
                    n_samples=self._get_n_frames(prot_data_files.name)
                )
            
            if xyz_i.shape[0] == 0:
                raise ValueError()
                
            # Store the frames.
            self._print("- Selected {} frames".format(repr(xyz_i.shape)))
            xyz.append(xyz_i)

        if not xyz:
            raise ValueError("No data found for {}".format(
                prot_data_files.name))
        xyz = np.concatenate(xyz, axis=0)
        
        # Sample frames with mode "ensemble".
        if self.frames_mode == "ensemble":
            xyz = self.sample_data(
                data=xyz,
                n_samples=self._get_n_frames(prot_data_files.name)
            )
        # Subsample.
        if self.subsample_frames is not None:
            n_subsample = int(xyz.shape[0]*self.subsample_frames)
            self._print("+ Subsampling {} frames".format(n_subsample))
            xyz = self.sample_data(data=xyz, n_samples=n_subsample)

        self._print("+ Will store {} frames".format(repr(xyz.shape)))

        # Get template data.
        if self.tbm.get("mode") is None:
            xyz_tem = None
        elif self.tbm["mode"] in ("single", "random"):
            ### if self.tbm["mode"] == "single":  # Get a single template trajectory.
            ###     xyz_tem = self._load_xyz_from_traj(
            ###         traj_fp=prot_data_files.template[0],
            ###         top_fp=prot_data_files.top_fp
            ###     )
            ### elif self.tbm["mode"] == "random":
            ###     # xyz_tem = xyz
            ###     raise NotImplementedError()
            ### else:
            ###     raise KeyError(self.tbm["mode"])
            raise NotImplementedError()
        else:
            raise KeyError(self.tbm["mode"])

        # Get the residue indices.
        res_ids = self._get_residue_indices(top_traj)
        
        # Initialize a CG_Protein object.
        protein_class = self._get_protein_class()
        protein_args = self._get_protein_args(top_traj)
        protein_obj = protein_class(
            name=prot_data_files.name,
            seq=seq_three,
            alphabet=self.alphabet,
            xyz=xyz,
            xyz_tem=xyz_tem,
            r=res_ids,
            **protein_args
        )
        return protein_obj
    
    def _get_protein_class(self):
        return CG_Protein
    
    def _get_protein_args(self, top_traj):
        return {}
    
    def _load_xyz_from_traj(self, traj_fp, top_fp):
        traj = mdtraj.load(traj_fp, top=top_fp)
        traj = self.filter_traj(traj)
        return traj.xyz


    def __len__(self):
        return len(self.frames)

    def len(self):
        return self.__len__()


    #---------------------------------------------------------------------------
    # Methods for getting the data when iterating over the dataset.            -
    #---------------------------------------------------------------------------

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):

        use_crops = False

        prot_idx, n_residues, frame_idx = self.frames[idx]

        data = {}
        
        # xyz data.
        data.update(self.get_xyz_data(prot_idx, frame_idx))
        
        # Amino acid data.
        data.update(self.get_aa_data(prot_idx))

        # Get residue indices.
        data.update(self.get_res_ids_data(prot_idx))
        
        # Crop sequences.
        data, crop_data, n_used_residues = self.crop_sequences(
            data=data, n_residues=n_residues, use_crops=use_crops, idx=idx)
        
        # Additional data, class-dependant (e.g.: encodings).
        data = self._update_graph_args(data, prot_idx, frame_idx, crop_data)

        # Return an object storing data for the selected conformation.
        return StaticData(**data)
    

    def get_xyz_data(self, prot_idx, frame_idx):
        """Returns a xyz frame with shape (L, 3).
        Will also convert to tensors."""

        if self.use_metaencoder:
            raise NotImplementedError()
        
        # Get the xyz frame.
        xyz = self.protein_list[prot_idx].xyz[frame_idx]
        # Get a template xyz frame.
        if self.tbm.get("mode") is None:
            xyz_tem = None
        elif self.tbm["mode"] in ("single", ):
            ### # Get a single, pre-defined template frame.
            ### if self.tbm["mode"] == "single":
            ###     xyz_tem = self.protein_list[prot_idx].xyz_tem[0]
            ### # Get a random template frame.
            ### # elif self.tbm["mode"] == "random":
            ### #     tem_frame_idx = np.random.choice(
            ### #         self.protein_list[prot_idx].enc_tem.shape[0])
            ### #     enc_tem = self.protein_list[prot_idx].enc_tem[tem_frame_idx]
            ### else:
            ###     raise KeyError(self.tbm["mode"])
            ### if self.tbm.get("perturb") is not None:
            ###     w_tem = self.tbm["perturb"]
            ###     xyz_tem = xyz_tem*w_tem + np.random.randn(*xyz_tem.shape)*(1-w_tem)
            raise NotImplementedError()
        else:
            raise KeyError(self.tbm["mode"])

        # Add Gaussian noise to original coordinates.
        xyz = self._add_noise_to_xyz(xyz)

        # Weighted sum of Gaussian noise and original coordinates.
        xyz = self._perturb_xyz(xyz)
                
        # Convert to tensors and return data.
        data = {"x": torch.tensor(xyz, dtype=torch.float)}
        if self.tbm.get("mode") is not None:
            ### data["x_t"] = torch.tensor(xyz_tem, dtype=torch.float)
            raise NotImplementedError()
        return data

    def _add_noise_to_xyz(self, xyz: np.ndarray):
        """
        Add a small amount of Gaussian noise to original coordinates.
        """
        if self.xyz_sigma is not None:
            xyz = xyz + np.random.randn(*xyz.shape)*self.xyz_sigma
           #  if self.tbm.get("mode") is not None:
           #      # xyz_t = xyz_t + np.random.randn(*xyz_t.shape)*self.xyz_sigma
           #      raise NotImplementedError()
        return xyz
    
    def _perturb_xyz(self, xyz: np.ndarray):
        if self.xyz_perturb is not None and self.xyz_perturb.get("use"):
            # Apply with a probability equal to 'prob'.
            if np.random.rand() < self.xyz_perturb["prob"]:
                # Get the weights.
                if self.xyz_perturb["sched"] == "exp":
                    w_x0 = 1-np.exp(-np.random.rand()*np.pi)  # np.e*2
                elif self.xyz_perturb["sched"] == "triangular":
                    w_x0 = np.random.triangular(0, 1, 1)
                elif self.xyz_perturb["sched"] == "power":
                    w_x0 = np.random.power(3)
                else:
                    raise KeyError(self.xyz_perturb["sched"])
                # Sum the original and random point clouds.
                noise = np.random.randn(*xyz.shape)*self.xyz_perturb["scale"]
                xyz = xyz*w_x0 + noise*(1-w_x0)
        return xyz
    
    def _update_graph_args(self, args, prot_idx, frame_idx, crop_data):
        return args


#---------------------------------------------------------------------------
# Save an encoded dataset.                                                 -
#---------------------------------------------------------------------------

def build_enc_std_scaler(
    model_cfg: dict,
    encoder: Callable,
    device: str,
    input: list,
    scaler_fp: str,
    proteins: Union[list, str] = None,
    batch_size: int = 32,
    n_frames: int = None,
    n_trajs: int = None,
    re_filter: str = None,
    res_ids_mode: str = None,
    frames_mode: str = "ensemble",
    verbose: int = 1,
    dataset_cls: type = ProteinDataset,
    data_type: str = "xyz"
    ):

    tag = "building_scaler"

    print_msg(msg="# Building a standard scaler for the encodings.",
              verbose=verbose >= 1, tag=tag)

    # Standard scaler data.
    encoding_dim = model_cfg["generative_stack"]["encoding_dim"]
    scaler_e_acc = torch.zeros(encoding_dim)
    scaler_e2_acc = torch.zeros(encoding_dim)
    scaler_n_acc = 0

    # Process xyz data for each molecule.
    input_list = get_input_list(input=input,
                                re_filter=re_filter,
                                proteins=proteins,
                                data_type=data_type)

    for prot_idx, input_i in enumerate(input_list):

        print_msg(
            msg="+ Collecting data for {} ({} of {}).".format(
                input_i["name"], prot_idx+1, len(input_list)),
            verbose=verbose >= 1, tag=tag)

        # Dataset.
        dataset_i = dataset_cls(
            input=[input_i],
            n_trajs=n_trajs,
            n_frames=n_frames,
            frames_mode=frames_mode,
            proteins=None,
            res_ids_mode=res_ids_mode,
            bead_type="ca",
            xyz_sigma=None,
            xyz_perturb=None,
            verbose=verbose >= 2
        )
        # Dataloader.
        dataloader_i = torch.utils.data.dataloader.DataLoader(
            dataset=dataset_i,
            batch_size=batch_size, shuffle=True)

        processed = 0
        for batch in dataloader_i:
            print_msg(
                msg="- Collecting data for {}/{}".format(
                    processed, len(dataloader_i.dataset)),
                verbose=verbose >= 2, tag=tag)
            # Encode the xyz coordinates.
            batch = batch.to(device)
            with torch.no_grad():
                enc_i = encoder.nn_forward(batch).cpu()
            # Store data needed to build a standard scaler.
            scaler_e_acc += enc_i.sum(axis=(0, 1))
            scaler_e2_acc += torch.square(enc_i).sum(axis=(0, 1))
            scaler_n_acc += enc_i.shape[0]*enc_i.shape[1]
            processed += batch.num_graphs

    # Save encodings transformed with a standard scaler.
    print_msg(msg="+ Save a standard scaler.", verbose=verbose >= 1, tag=tag)

    scaler_u = scaler_e_acc/scaler_n_acc
    scaler_s = torch.sqrt(scaler_e2_acc/scaler_n_acc - torch.square(scaler_u))
    scaler_u = scaler_u.reshape(1, 1, -1)
    scaler_s = scaler_s.reshape(1, 1, -1)
    enc_std_scaler = {"u": scaler_u, "s": scaler_s}
    torch.save(enc_std_scaler, scaler_fp)

    print_msg(msg="# Done.", verbose=verbose >= 1, tag=tag)

    return enc_std_scaler


def save_dataset(
        model_cfg: dict,
        encoder: Callable,
        input: list,
        out_dp: str,
        proteins: Union[list, str] = None,
        device: str = "cpu",
        force: bool = False,
        template: str = None,
        batch_size: int = 32,
        re_filter: str = None,
        res_ids_mode: str = None,
        verbose: int = 1,
        dataset_cls: type = ProteinDataset,
        data_type: str = "xyz",
        sleep: float = None,
        half: bool = False,
        subsample: float = None,
    ):

    tag = "encoding"

    if template not in (None, "topology", "first"):
        raise ValueError(f"Unknown template mode: {template}")

    print_msg(msg="# Saving dataset at: {}.".format(out_dp),
              verbose=verbose >= 1, tag=tag)

    # Setup the output directory.
    if os.path.isdir(out_dp):
        if not force:
            raise FileExistsError(
                f"Output directory '{out_dp}' already exists. Make use of the"
                " 'force' argument to overwrite.")
        shutil.rmtree(out_dp)
    os.makedirs(out_dp)

    with open(os.path.join(out_dp, "notes.txt"), "w") as o_fh:
        # o_fh.write("crop_size: {}\n".format(crop_size))
        pass

    # Process xyz data for each molecule.
    input_list = get_input_list(input=input,
                                re_filter=re_filter,
                                proteins=proteins,
                                data_type=data_type)

    # Process xyz data for each molecule.
    for prot_idx, input_i in enumerate(input_list):
        
        prot_name = input_i["name"]
        print_msg(
            msg="+ Encoding files for {} ({} of {}).".format(
                prot_name, prot_idx+1, len(input_list)),
            verbose=verbose >= 1, tag=tag)
        
        ############################################################################################
        # TODO: move below!!!  Confirm that the new/old versions are doing EXACTLY the same thing. #
        ############################################################################################

        #******************************
        # # Setup the dataset and the dataloader for the current protein system.
        # # Dataset.
        # dataset_i = dataset_cls(
        #     input=[input_i],
        #     n_trajs=None,
        #     n_frames=None,
        #     frames_mode="trajectory",
        #     res_ids_mode=res_ids_mode,
        #     bead_type="ca",  # TODO: fix it.
        #     xyz_sigma=None,
        #     xyz_perturb=None,
        #     load_data=False,  # Don't load data now.
        #     verbose=verbose >= 2
        # )
        # # Dataloader.
        # dataloader_i = torch.utils.data.dataloader.DataLoader(
        #     dataset=dataset_i,
        #     batch_size=batch_size, shuffle=False)
        #******************************
        
        # Define the output directory.
        prot_dp = os.path.join(out_dp, prot_name)
        prot_path = pathlib.Path(prot_dp)
        if prot_path.is_dir():
            raise FileExistsError(prot_dp)
        os.mkdir(prot_dp)
        top_path = prot_path / "top.pdb"

        # Copy the topology file in the output directory.
        shutil.copy(input_i["topology"], top_path)
        
        # Save an encoding file for each trajectory file.
        enc_fp_l = []
        enc_fp_m = {}
        tem_fp_l = []
        json_path = prot_path / "enc.json"
        
        trajectories = input_i["trajectories"]
        for traj_idx, sel_trajectory in enumerate(trajectories):
            
            enc_fp_i = str(prot_path / f"enc_{traj_idx}.pt")
            print_msg(
                msg=f"- Saving '{sel_trajectory}' to '{enc_fp_i}'"
                    f" ({traj_idx+1}/{len(trajectories)})",
                verbose=verbose >= 1, tag=tag)

            #++++++++++++++++++++++++++
            # Setup the dataset and the dataloader for the current protein system.
            # Dataset.
            json_data_i = {
                "name": input_i["name"],
                "topology": input_i["topology"],
                "trajectories": [sel_trajectory]
            }
            if "ofo_topology" in input_i:
                json_data_i["ofo_topology"] = input_i["ofo_topology"]
            dataset_i = dataset_cls(
                input=[json_data_i],
                n_trajs=None,
                n_frames=None,
                subsample_frames=subsample,
                frames_mode="trajectory",
                res_ids_mode=res_ids_mode,
                bead_type="ca",  # TODO: fix it.
                xyz_sigma=None,
                xyz_perturb=None,
                load_data=True,  # Load data now.
                verbose=verbose >= 2
            )
            # Dataloader.
            dataloader_i = torch.utils.data.dataloader.DataLoader(
                dataset=dataset_i,
                batch_size=batch_size,
                shuffle=False
            )
            #++++++++++++++++++++++++++

            # Load data for a single trajectory file.
            #**************************
            # dataset_i.load_data(sel_trajectory=sel_trajectory)
            #**************************

            #++++++++++++++++++++++++++
            dataset_i.load_data()
            #++++++++++++++++++++++++++

            # Encode.
            enc = []
            saved = 0
            for batch in dataloader_i:
                # Encode the xyz coordinates.
                batch = batch.to(device)
                with torch.no_grad():
                    enc_i = encoder.nn_forward(batch)
                    if sleep is not None:
                        time.sleep(sleep/2)
                    enc_i = enc_i.cpu()
                # Store the encodings from the current batch.
                enc.append(enc_i)
                saved += batch.num_graphs
                print_msg(msg="- Encoded {}/{}".format(
                    saved, len(dataloader_i.dataset)),
                    verbose=verbose >= 2, tag=tag)
                if sleep is not None:
                    time.sleep(sleep/2)
            enc = torch.cat(enc, axis=0)
            enc_fp_l.append(enc_fp_i)
            enc_fp_m[enc_fp_i] = sel_trajectory

            if half:
                enc = enc.to(dtype=torch.float16)
            torch.save(enc, enc_fp_i)
            if template == "first" and traj_idx == 0:
                tem_fp_i = str(prot_path / f"template_{traj_idx}.pt")
                torch.save(enc[0][None,...], tem_fp_i)
                tem_fp_l.append(tem_fp_i)
        
        # Encode the 3d structure in the topology file as a template.
        if template == "topology":
            # Dataset.
            json_tem = {
                "name": input_i["name"],
                "topology": input_i["topology"],
                "trajectories": [input_i["topology"]]
            }
            if "ofo_topology" in input_i:
                json_tem["ofo_topology"] = input_i["ofo_topology"]
            if data_type == "ofo":
                if dataset_cls.__name__ != "PreProcessedAllAtomProteinDataset":
                    raise TypeError()
                dataset_cls_t = dataset_cls.__bases__[0]
            else:
                dataset_cls_t = dataset_cls
            dataset_t = dataset_cls_t(
                input=[json_tem],
                n_trajs=None,
                n_frames=None,
                frames_mode="trajectory",
                res_ids_mode=res_ids_mode,
                bead_type="ca",  # TODO: fix it.
                xyz_sigma=None,
                xyz_perturb=None,
                load_data=True,
                verbose=verbose >= 2
            )
            # Dataloader.
            dataloader_t = torch.utils.data.dataloader.DataLoader(
                dataset=dataset_t,
                batch_size=batch_size,
                shuffle=False
            )
            # Encode.
            for batch in dataloader_t:
                # Encode the xyz coordinates.
                batch = batch.to(device)
                with torch.no_grad():
                    enc_tem = encoder.nn_forward(batch)
                    if sleep is not None:
                        time.sleep(sleep/2)
                    enc_tem = enc_tem.cpu()
                # Store the encodings from the current batch.
                print_msg(
                    msg="- Encoded template".format(verbose=verbose >= 2, tag=tag)
                )
            tem_fp_i = str(prot_path / "template_0.pt")
            if half:
                enc_tem = enc_tem.to(dtype=torch.float16)
            torch.save(enc_tem[0][None,...], tem_fp_i)
            tem_fp_l.append(tem_fp_i)
        
        # Save a JSON file with all the information for this protein system.
        prot_data = {}
        prot_data["topology"] = str(top_path)
        prot_data["trajectories"] = enc_fp_l
        prot_data["original_trajectories"] = enc_fp_m
        prot_data["seq"] = dataset_i.protein_list[0].seq
        if template is not None:
            prot_data["template"] = tem_fp_l
        with open(json_path, "w") as o_fh:
            o_fh.write(json.dumps(prot_data, indent=2))
    
    print_msg(msg="# Done.", verbose=verbose >= 1, tag=tag)


################################################################################
# Dataset for protein structure encodings, stored in files.                    #
################################################################################

class EncodedProteinDataset(torch.utils.data.dataset.Dataset,
                            CG_ProteinDatasetMixin):

    data_type = "enc"

    def __init__(self,
                 input: Union[List[str], List[dict], str],
                 enc_scaler_fp: str = None,
                 n_trajs: int = None,
                 n_frames: int = None,
                 subsample_frames: float = None,
                 frames_mode: str = "ensemble",
                 proteins: Union[list, str] = None,
                 per_protein_frames: Union[dict, str] = None,
                 n_systems: int = None,
                 re_filter: str = None,
                 res_ids_mode: str = None,
                 bead_type: str = "ca",
                 alphabet: str = None,
                 aa_embeddings_dp: str = None,
                 tbm: dict = {},
                 attributes: list = [],
                 load_data: bool = True,
                 verbose: bool = True,
                 random_seed: int = None
                ):

        self.enc_scaler_fp = enc_scaler_fp
        if self.enc_scaler_fp is not None:
            self.enc_scaler = torch.load(self.enc_scaler_fp)
        else:
            self.enc_scaler = None
        self.use_enc_scaler = self.enc_scaler is not None
        self.use_metaencoder = False

        self._init_common(input=input,
                          n_trajs=n_trajs,
                          n_frames=n_frames,
                          subsample_frames=subsample_frames,
                          frames_mode=frames_mode,
                          proteins=proteins,
                          per_protein_frames=per_protein_frames,
                          n_systems=n_systems,
                          re_filter=re_filter,
                          res_ids_mode=res_ids_mode,
                          bead_type=bead_type,
                          alphabet=alphabet,
                          aa_embeddings_dp=aa_embeddings_dp,
                          tbm=tbm,
                          attributes=attributes,
                          verbose=verbose,
                          random_seed=random_seed)

        if load_data:
            self.load_data()
    
    
    #---------------------------------------------------------------------------
    # Methods for loading the data when the dataset is initialized.            -
    #---------------------------------------------------------------------------

    def load_protein_data(self,
            prot_data_files,
            # sel_trajectory=None
        ):
        """Load encoding data for a single protein."""
        
        self._print("* Loading enc data")
        
        # Select trajectory files.
        trajectories = self._sample_traj_files(
            prot_data_files,
            # sel_trajectory
        )

        # Load the topology.
        seq = prot_data_files.seq
        self._print("+ Sequence: {}".format(seq))
        top_traj = self.load_top_traj(prot_data_files.top_fp)
        if seq is None:
            seq = self._get_seq_from_traj(traj=top_traj, three_letters=False)

        # Read enc data from trajectory files.
        enc = []

        for traj_fp_i in trajectories:

            self._print("+ Parsing {}".format(traj_fp_i))
            
            # Load the trajectory.
            enc_i = self._load_enc_file(traj_fp_i)
            # Standardize the encodings.
            if self.use_enc_scaler:
                enc_i = self._scale_encoding(enc_i)  # (enc_i - self.enc_scaler["u"])/self.enc_scaler["s"]
                
            self._print("- Parsed a trajectory with shape: {}".format(
                tuple(enc_i.shape)))
            
            # Sample frames with mode "trajectory".
            if self.frames_mode == "trajectory":
                enc_i = self.sample_data(
                    data=enc_i,
                    n_samples=self._get_n_frames(prot_data_files.name)
                )
            
            if enc_i.shape[0] == 0:
                raise ValueError()
                
            # Store the frames.
            self._print("- Selected {} frames".format(repr(enc_i.shape)))
            enc.append(enc_i)

        if not enc:
            raise ValueError("No data found for {}".format(
                prot_data_files.name))
        enc = torch.cat(enc, axis=0)

        # Sample frames with mode "ensemble".
        if self.frames_mode == "ensemble":
            enc = self.sample_data(
                data=enc,
                n_samples=self._get_n_frames(prot_data_files.name)
            )
        # Subsample.
        if self.subsample_frames is not None:
            n_subsample = int(enc.shape[0]*self.subsample_frames)
            self._print("+ Subsampling {} frames".format(n_subsample))
            enc = self.sample_data(data=enc, n_samples=n_subsample)

        self._print("+ Will store {} frames".format(repr(enc.shape)))

        # Get a template encoding.
        if self.tbm.get("type", "enc") == "enc":
            if self.tbm.get("mode") is None:
                enc_tem = None
            elif self.tbm["mode"] in ("single", "random"):
                ### # Get a single template encoding.
                ### if self.tbm["mode"] == "single":
                ###     enc_tem = self._load_enc_file(prot_data_files.template[0])
                ###     if self.use_enc_scaler:  # Standardize the encodings.
                ###         enc_tem = self._scale_encoding(enc_tem)
                ### elif self.tbm["mode"] == "random":
                ###     # enc_tem = self._load_enc_file(
                ###     #     np.random.choice(prot_data_files.template)
                ###     # )
                ###     enc_tem = enc
                ### else:
                ###     raise KeyError(self.tbm["mode"])
                raise NotImplementedError()
            else:
                raise KeyError(self.tbm["mode"])

        # Get a template 3D structure.
        elif self.tbm["type"] == "xyz":
            
            # Get a single template encoding.
            if self.tbm["mode"] == "single":
                # Select a single template, in PDB format.
                tem_fp = prot_data_files.template["xyz"]["single"]
                if not tem_fp.endswith(".pdb"):
                    raise ValueError(f"Not a PDB: {tem_fp}")
                tem_traj = mdtraj.load(tem_fp)
            elif self.tbm["mode"] == "random":
                # Select a random template file from (maybe) many.
                tem_fp_l = prot_data_files.template["xyz"]["multiple"]
                tem_fp = np.random.choice(tem_fp_l)
                tem_traj = mdtraj.load(tem_fp, top=prot_data_files.top_fp)
            else:
                raise KeyError(self.tbm["mode"])
            
            enc_tem = get_atom14_sam_data(tem_traj)
        
        else:
            raise TypeError(self.tbm["type"])

        # Get the residue indices.
        res_ids = self._get_residue_indices(top_traj)

        # Initialize a CG_Protein object.
        protein_obj = CG_Protein(
            name=prot_data_files.name,
            seq=seq,
            xyz=None,
            r=res_ids,
            aa_emb=self._get_aa_embeddings(prot_data_files.name),
            attributes=prot_data_files.attributes
        )

        protein_obj.set_encoding(enc, enc_tem=enc_tem)
        return protein_obj


    def _load_enc_file(self, enc_fp):
        return torch.load(enc_fp)
    
    def _scale_encoding(self, enc):
        return (enc - self.enc_scaler["u"])/self.enc_scaler["s"]


    def __len__(self):
        return len(self.frames)

    def len(self):
        return self.__len__()
    

    #---------------------------------------------------------------------------
    # Methods for getting the data when iterating over the dataset.            -
    #---------------------------------------------------------------------------
    
    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):

        prot_idx, n_residues, frame_idx = self.frames[idx]

        data = {}
        
        # Encoding data.
        data.update(self.get_enc_data(prot_idx, n_residues, frame_idx))
        
        # Amino acid data.
        data.update(self.get_aa_data(prot_idx))

        # Residue indices.
        data.update(self.get_res_ids_data(prot_idx))
        
        # Additional data, class-dependant (e.g.: encodings).
        data = self._update_graph_args(data, prot_idx, frame_idx)

        # Return an object storing data for the selected conformation.
        return StaticDataEnc(**data)
    

    def get_enc_data(self, prot_idx, n_residues, frame_idx):
        """Returns a enc frame with shape (L, E_e).
        Will also convert to tensors."""
        
        # Get the enc frame.
        enc = self.protein_list[prot_idx].enc[frame_idx]
        # Get a template enc frame.
        if self.tbm.get("type", "enc") == "enc":
            ### if self.tbm.get("mode") is None:
            ###     enc_tem = None
            ### elif self.tbm["mode"] in ("single", "random"):
            ###     # Get a single, pre-defined template frame.
            ###     if self.tbm["mode"] == "single":
            ###         enc_tem = self.protein_list[prot_idx].enc_tem[0]
            ###     # elif self.tbm["mode"] == "single_walk":
            ###     #     enc_tem = self.protein_list[prot_idx].enc_tem[frame_idx]
            ###     # Get a random template frame.
            ###     elif self.tbm["mode"] == "random":
            ###         tem_frame_idx = np.random.choice(
            ###             self.protein_list[prot_idx].enc_tem.shape[0])
            ###         enc_tem = self.protein_list[prot_idx].enc_tem[tem_frame_idx]
            ###     else:
            ###         raise KeyError(self.tbm["mode"])
            ###     if self.tbm.get("perturb") is not None:
            ###         w_tem = self.tbm["perturb"]
            ###         enc_tem = enc_tem*w_tem + torch.randn_like(enc_tem)*(1-w_tem)
            ### else:
            ###     raise KeyError(self.tbm["mode"])
            raise NotImplementedError()

        elif self.tbm["type"] == "xyz":
            if self.tbm.get("mode") is None:
                enc_tem = None
            elif self.tbm["mode"] in ("single", "random"):
                if self.tbm["mode"] == "single":
                    # Get a single, pre-defined template frame.
                    tem_frame = 0
                elif self.tbm["mode"] == "random":
                    # Get a random snapshots from a template ensemble.
                    tem_frame = np.random.choice(
                        self.protein_list[prot_idx].enc_tem["xyz"].shape[0]
                    )
                else:
                    raise KeyError(self.tbm["mode"])
                enc_tem = {
                    "xyz": self.protein_list[prot_idx].enc_tem["xyz"][tem_frame],
                    "top": self.protein_list[prot_idx].enc_tem["top"][tem_frame]
                }
                if self.tbm.get("perturb") is not None:
                    w_tem = self.tbm["perturb"]
                    enc_tem["xyz"] = enc_tem["xyz"]*w_tem + \
                        np.random.randn(*enc_tem["xyz"].shape)*(1-w_tem)
            else:
                raise KeyError(self.tbm["mode"])

        else:
            raise KeyError(self.tbm["type"])
        # Return data.
        data = {"z": enc}
        if self.tbm.get("mode") is not None:
            if self.tbm.get("type", "enc") == "enc":
                ### data["z_t"] = enc_tem
                raise NotImplementedError()
            elif self.tbm["type"] == "xyz":
                data["z_t"] = torch.tensor(enc_tem["xyz"], dtype=torch.float)
                data["z_top"] = torch.tensor(enc_tem["top"], dtype=torch.long)
            else:
                raise KeyError(self.tbm["type"])
        return data

    
    def _update_graph_args(self, args, prot_idx, frame_idx):
        if self.attributes:
            prot = self.protein_list[prot_idx]
            if "temperature" in self.attributes:
                args.update({
                    "temperature": torch.tensor(prot.attributes["temperature"])  # prot.attributes["temperature"]
                })
        return args


#
# TODO: remove and replace with minimal dataset class (see below).
#

class EvalEncodedProteinDataset(EncodedProteinDataset):
    """
    Dataset for encodings of a single protein used at inference time for
    generating conformations for that protein.
    """
    def __init__(self,
                 name: str,
                 seq: str,
                 enc_dim: int,
                 n_frames: int,
                 res_ids_mode: str = None,
                 aa_emb: torch.Tensor = None,
                 tbm: dict = {},
                 attributes: dict = {},
                 tem_enc: torch.Tensor = None,
                 verbose: bool = True
                 ):
        # Initialize.
        torch.utils.data.dataset.Dataset.__init__(self)
        self.n_frames = n_frames
        self.subsample_frames = None
        self.res_ids_mode = res_ids_mode
        if self.res_ids_mode is not None:
            raise NotImplementedError()
        self.verbose = verbose
        self.use_aa_embeddings = aa_emb is not None
        self.tbm = tbm
        if self.tbm.get("mode") not in (None, "single"):
            raise KeyError(self.tbm["mode"])
        self.alphabet = None
        self.attributes = attributes

        # Add a protein object.
        idx = 0
        n_residues = len(seq)
        enc_i = torch.zeros(self.n_frames, n_residues, enc_dim)
        xyz_i = None
        self._print("- Loading enc=%s for %s" % (enc_i.shape, name))
        self._print("- Using attributes: %s" % (repr(attributes)))
        protein = CG_Protein(
            name=name, seq=seq, xyz=xyz_i, aa_emb=aa_emb, attributes=attributes
        )
        protein.set_encoding(enc_i, enc_tem=tem_enc)

        # Add the snapshots for this IDP to the dataset.
        self.frames = []
        self.protein_list = []
        for i in range(protein.enc.shape[0]):
            self.frames.append([idx, n_residues, i])
        self.protein_list.append(protein)


################################################################################
# Dataset for proteins structures encoded at training time.                    #
################################################################################

class LiveEncodedDatasetMixin(EncodedProteinDataset):
    """
    TODO.
    """
    
    data_type = "xyz"

    def __init__(self,
                 input: Union[List[str], List[dict], str],
                 encoder: Callable,
                 enc_scaler_fp: str = None,
                 enc_batch_size: int = 8,
                 enc_device: str = "cuda",
                 n_trajs: int = None,
                 n_frames: int = None,
                 subsample_frames: float = None,
                 frames_mode: str = "ensemble",
                 proteins: Union[list, str] = None,
                 per_protein_frames: Union[dict, str] = None,
                 n_systems: int = None,
                 re_filter: str = None,
                 res_ids_mode: str = None,
                 bead_type: str = "ca",
                 alphabet: str = None,
                 aa_embeddings_dp=None,
                 tbm: dict = {},
                 load_data: bool = True,
                 verbose: bool = False,
                 random_seed: int = None):

        self.encoder = encoder
        self.enc_scaler_fp = enc_scaler_fp
        if self.enc_scaler_fp is not None:
            self.enc_scaler = torch.load(self.enc_scaler_fp)
        else:
            self.enc_scaler = None
        self.use_enc_scaler = self.enc_scaler is not None
        self.enc_batch_size = enc_batch_size
        self.enc_device = enc_device

        self._init_common(input=input,
                          n_trajs=n_trajs,
                          n_frames=n_frames,
                          subsample_frames=subsample_frames,
                          frames_mode=frames_mode,
                          proteins=proteins,
                          per_protein_frames=per_protein_frames,
                          n_systems=n_systems,
                          re_filter=re_filter,
                          res_ids_mode=res_ids_mode,
                          bead_type=bead_type,
                          alphabet=alphabet,
                          # aa_embeddings_dp=aa_embeddings_dp,
                          tbm=tbm,
                          verbose=verbose,
                          random_seed=random_seed)

        if load_data:
            self.load_data()


    def load_data(self,
            # sel_trajectory=None
        ):

        # Load xyz data first.
        super().load_data(
            # sel_trajectory=sel_trajectory
        )
        
        # Process each protein system to encode its xyz conformations.
        self._print(
            f"- Beginning live encoding for {len(self.protein_list)} proteins"
            " systems",
            verbose=True
        )
        if self.enc_device == "cuda":
            self._print("- Clearing cuda cache before live encoding")
            torch.cuda.empty_cache()
        self.encoder.to(self.enc_device)

        t0 = time.time()

        for protein_obj in self.protein_list:
            self._print(f"- Encoding {protein_obj.name}")
            # Minimal dataset for a single protein system.
            min_live_dataset_cls = self._get_minimal_live_dataset_cls()
            min_live_dataset_args = self._get_minimal_live_dataset_args(protein_obj)
            min_dataset_i = min_live_dataset_cls(
                 name=protein_obj.name,
                 xyz=protein_obj.xyz,
                 seq=protein_obj.seq,
                 xyz_tem=protein_obj.xyz_tem,
                 bead_type=self.bead_type,
                 res_ids_mode=self.res_ids_mode,
                 verbose=self.verbose,
                 **min_live_dataset_args
            )
            # Dataloader.
            dataloader_i = torch.utils.data.dataloader.DataLoader(
                dataset=min_dataset_i,
                batch_size=self.enc_batch_size, shuffle=False)
            # Encode the xyz data.
            enc_i = []
            t0 = time.time()
            for batch_j in dataloader_i:
                batch_j = batch_j.to(self.enc_device)
                with torch.no_grad():
                    enc_j = self.encoder.nn_forward(batch_j).cpu()
                if self.use_enc_scaler:
                    enc_j = self._scale_encoding(enc_j)
                enc_i.append(enc_j)
            enc_i = torch.cat(enc_i, axis=0)
            # Optionally, encode the template xyz data.
            if self.tbm.get("mode") is not None:
                self._print(f"- Encoding templates {protein_obj.name}")
                min_dataset_i.load_live_data(use_tbm=True)
                enc_tem_i = []
                for batch_j in dataloader_i:
                    batch_j = batch_j.to(self.enc_device)
                    with torch.no_grad():
                        enc_tem_j = self.encoder.nn_forward(batch_j).cpu()
                    if self.use_enc_scaler:
                        enc_tem_j = self._scale_encoding(eenc_tem_jnc_j)
                    enc_tem_i.append(enc_tem_j)
                enc_tem_i = torch.cat(enc_tem_i, axis=0)
            else:
                enc_tem_i = None
            # Store the encodings and free memory from xyz data.
            protein_obj.set_encoding(enc=enc_i, enc_tem=enc_tem_i)
            del protein_obj.xyz
        
        self._print("- Moving encoder back to CPU.")
        self.encoder.to("cpu")
        if self.enc_device == "cuda":
            self._print("- Clearing cuda cache after live encoding")
            torch.cuda.empty_cache()
        
        tot_time = time.time() - t0
        self._print(f"- Live encoding completed in {tot_time:.2f} s",
                    verbose=True)

    def get(self, idx):
        return EncodedProteinDataset.get(
            self, idx=idx
        )
    
    def _update_graph_args(self, *args, **kwargs):
        return EncodedProteinDataset._update_graph_args(
            self, *args, **kwargs
        )

class LiveEncodedProteinDataset(LiveEncodedDatasetMixin, ProteinDataset):

    def _get_minimal_live_dataset_cls(self):
        return MinimalLiveDataset
    
    def _get_minimal_live_dataset_args(self, protein):
        return {}

    def load_protein_data(self,
            prot_data_files,
            # sel_trajectory=None
        ):
        return ProteinDataset.load_protein_data(
            self,
            prot_data_files=prot_data_files,
            # sel_trajectory=sel_trajectory
        )


class MinimalDataset(ProteinDataset):
    """
    Class for a minimal dataset storing data for only one protein system.
    Used for:
        - evaluation
        - etc...
    """
    
    def __init__(
            self,
            name: str,
            seq: str,
            xyz: np.ndarray = None,
            n_frames: int = None,
            tbm: dict = {},
            xyz_tem: np.ndarray = None,
            bead_type: str = "ca",
            alphabet: str = None,
            res_ids_mode: str = None,
            random_seed: int = None,
            verbose: bool = True
        ):
        self.bead_type = bead_type
        self.res_ids_mode = res_ids_mode
        self.xyz_sigma = None
        self.xyz_perturb = None
        self.use_aa_embeddings = False
        self.alphabet = alphabet
        self.verbose = verbose
        self._set_random_obj(random_seed)
        self.tbm = tbm
        self.use_metaencoder = False

        if xyz is None:
            if n_frames is None:
                raise ValueError()
            xyz = np.zeros((n_frames, len(seq), 3))
        else:
            if n_frames is not None:
                raise ValueError()

        prot_obj = CG_Protein(
            name=name,
            seq=seq,
            xyz=xyz,
            xyz_tem=xyz_tem
        )
        self.protein_list = [prot_obj]

        self.load_data()


    def load_data(self):
        self.frames = []
        n_residues = self.protein_list[0].xyz.shape[1]
        for i in range(self.protein_list[0].xyz.shape[0]):
            self.frames.append([0, n_residues, i])
        if not self.frames:
            raise ValueError("No frames found")


class MinimalLiveDataset(MinimalDataset):
    """
    Minimal dataset storing data for one protein system.
    This subclass is used only for live encoding.
    """
    
    # def __init__(self, *args, **kwargs):
    #     MinimalLiveDataset.__init__(self, *args, **kwargs)
    
    def load_data(self):
        self.load_live_data(use_tbm=False)

    def load_live_data(self, use_tbm: bool = False):
        """Add the snapshots for this protein to the dataset."""
        self.frames = []
        xyz = self.protein_list[0].xyz if not use_tbm \
                                       else self.protein_list[0].xyz_tem
        n_residues = xyz.shape[1]
        for i in range(xyz.shape[0]):
            self.frames.append([0, n_residues, i])
        if not self.frames:
            raise ValueError("No frames found")