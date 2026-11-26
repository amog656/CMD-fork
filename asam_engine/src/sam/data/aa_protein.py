"""
TODO:
    - preprocess OpenFold features.
    - implement save encoded datasets and live encoding.
"""

import time
import tempfile
from collections import namedtuple
from typing import Callable, Union, List, Dict

import numpy as np
import mdtraj
import torch

from sam.data.cg_protein import (
    StaticDataMixin, CG_Protein, ProteinDataset, LiveEncodedDatasetMixin
)
from sam.data.cg_protein import save_dataset as save_dataset_cg
from sam.data.sequences import (# aa_list,
                                # gpcrmd_alphabet_three_letters,
                                # aa_three_to_one_dict,
                                # check_alphabet,
                                is_standard_alphabet)
from sam.data.aa_topology import (
    slice_aa_traj,
    get_po_from_pdb_string,
    get_po_from_mdtraj,
    get_frames_from_po,
    per_conf_attrs,
    per_sys_attrs
)


staticdata_aa_xyz_keys = [
    # Main attributes.
    "x",
    "a", "a_e", "r",  # "x_t",
    # From OpenFold batches.
    "backbone_rigid_tensor",
    "backbone_rigid_mask",
    "rigidgroups_gt_frames",
    "rigidgroups_alt_gt_frames",
    "rigidgroups_gt_exists",
    "atom14_gt_positions",
    "atom14_alt_gt_positions",
    "atom14_atom_is_ambiguous",
    "atom14_gt_exists",
    "atom14_alt_gt_exists",
    "atom14_atom_exists",
    "seq_mask",
    "chi_mask",
    "chi_angles_sin_cos",
    # Optional.
    "aa_x"
]
_StaticDataAA = namedtuple(
    "StaticData",
    staticdata_aa_xyz_keys,
    defaults=[0 for i in range(len(staticdata_aa_xyz_keys))]
)

class StaticDataAA(_StaticDataAA, StaticDataMixin):
    __slots__ = ()

    @property
    def sel_val(self):
        return self.backbone_rigid_tensor


meta_prefix = "META"
meta_staticdata_aa_xyz_keys = staticdata_aa_xyz_keys + [
    f"{meta_prefix}_{k}" for k in per_conf_attrs
]
_MetaStaticDataAA = namedtuple(
    "MetaStaticData",
    meta_staticdata_aa_xyz_keys,
    defaults=[0 for i in range(len(meta_staticdata_aa_xyz_keys))]
)

class MetaStaticDataAA(_MetaStaticDataAA, StaticDataMixin):
    __slots__ = ()

    @property
    def sel_val(self):
        return self.backbone_rigid_tensor
    
    def get_metaencoder_batches(self):
        # Batch i.
        batch_i = StaticDataAA(
            **{k: getattr(self, k) for k in staticdata_aa_xyz_keys}
        )
        # Batch j.
        batch_j_data = {}
        for k in staticdata_aa_xyz_keys:
            k_j = k if not k in per_conf_attrs else f"{meta_prefix}_{k}"
            batch_j_data[k] = getattr(self, k_j)
        batch_j = StaticDataAA(**batch_j_data)
        return batch_i, batch_j
    

class AA_Protein(CG_Protein):
    """
    Class to store data for a protein molecule used in a training epoch.
    See the CG_protein class for more information.
    This class stores an additional topology attribute, that can be used to save
    PDB format content for the xyz conformations. This content will be used
    to initialize protein data with OpenFold.
    """

    def __init__(
        self,
        *args,
        topology: mdtraj.Topology,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.topology = topology
        self.ca_ids = [a.index for a in self.topology.atoms if a.name == "CA"]

class OFO_AA_Protein(AA_Protein):
    """
    Stores also pre-processed OpenFold data.
    """

    def __init__(
        self,
        *args,
        ofo: dict,
        ofo_topology: torch.Tensor,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        # OpenFold features (assigned here only when using a pre-processed OFO
        # dataset).
        self.ofo = ofo
        self.ofo_topology = ofo_topology


class AllAtomProteinDataset(ProteinDataset):

    data_type = "xyz"

    def __init__(self, *args, use_raw_xyz: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_raw_xyz = use_raw_xyz

    #---------------------------------------------------------------------------
    # Methods for loading the data when the dataset is initialized.            -
    #---------------------------------------------------------------------------
    
    def filter_traj(self, traj):
        traj = slice_aa_traj(
            traj=traj,
            standard=is_standard_alphabet(self.alphabet)
        )
        return traj

    
    def load_top_traj(self, top_fp):
        if top_fp.endswith("pdb"):
            return slice_aa_traj(
                traj=mdtraj.load(top_fp),
                standard=is_standard_alphabet(self.alphabet),
            )
        else:
            raise ValueError("Unknown topology file type: {}".format(top_fp))


    def _get_protein_class(self):
        return AA_Protein
    
    def _get_protein_args(self, top_traj):
        return {"topology": top_traj.topology}


    def load_data(self, *args, **kwargs):  ###
        if hasattr(self, "time_c"):  ###
            print(f"[TIME] It took m={self.time_m} a={self.time_a} for {self.time_c}")  ###
        self.time_a = 0  ###
        self.time_m = 0  ###
        self.time_c = 0  ###
        super().load_data(*args, **kwargs)  ###

    #---------------------------------------------------------------------------
    # Methods for getting the data when iterating over the dataset.            -
    #---------------------------------------------------------------------------

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
        
        # Additional data, class-dependant (e.g.: encodings).
        data = self._update_graph_args(data, prot_idx, frame_idx)

        # Return an object storing data for the selected conformation.
        if not self.use_metaencoder:
            return StaticDataAA(**data)
        else:
            return MetaStaticDataAA(**data)
    

    def get_xyz_data(self, prot_idx, frame_idx):
        """
        Returns a frames storing OpenFold features. Will also convert to
        tensors.
        """

        if self.use_metaencoder:
            raise NotImplementedError()
        
        ### Get raw data.

        # Get the xyz frame.
        xyz = self.protein_list[prot_idx].xyz[frame_idx]

        # Get a template xyz frame.
        if self.tbm.get("mode") is not None:
            raise KeyError(self.tbm["mode"])


        ### Perturb.

        # Add Gaussian noise to original coordinates.
        xyz = self._add_noise_to_xyz(xyz)

        # Weighted sum of Gaussian noise and original coordinates.
        xyz = self._perturb_xyz(xyz)


        ### Extract OpenFold features.
        t0 = time.time()  ###

        positions = xyz*10.0
        xyz_ca = xyz[self.protein_list[prot_idx].ca_ids]
        self.time_m += time.time()-t0  ###

        t0 = time.time()  ###
        po_of = get_po_from_mdtraj(
            topology=self.protein_list[prot_idx].topology,
            xyz=positions,
            unsqueeze=True
        )
        fr_of = get_frames_from_po(po_of)

        self.time_a += time.time()-t0  ###
        self.time_c += 1  ###

        ### Convert to tensors and return data.
        data = {
            "x": torch.tensor(xyz_ca, dtype=torch.float),
            "backbone_rigid_tensor": fr_of["backbone_rigid_tensor"][0],
            "backbone_rigid_mask": fr_of["backbone_rigid_mask"][0],
            "rigidgroups_gt_frames": fr_of["rigidgroups_gt_frames"][0],
            "rigidgroups_alt_gt_frames": fr_of["rigidgroups_alt_gt_frames"][0],
            "rigidgroups_gt_exists": fr_of["rigidgroups_gt_exists"][0],
            "atom14_gt_positions": fr_of["atom14_gt_positions"][0],
            "atom14_alt_gt_positions": fr_of["atom14_alt_gt_positions"][0],
            "atom14_atom_is_ambiguous": fr_of["atom14_atom_is_ambiguous"][0],
            "atom14_gt_exists": fr_of["atom14_gt_exists"][0],
            "atom14_alt_gt_exists": fr_of["atom14_alt_gt_exists"][0],
            "atom14_atom_exists": fr_of["atom14_atom_exists"][0],
            "seq_mask": fr_of["seq_mask"][0],
            "chi_mask": fr_of["chi_mask"][0],
            "chi_angles_sin_cos": fr_of["chi_angles_sin_cos"][0]
        }
        data["atom14_gt_positions"] = data["atom14_gt_positions"].to(dtype=torch.float)
        data["atom14_alt_gt_positions"] = data["atom14_alt_gt_positions"].to(dtype=torch.float)
        data["chi_mask"] = data["chi_mask"].to(dtype=torch.float)
        data["chi_angles_sin_cos"] = data["chi_angles_sin_cos"].to(dtype=torch.float)
        if self.use_raw_xyz:
            data["aa_x"] = xyz
        return data


    def _update_graph_args(self, args, prot_idx, frame_idx):
        return args


def save_dataset(*args, dataset_cls=AllAtomProteinDataset, **kwargs):
    return save_dataset_cg(*args, dataset_cls=dataset_cls, **kwargs)


class PreProcessedAllAtomProteinDataset(AllAtomProteinDataset):
    """
    Stores OpenFold features. Used only for training. OpenFold features occupy
    a lot of memory, so make sure to use small `n_frames` values when loading
    data for a lot of protein systems.
    Only used for training an autoencoder, not for static or live encoding.
    """

    data_type = "ofo"

    def _get_data_shape(self, data):
        return data["x"].shape[0]

    def _slice_data(self, data, ids):
        return {k: data[k][ids] for k in data}
    
    def _get_protein_class(self):
        return OFO_AA_Protein

    def load_protein_data(
            self,
            prot_data_files: list,
            # sel_trajectory: str = None  # TODO: remove.
        ):
        """Load xyz data for a single protein."""

        self._print("* Loading ofo data")
        
        # Select trajectory files.
        trajectories = self._sample_traj_files(
            prot_data_files,
            # sel_trajectory
        )
        
        # Load the topology.
        top_traj = self.load_top_traj(prot_data_files.top_fp)
        # Load the OpenFold topology data.
        if prot_data_files.ofo_top_fp is None:
            raise ValueError(f"No OpenFold topology for {prot_data_files.name}")
        ofo_top = torch.load(prot_data_files.ofo_top_fp)
        # Get the sequence.
        seq, seq_three = self._get_seq_from_traj(
            traj=top_traj,
            three_letters=True
        )  # seq = "".join([r.code for r in top_traj.topology.residues])
        self._print("+ Sequence: {}".format(seq))

        # Read xyz data from a trajectory file.
        ofo = []

        # Actually parse each trajectory file.
        for traj_fp_i in trajectories:

            self._print("+ Parsing {}".format(traj_fp_i))

            # Load the trajectory.
            ofo_i = torch.load(traj_fp_i)
            ofo_i = self._process_ofo(ofo_i)
            if ofo_i["x"].shape[0] == 0 or ofo_i["x"].shape[1] == 0:
                raise ValueError("No atoms found in the parsed trajectory")
                
            self._print("- Parsed a trajectory with Ca xyz shape: {}".format(
                tuple(ofo_i["x"].shape))
            )
            
            # Sample frames with mode "trajectory".
            if self.frames_mode == "trajectory":
                ofo_i = self.sample_data(
                    data=ofo_i,
                    n_samples=self._get_n_frames(prot_data_files.name)
                )
            xyz_i = ofo_i["x"]
            
            if xyz_i.shape[0] == 0:
                raise ValueError()
                
            # Store the frames.
            self._print("- Selected {} frames".format(repr(xyz_i.shape)))
            ofo.append(ofo_i)

        if not ofo:
            raise ValueError("No data found for {}".format(
                prot_data_files.name))
        
        # Concatenate all the OpenFold trajectories.
        ofo = {k: np.concatenate([ofo_i[k] for ofo_i in ofo], axis=0) \
               for k in ofo[0].keys()}
        
        # Sample frames with mode "ensemble".
        if self.frames_mode == "ensemble":
            ofo = self.sample_data(
                data=ofo,
                n_samples=self._get_n_frames(prot_data_files.name)
            )

        self._print("+ Will store {} frames".format(repr(ofo["x"][0].shape)))

        # Get template data.
        if self.tbm.get("mode") is None:
            # xyz_tem = None
            pass
        elif self.tbm["mode"] in ("single", "random"):
            # if self.tbm["mode"] == "single":  # Get a single template trajectory.
            #     xyz_tem = self._load_xyz_from_traj(
            #         traj_fp=prot_data_files.template[0],
            #         top_fp=prot_data_files.top_fp
            #     )
            # elif self.tbm["mode"] == "random":
            #     # TODO.
            #     xyz_tem = xyz
            # else:
            #     raise KeyError(self.tbm["mode"])
            raise NotImplementedError()
        else:
            raise KeyError(self.tbm["mode"])

        # Get the residue indices.
        res_ids = self._get_residue_indices(top_traj)
        
        # Initialize a CG_Protein object.
        protein_class = self._get_protein_class()
        protein_args = self._get_protein_args(top_traj, ofo, ofo_top)
        protein_obj = protein_class(
            name=prot_data_files.name,
            seq=seq_three,
            alphabet=self.alphabet,
            xyz=None,
            xyz_tem=None,
            r=res_ids,
            **protein_args
        )
        return protein_obj
    
    def _get_protein_args(self, top_traj, ofo, ofo_top):
        return {
            "topology": top_traj.topology,
            "ofo": ofo,
            "ofo_topology": ofo_top
        }
    
    def _process_ofo(self, ofo):
        if "x" in ofo:
            return ofo
        else:
            # Add 'x'.
            ofo['x'] = ofo['atom14_gt_positions'][:,:,1,:]*0.1
            # Add 'backbone_rigid_tensor'.
            ofo['backbone_rigid_tensor'] = ofo['rigidgroups_gt_frames'][:,:,0,...]
            # Add 'rigidgroups_alt_gt_frames'.
            rigidgroups_alt_gt_frames = ofo['rigidgroups_gt_frames'].clone()
            rigidgroups_alt_gt_frames[:,:,5:7,...] = ofo['rigidgroups_alt_gt_frames']
            ofo['rigidgroups_alt_gt_frames'] = rigidgroups_alt_gt_frames
            # Add 'atom14_alt_gt_positions'.
            atom14_alt_gt_positions = ofo['atom14_gt_positions'].clone()
            atom14_alt_gt_positions[:,:,6:10,...] = ofo['atom14_alt_gt_positions']
            ofo['atom14_alt_gt_positions'] = atom14_alt_gt_positions
            return ofo


    def get_xyz_data(self, prot_idx, frame_idx):
        """
        Returns a frames storing OpenFold features. Will also convert to
        tensors.
        """
        
        ### Get raw data.

        # Get the OFO frame.
        prot_obj = self.protein_list[prot_idx]

        # Get a template xyz frame.
        if self.tbm.get("mode") is not None:
            raise KeyError(self.tbm["mode"])


        ### Perturb.

        # Add Gaussian noise to original coordinates.
        # xyz = self._add_noise_to_xyz(xyz)

        # Weighted sum of Gaussian noise and original coordinates.
        # xyz = self._perturb_xyz(xyz)

        ### Convert to tensors and return data.
        data = {}
        for k in per_conf_attrs:  # Per-conformer features.
            data[k] = prot_obj.ofo[k][frame_idx]
            if self.use_metaencoder:
                data[f"{meta_prefix}_{k}"] = prot_obj.meta_ofo[k][frame_idx]
        
        for k in per_sys_attrs:  # Per-system features.
            if k not in ("a", "r"):
                data[k] = prot_obj.ofo_topology[k]
        # if self.use_raw_xyz:
        #     data["aa_x"] = xyz
        return data


################################################################################
# Dataset for proteins structures encoded at training time.                    #
################################################################################

class LiveEncodedAllAtomProteinDataset(LiveEncodedDatasetMixin, AllAtomProteinDataset):

    def _get_minimal_live_dataset_cls(self):
        return AllAtomMinimalDataset
    
    def _get_minimal_live_dataset_args(self, protein):
        return {"topology": protein.topology}

    def load_protein_data(self,
            prot_data_files,
            # sel_trajectory=None
        ):
        return AllAtomProteinDataset.load_protein_data(
            self,
            prot_data_files=prot_data_files,
            # sel_trajectory=sel_trajectory
        )


class AllAtomMinimalDataset(AllAtomProteinDataset):
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
            topology: mdtraj.Topology,
            xyz: np.ndarray = None,
            n_frames: int = None,
            xyz_tem: np.ndarray = None,
            bead_type: str = "ca",
            alphabet: str = None,
            res_ids_mode: str = None,
            random_seed: int = None,
            verbose: bool = True,
            
        ):
        self.bead_type = bead_type
        self.res_ids_mode = res_ids_mode
        self.xyz_sigma = None
        self.xyz_perturb = None
        self.use_aa_embeddings = False
        self.alphabet = alphabet
        self.verbose = verbose
        self._set_random_obj(random_seed)
        self.tbm = {}
        self.use_raw_xyz = False

        if xyz is None:
            raise ValueError()

        prot_obj = AA_Protein(
            name=name,
            seq=seq,
            xyz=xyz,
            xyz_tem=xyz_tem,
            topology=topology
        )
        self.protein_list = [prot_obj]

        self.load_data()


    def load_data(self):
        self.time_a = 0  ###
        self.time_m = 0  ###
        self.time_c = 0  ###
        self.frames = []
        n_residues = len(self.protein_list[0].seq)
        for i in range(self.protein_list[0].xyz.shape[0]):
            self.frames.append([0, n_residues, i])
        if not self.frames:
            raise ValueError("No frames found")


class AllAtomMinimalLiveDataset(AllAtomMinimalDataset):
    """
    Minimal dataset storing data for one protein system.
    This subclass is used only for live encoding.
    """
    
    # def __init__(self, *args, **kwargs):
    #     MinimalLiveDataset.__init__(self, *args, **kwargs)
    
    def load_data(self):
        self.time_a = 0  ###
        self.time_m = 0  ###
        self.time_c = 0  ###
        self.load_live_data(use_tbm=False)

    def load_live_data(self, use_tbm: bool = False):
        """Add the snapshots for this protein to the dataset."""
        self.frames = []
        xyz = self.protein_list[0].xyz if not use_tbm \
                                       else self.protein_list[0].xyz_tem
        n_residues = len(self.protein_list[0].seq)
        for i in range(xyz.shape[0]):
            self.frames.append([0, n_residues, i])
        if not self.frames:
            raise ValueError("No frames found")