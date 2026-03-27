import os
import h5py
import torch
import numpy as np
import pytorch_lightning as pl

from torch import Tensor
from typing import List, Union
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

import mrlfads.paths as path
from .utils.common_utils import Batch, convert_byte, apply_along_axis

class BasicDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for session-based, multi-area neural data.

    This DataModule loads neural datasets stored in HDF5 files and organized
    by session. Each session contains multiple data arrays corresponding to
    neural activity, inputs, ground truth signals, metadata, etc.

    Expected HDF5 file structure:
        File
        ├── session_index
        │   ├── hidden_states
        │   ├── inputs
        │   ├── message
        │   ├── ground_truth
        │   └── info
    """
    
    def __init__(
        self,
        # ----- main parameters ---------------------- #
        filename,
        area_names,
        session_idxs: list = [0],              # index of the session to load from HDF5
        session_area_dict: dict = None,        # session-area specification, overrides session_idxs
        time_dim: int = 200,

        # ----- train / validation specification ----- #
        batch_size: int = 16,
        p_split: list = [0.85, 0.15],          # train/validation split
        train_val_split_seed: int = None,      # for reproducibility
        batch_lim: int = None,                 # limits maximum total batch number
        nn_lim: int = None,                    # limits maximum amount of neurons used
        nn_select: str = "maximum",            # use maximally responsive neurons
        shuffle: bool = False,                 # shuffle batches before selection

        # ----- others ------------------------------- #
        ignore_info: bool = False,             # if True, omits loading metadata (e.g. ground truth)
        use_input: bool = False,               # whether external input data (e.g. photostimulation) is used
        input_areas: list = [],                # areas that receive external input data
        datapath_override: str = None,
        message_names_override: list = None,
    ):
        """
        Args:
            filename: Filename of the HDF5 data file.
            area_names: List of brain area names to load from the dataset.
            session_idxs: Indices of the sessions to load from the HDF5 file.
                Defaults to [0].

            batch_size: Number of samples per batch. Defaults to 16.
            p_split: Proportions used for train/validation splitting. Defaults to
                [0.85, 0.15].
            train_val_split_seed: Random seed used for reproducible train/validation
                splits. Defaults to None.
            batch_lim: Optional limit on the total number of batches loaded.
                Defaults to None.
            nn_lim: Optional limit on the number of neurons used. Defaults to None.
            nn_select: Strategy for selecting neurons when `nn_lim` is specified.
                Defaults to "maximum".
            shuffle: If True, shuffle batches before selection or truncation.
                Defaults to False.

            ignore_info: If True, metadata (e.g., ground truth or auxiliary info)
                is not loaded. Defaults to False.
            use_input: If True, include photostimulation-related inputs when
                loading the data. Defaults to False.
            datapath_override: Optional override for the base data directory.
                Defaults to None.
            message_names_override: Optional list of message names to load instead
                of those defined in the dataset. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        hps = self.hparams
        self.train_session_datasets = []
        self.val_session_datasets = []
        self.val_session_indices = []
        
        datapath = path.datapath if hps.datapath_override is None else hps.datapath_override
        filename = os.path.join(datapath, str(hps.filename), "data.h5")
            
        # Overrides for session-area information
        if isinstance(hps.session_area_dict, type(None)):
            hps.session_area_dict = {sidx: hps.area_names for sidx in hps.session_idxs}
        hps.session_idxs = list(hps.session_area_dict.keys())
            
        with h5py.File(filename, 'r') as file:
            
            for session_idx in hps.session_idxs:
                group = file[str(session_idx)]

                # Turn data into dictionary, then SessionAreaDataset
                area_data_dict = {}
                for area_name in hps.session_area_dict[session_idx]:

                    if isinstance(area_name, str):
                        dataset_name = f"area-{area_name}"
                        ds = group[dataset_name]
                        assert ds.attrs.get("type") == "hidden_state"

                        arr = ds[:]
                        label_area_name = area_name

                    elif isinstance(area_name, list):
                        arrs = []
                        for sub_area_name in area_name[1:]:
                            dataset_name = f"area-{sub_area_name}"
                            ds = group[dataset_name]
                            assert ds.attrs.get("type") == "hidden_state"

                            arr = ds[:]
                            arrs.append(arr)
                        arr = np.concatenate(arrs, axis=2)
                        label_area_name = area_name[0]

                    # Shuffle
                    batch_dim = len(arr) if not hps.batch_lim else min([hps.batch_lim, len(arr)])
                    time_dim = arr.shape[1]
                    if hps.shuffle:
                        batches = np.array(range(batch_dim))
                        np.random.shuffle(batches)
                        arr = arr[batches]

                    # Neuron number limits, get the ones with most fluctuation
                    if not hps.nn_lim:
                        self.included_neurons = np.arange(arr.shape[-1]).astype(int)
                    else:
                        if hps.nn_select == 'maximum':
                            fluctuations = np.std(arr, axis=(0, 1))
                            self.included_neurons = self.included_neurons[np.flip(np.argsort(fluctuations))][:hps.nn_lim]
                        else:
                            self.included_neurons = self.included_neurons[:hps.nn_lim]

                    # Separate data into batches
                    for bi in range(batch_dim):
                        if bi not in area_data_dict.keys():
                            area_data_dict[bi] = {}
                        temp = arr[bi] # to avoid axis issues
                        area_data_dict[bi][label_area_name] = temp[:, self.included_neurons]
                        
                # Fillers for missing areas
                missing_areas = [an for an in hps.area_names if an not in hps.session_area_dict[session_idx]]
                for bi in range(batch_dim):
                    for missing_area in missing_areas:
                        area_data_dict[bi][missing_area] = np.zeros((time_dim, 0))
                    
                    # Re-order by area_names
                    area_data_dict[bi] = {an: area_data_dict[bi][an] for an in hps.area_names}

                # Get external inputs
                ext_input_dict = {}
                if hps.use_input:
                    if len(hps.input_areas) == 0: input_areas = hps.area_names
                    else: input_areas = hps.input_areas
                    
                    for area_name in hps.area_names:
                        dataset_name = f"inputs-{area_name}"
                        ds = group[dataset_name]
                        assert ds.attrs.get("type") == "inputs"

                        arr = ds[:]
                        batch_dim = len(arr)

                        for bi in range(batch_dim):
                            if area_name == hps.area_names[0]:
                                ext_input_dict[bi] = {}
                                
                            if area_name in input_areas:
                                temp = arr[bi]
                            else:
                                temp = np.zeros((hps.time_dim, 1))
                            ext_input_dict[bi][area_name] = temp

                # Get messages, info and ground truth
                if isinstance(hps.message_names_override, type(None)):
                    message_names = []
                    for dataset_name in group.keys():
                        dataset = group[dataset_name]
                        datatype = dataset.attrs.get("type")
                        if datatype in ["message", "info", "ground_truth"]:
                            message_names.append(dataset_name)
                else:
                    message_names = hps.message_names_override

                # Append messages and info
                info_strings = []
                for bi in range(batch_dim):
                    info_dict = {"batch": np.zeros((arr.shape[1], 0))} # (time, 0)
                    if not hps.ignore_info:
                        for message_name in message_names:
                            try:
                                item = group[message_name][:][bi]
                                if not isinstance(item, bytes):
                                    info_dict[message_name.split("-")[1]] = item
                                else:
                                    info_dict[message_name.split("-")[1]] = item.decode('utf-8')
                            except:
                                if bi==0:
                                    print(f"Warning: message {message_name} is not imported.")
                    info_strings.append(info_dict)

                # Get SessionAreaDataset
                session_dataset = SessionAreaDataset(area_data_dict, info_strings, ext_input_dict)
                if hps.train_val_split_seed:
                    train_ds, val_ds = random_split(session_dataset, hps.p_split)
                else:
                    train_ds, val_ds = random_split(
                        session_dataset,
                        hps.p_split,
                        generator=torch.Generator().manual_seed(hps.train_val_split_seed)
                    )
                self.train_session_datasets.append(train_ds)
                self.val_session_datasets.append(val_ds)
                self.val_session_indices.append(val_ds.indices)
        
    def train_dataloader(self, shuffle=True):
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                shuffle=shuffle,
                drop_last=False,
            )
            for i, ds in enumerate(self.train_session_datasets)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=len(ds),
                shuffle=False,
            )
            for i, ds in enumerate(self.val_session_datasets)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")

class SessionAreaDataset(Dataset):
    def __init__(
        self,
        data_dict: dict, # batch index --> area_name --> tensor of shape (time, num neurons)
        info_list: list, # shape (batch, *)
        hps: dict,
        ext_input_dict: dict = {},
        session_idx: int = 0,
    ):
        self.hps = hps
        self.session_idx = session_idx
        
        # Process data, info
        self.data_list = list(data_dict.values())
        self.info_list = info_list
        
        # Get external input, if applicable
        ext_input_list = list(ext_input_dict.values())
        self.empty_dict = {key: np.zeros((*value.shape[:1], 0))
                           for key, value in self.data_list[0].items()
                          }
        
        if len(ext_input_list) > 0:
            self.ext_input_gen = lambda idx: ext_input_list[idx]
        else:
            self.ext_input_gen = lambda idx: self.empty_dict
        
    def __getitem__(self, idx):
        return Batch(
                    encod_data=self.data_list[idx],
                    ext_input=self.ext_input_gen(idx),
                ), self.info_list[idx]
    
    def __len__(self): return len(self.data_list)
    
def sample_and_split(
    filepath: str,
    newfilepath: str,
    area_names: list,
    sess_dict: dict,
):
    """Split an HDF5 dataset into sessions and subsample batches/neurons.

    Reads an existing HDF5 dataset, creates multiple sessions according to
    `sess_dict`, and writes the resulting dataset to `newfilepath`.

    Args:
        filepath: Path to the source HDF5 (.h5) file.
        newfilepath: Path to the output HDF5 (.h5) file to write.
        area_names: Names of brain areas to include.
        sess_dict: Mapping from session identifier to selection indices.
            Each value must be a two-element sequence:
            (batch_indices_to_keep, neuron_indices_to_keep).
    """
    # Create folder and copy everything
    shutil.copytree(os.path.dirname(filepath), os.path.dirname(newfilepath), dirs_exist_ok=True)
    
    with h5py.File(filepath, 'r') as src, h5py.File(newfilepath, 'w') as dst:
        group = src['0']
        
        # Create multiple sessions
        for sess, (batch_idxs, nn_idxs) in sess_dict.items():
            
            dst_group = dst.create_group(str(sess))
            
            # Handle hidden states
            for area_name in area_names:

                # Extract
                if isinstance(area_name, str):
                    dataset_name = f"area-{area_name}"
                    ds = group[dataset_name]
                    assert ds.attrs.get("type") == "hidden_state"

                    arr = ds[:]
                    label_area_name = area_name

                elif isinstance(area_name, list):
                    arrs = []
                    for sub_area_name in area_name[1:]:
                        dataset_name = f"area-{sub_area_name}"
                        ds = group[dataset_name]
                        assert ds.attrs.get("type") == "hidden_state"

                        arr = ds[:]
                        arrs.append(arr)
                    arr = np.concatenate(arrs, axis=2)
                    label_area_name = area_name[0]

                # Sample based on sess_dict
                arr_modified = arr[batch_idxs][..., nn_idxs]
                dst_ds = dst_group.create_dataset('area-'+label_area_name, data=arr_modified)
                dst_ds.attrs["type"] = ds.attrs.get("type", "unknown")
                
            # Handle all other metadata
            for key in group.keys():
                if not key.startswith('area'):
                    ds = group[key]
                    arr_modified = ds[:][batch_idxs]
                    dst_ds = dst_group.create_dataset(key, data=arr_modified)
                    dst_ds.attrs["type"] = ds.attrs.get("type", "unknown")
