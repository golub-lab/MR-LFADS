import os
import h5py
import shutil
import torch
import numpy as np
import pytorch_lightning as pl
import mrlfads.paths as path

from torch import Tensor
from typing import List, Union
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn.model_selection import train_test_split
from .utils import Batch
    
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
        session_idx: int = 0,                  # index of the session to load from HDF5

        # ----- train / validation specification ----- #
        batch_size: int = 16,
        p_split: list = [0.85, 0.15],          # train/validation split
        no_split: bool = False,                # if True, return same dataset for both training and validation
        train_val_split_seed: int = None,
        batch_lim: int = None,                 # limits maximum total batch number
        nn_lim: int = None,                    # limits maximum amount of neurons used
        nn_select: str = "maximum",            # use maximally responsive neurons
        shuffle: bool = False,                 # shuffle batches before selection

        # ----- others ------------------------------- #
        ignore_info: bool = False,             # if True, omits loading metadata (e.g. ground truth)
        use_photostim: bool = False,           # whether photostimulation data is used
        datapath_override: str = None,
        message_names_override: list = None,
    ):
        """
        Args:
            filename: Filename of the HDF5 data file.
            area_names: List of brain area names to load from the dataset.
            session_idx: Index of the session to load from the HDF5 file.
                Defaults to 0.

            batch_size: Number of samples per batch. Defaults to 16.
            p_split: Proportions used for train/validation splitting. Defaults to
                [0.85, 0.15].
            no_split: If True, the same dataset is returned for both training and
                validation. Defaults to False.
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
            use_photostim: If True, include photostimulation-related inputs when
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
            
        with h5py.File(filename, 'r') as file:
            group = file[str(hps.session_idx)]
            
            # Turn data into dictionary, then SessionAreaDataset
            area_data_dict = {}
            for area_name in hps.area_names:
                
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
                    
            # Get external inputs
            ext_input_dict = {}
            if hps.use_photostim:
                for area_name in hps.area_names:
                    dataset_name = f"inputs-{area_name}"
                    ds = group[dataset_name]
                    assert ds.attrs.get("type") == "inputs"

                    arr = ds[:]
                    batch_dim = len(arr)

                    for bi in range(batch_dim):
                        if area_name == hps.area_names[0]:
                            ext_input_dict[bi] = {}
                        temp = arr[bi]
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
            if hps.no_split:
                train_ds = val_ds = session_dataset
            elif not hps.train_val_split_seed:
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
    """Dataset class for per-session neural activity organized by brain area.

    Each dataset item corresponds to a batch and contains neural activity
    grouped by brain area, along with optional external inputs.
    """
    def __init__(
        self,
        data_dict: dict,
        info_list: list,
        ext_input_dict: dict = {},
    ):
        """
        Args:
            data_dict: Mapping from batch index to a mapping of area name to
                arrays of shape (num_timesteps, num_neurons).
            info_list: Per-batch metadata.
            ext_input_dict: Optional mapping of external inputs with the same
                structure as `data_dict`. Defaults to an empty dict.
        """
        self.data_list = list(data_dict.values()) 
        self.info_list = info_list
        self.empty_dict = {key: np.zeros(list(value.shape[:1]) + [0]) for key, value in self.data_list[0].items()}
        
        # Get external input, if applicable
        ext_input_list = list(ext_input_dict.values())
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
                
        
                
        
            
            