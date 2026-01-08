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
from .utils import Batch, apply_along_axis, convert_byte
from .datamodules import SessionAreaDataset

class MesoMapDataModule(pl.LightningDataModule):
    """
    Mesoscale activity map data from 
        ``Brain-wide neural activity underlying memory-guided movement``, Cell, Chen et al. (2024).
    
    Reads a .h5 file with the hierarchical structure of:
        file (subject) --> group (session) --> dataset (area data or meta data)
        
        - ``file`` is named as "sub-{subject_id}"
        - ``session`` is named as "ses-{session_id}"
        - ``area data`` is named as "area-{abbreviation}", and has attribute ``type`` == "data"
        - ``info`` is named as "{information_type}", and has attribute ``type`` == "info"
    """
    def __init__(
        self,
        subject_id: int,
        session_idx: Union[int, str],
        area_names: list,
        time_dim: int = 500,
        
        # Dataset params
        batch_size: int = 16,
        p_split: list = [0.8, 0.2],
        shuffle: bool = False,
        train_val_split_seed: int = None,
        no_split: bool = False,
        batch_lim: int = None,
        time_lim: int = None,
        
        # Photostim
        use_photostim: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        hps = self.hparams
        to_string = lambda l: [s.decode('utf-8') for s in l]
        
        filename = os.path.join(path.datapath, f"sub-{hps.subject_id}", f"sub-{hps.subject_id}.h5")
        with h5py.File(filename, "r") as file:

            # Use sessions according to session_idx: can be list, str, or 'all'
            session_names = list(file.keys())
            if isinstance(hps.session_idx, list): session_idxs = hps.session_idx
            elif hps.session_idx == "all": session_idxs = list(range(len(session_names)))
            else:
                raise ValueError("session_idx cannot be ", hps.session_idx)
            
            # Only retain sessions according to available keys
            area_names_flattened = []
            for area_name in hps.area_names:
                if isinstance(area_name, str): area_names_flattened.append(area_name)
                elif isinstance(area_name, list): area_names_flattened += area_name[1:] # First one is the combined name
                else: raise TypeError()
            dataset_names = [f"area-{key}" for key in area_names_flattened]
            available_keys = get_session_and_areas(hps.subject_id, return_ds_name=True)
            session_idxs = [si for si in session_idxs if
                           np.all([dsname in available_keys[si] for dsname in dataset_names])]
            
            # Iterate through sessions
            self.train_session_datasets = []
            self.val_session_datasets = []
            self.val_session_indices = [] # store validation batch indices for analysis purposes
            self.included_batch_indices = [] # to get true validation batch, use arr[included_batch_indices][val_session_indices]
            if hps.use_photostim: self.photostim_batch_indices = []
            
            for seidx, si in enumerate(session_idxs):
                group = file[session_names[si]]
                
                # Filter data by photostim_onset
                filter1 = group["photostim_onset"][:]
                if not hps.use_photostim: 
                    included_batches1 = np.where(filter1 == b"N/A")[0]
                else:
                    included_batches1 = np.array(range(len(filter1)))
                    photostim_batches1 = np.where(filter1 != b"N/A")[0]
                    
                # Filter data by trial duration (needs to be within [4.5, 5.5])
                trial_starts = group["start_time"][:]
                trial_stops = group["stop_time"][:]
                trial_durs = trial_stops - trial_starts
                included_batches2 = np.where(np.abs(trial_durs - 5.0) <= 0.5)[0]
                included_batches = np.intersect1d(included_batches1, included_batches2)
                
                # Batch limitations
                if hps.batch_lim:
                    included_batches = included_batches[:hps.batch_lim]
                    max_batches = included_batches[-1]
                    photostim_batches1 = np.array([pb for pb in photostim_batches1 if pb <= max_batches])
                
                # Photostim batches
                if hps.use_photostim:
                    photostim_batches = np.intersect1d(photostim_batches1, included_batches2)
                batch_dim = len(included_batches)
                
                # Get necessary information
                info1 = group["trial_instruction"][:][included_batches]
                info2 = group["outcome"][:][included_batches]
                info_strings = ["_".join(to_string(info)) for info in zip(info1, info2)]
                
                # Get ext input, if applicable
                if hps.use_photostim:
                    trial_durations = group["stop_time"][:] - group["start_time"][:]
                    ext_input_array = self.gen_ext_input(
                        photostim_batches,
                        hps.time_dim,
                        trial_durations,
                        group["photostim_onset"][:],
                        group["photostim_duration"][:],
                        group["photostim_power"][:]
                    )
                    
                # Turn data into dictionary, then SessionAreaDataset
                area_data_dict = {}
                ext_input_dict = {}
                info_dict = {}
                
                for area_name in hps.area_names:
                    
                    # For each area, give it a different batch order
                    if hps.shuffle: np.random.shuffle(included_batches)
                    
                    # If the area is a single area
                    if isinstance(area_name, str):
                        dataset_name = f"area-{area_name}"
                        ds = group[dataset_name]
                        assert ds.attrs.get("type") == "data"
                        
                        arr = ds[:][included_batches]
                        arr = np.swapaxes(arr, 1, 2) # shape = (batch, time, # neurons)
                        arr = arr[:, :hps.time_lim]
                        label_area_name = area_name
                    
                    # If the area is comprised of multiple areas
                    elif isinstance(area_name, list):
                        arrs = []
                        for sub_area_name in area_name[1:]:
                            dataset_name = f"area-{sub_area_name}"
                            ds = group[dataset_name]
                            assert ds.attrs.get("type") == "data"

                            arr = ds[:][included_batches]
                            arr = np.swapaxes(arr, 1, 2) # shape = (batch, time, # neurons)
                            arr = arr[:, :hps.time_lim]
                            arrs.append(arr)
                        arr = np.concatenate(arrs, axis=2)
                        label_area_name = area_name[0]
                        
                    # Separate data into batches
                    for bi in range(batch_dim):
                        if bi not in area_data_dict.keys():
                            area_data_dict[bi] = {}
                        area_data_dict[bi][label_area_name] = arr[bi]
                        
                        if hps.use_photostim:
                            if bi not in ext_input_dict.keys(): ext_input_dict[bi] = {}
                            if True: #in_photostim_target(area_name):
                                ext_input_dict[bi][label_area_name] = ext_input_array[bi] # shape = (time, 1)
                            else:
                                ext_input_dict[bi][label_area_name] = np.zeros(hps.time_dim).reshape(-1, 1) # shape = (time, 1)
                                
                    # Separate metadata into batches
                    info_strings_in_batch = []
                    for bi in range(batch_dim):
                        info_dict = {"instruction_outcome": info_strings[bi]}
                        info_strings_in_batch.append(info_dict)
                        
                session_dataset = SessionAreaDataset(area_data_dict, info_strings_in_batch, ext_input_dict)
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
                self.included_batch_indices.append(included_batches)
                
                if hps.use_photostim:
                    self.photostim_batch_indices.append(photostim_batches)
                
    def gen_ext_input(self, batches, time, trial_durations, onset, duration, power):
        
        def pulse_func(x, start, end, amplitude):
            func1 = np.heaviside(x - start, 1)
            func2 = np.heaviside(end - x, 1)
            return (func1 + func2 - np.ones(func1.shape)) * amplitude
        
        x = np.array(range(time))
        photostim = np.zeros((len(onset), time, 1))
        for bi in batches:
            start = int( float(onset[bi]) / trial_durations[bi] * time )
            end = int( (float(onset[bi]) + float(duration[bi])) / trial_durations[bi] * time )
            photostim[bi, :, 0] = pulse_func(x, start, end, float(power[bi]))
            
        return photostim # shape = (batch, time, 1)
                
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
    
def get_session_and_areas(subject_id, return_ds_name=False):
    '''Returns {session: available areas} in file `sub-{subject_id}`.
    '''
    filename = os.path.join(path.datapath, f"sub-{subject_id}", f"sub-{subject_id}.h5")
    with h5py.File(filename, 'r') as file:

        # Use sessions according to session_idx
        session_names = list(file.keys())
        session_idxs = list(range(len(session_names)))
        
        area_names_session = {}
        for si in session_idxs:
            group = file[session_names[si]]
            
            dataset_names = []
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset) and ("area" in name):
                    dataset_names.append(name)
            group.visititems(collect_datasets)
        
            if return_ds_name:
                area_names_session[si] = dataset_names
            else:
                area_names = [k[5:] for k in dataset_names]
                area_names_session[si] = area_names
            
        return area_names_session
    
def get_behavioral_events(subject_id, session_idx, key, convert=True):
    filename_behavior = os.path.join(path.datapath, f"sub-{subject_id}_behavioral_events/sub-{subject_id}_behavioral_events.h5") 
    file = h5py.File(filename_behavior, "r")

    session_names = list(file.keys())
    session_name = session_names[session_idx]
    group = file[session_name]
    info_dict = {k: v[:] for k, v in group.items()}
    file.close()
    
    if not key:
        print(info_dict.keys())
    else:
        arr = info_dict[key]
        if convert: arr = apply_along_axis(arr, convert_byte)
        return arr