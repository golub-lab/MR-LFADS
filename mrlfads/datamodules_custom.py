import os
import h5py
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch import Tensor
from typing import List, Union
# from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

import mrlfads.paths as path
from .datamodules import SessionAreaDataset
from .utils.common_utils import Batch, convert_byte, apply_along_axis

class MesoMapDataModule(pl.LightningDataModule):
    """
    Mesoscale activity map data from 
        ``Brain-wide neural activity underlying memory-guided movement``, Chen et al. (2024).
    
    Reads a .h5 file with the hierarchical structure of:
        file (subject) --> group (session) --> dataset (area data or info)
        
        - ``file`` is named as "sub-{subject_id}"
        - ``session`` is named as "ses-{session_id}"
        - ``area data`` is named as "area-{abbreviation}", and has attribute ``type`` == "data"
        - ``info`` is named as "{information_type}", and has attribute ``type`` == "info"
    """
    def __init__(
        self,
        subject_id: int,
        session_idx: Union[int, str, list],
        area_names: list,
        time_dim: int = 500,
        session_area_dict: dict = None, 
        
        # Train/val batch settings
        batch_size: int = 16,
        p_split: list = [0.8, 0.2],
        train_val_split_seed: int = None,
        
        # Data restrictions
        batch_lim: int = None,
        time_lim: int = None,
        shuffle: bool = False,
        
        # Photostim
        use_photostim: bool = False,
        photostim_areas: list = ['ALM'],
        
        # Metadata
        info_keys: dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        hps = self.hparams
        to_string = lambda l: [s.decode('utf-8') for s in l]
        
        # Overrides for session-area information ##
        if isinstance(hps.session_area_dict, type(None)):
            hps.session_area_dict = {sidx: list(hps.area_names) for sidx in hps.session_idx}
        hps.session_idx = list(hps.session_area_dict.keys())
        
        filename = os.path.join(path.datapath, f"sub-{hps.subject_id}", f"sub-{hps.subject_id}.h5")
        with h5py.File(filename, "r") as file:

            # Use sessions according to session_idx
            session_names = list(file.keys())
            if isinstance(hps.session_idx, list): session_idxs = hps.session_idx
            elif hps.session_idx == "all": session_idxs = list(range(len(session_names)))
            else:
                raise ValueError("session_idx cannot be ", hps.session_idx)
            
            # Only retain sessions according to available keys
            # area_names_flattened = []
            # for area_name in hps.area_names:
            #     if isinstance(area_name, str): area_names_flattened.append(area_name)
            #     elif isinstance(area_name, list): area_names_flattened += area_name[1:] # First one is the combined name
            #     else: raise TypeError()
            # dataset_names = [f"area-{key}" for key in area_names_flattened]
            # available_keys = get_session_and_areas(hps.subject_id, return_ds_name=True)
            # session_idxs = [si for si in session_idxs if
            #                np.all([dsname in available_keys[si] for dsname in dataset_names])]
            
            # Iterate through sessions
            # Data will get excluded based on certain criteria
            # To get true validation batch, use arr[included_batch_indices][val_session_indices]
            self.train_session_datasets = []
            self.val_session_datasets = []
            self.train_session_indices = []
            self.val_session_indices = []
            self.included_batch_indices = []
            if hps.use_photostim: self.photostim_batch_indices = []
            
            for seidx, si in enumerate(hps.session_idx):
                group = file[session_names[si]]
                
                # Filter data by photostim_onset
                filter1 = group["photostim_onset"][:]
                if not hps.use_photostim: 
                    included_batches1 = np.where(filter1 == b"N/A")[0]
                else:
                    included_batches1 = np.array(range(len(filter1)))
                    photostim_batches1 = np.where(filter1 != b"N/A")[0]
                    
                # Filter data by trial duration
                # Trials with substantially different start times are excluded
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
                
                # Get metadata for different trial conditions
                info1 = group["trial_instruction"][:][included_batches]
                info2 = group["outcome"][:][included_batches]
                info_strings = ["_".join(to_string(info)) for info in zip(info1, info2)]
                other_info_strings = {}
                for info_key, (info_process_func, info_kwargs) in hps.info_keys.items():
                    if isinstance(info_kwargs, type(None)): info_kwargs = {}
                    other_info_strings[info_key] = info_process_func(
                        group,
                        hps,
                        **info_kwargs,
                    )[included_batches]
                
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
                
                for area_name in hps.session_area_dict[si]: 
                    
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
                        
                    else:
                        raise TypeError()
                        
                    # Separate data into batches
                    for bi in range(batch_dim):
                        if bi not in area_data_dict.keys():
                            area_data_dict[bi] = {}
                            info_dict[bi] = {}
                        area_data_dict[bi][label_area_name] = arr[bi]
                        info_dict[bi]["instruction_outcome"] = info_strings[bi]
                        for info_key in hps.info_keys:
                            info_dict[bi][info_key] = other_info_strings[info_key][bi]
                        
                        if hps.use_photostim:
                            if bi not in ext_input_dict.keys(): ext_input_dict[bi] = {}
                            if label_area_name in hps.photostim_areas:
                                ## is this even the correct batch??
                                ext_input_dict[bi][label_area_name] = ext_input_array[bi] # (time, 1)
                            else:
                                ext_input_dict[bi][label_area_name] = np.zeros((hps.time_dim, 1))
                        
                session_dataset = SessionAreaDataset(
                    area_data_dict, info_dict, self.hparams, ext_input_dict, seidx
                )
                if not hps.train_val_split_seed:
                    train_ds, val_ds = random_split(session_dataset, hps.p_split)
                else:
                    train_ds, val_ds = random_split(
                        session_dataset,
                        hps.p_split,
                        generator=torch.Generator().manual_seed(hps.train_val_split_seed)
                    )
                self.train_session_datasets.append(train_ds)
                self.val_session_datasets.append(val_ds)
                self.train_session_indices.append(train_ds.indices)
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
                num_workers=min(8, os.cpu_count()),
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
    
def get_behavior(subject_id, session_idx, key):
    filename_behavior = os.path.join(path.datapath, f"sub-{subject_id}_behavior/{subject_id}_behavior.h5") 
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
        return arr.astype('float32')
    
def get_neuron_statistics(subject_id, session_idx, area_names):
    
    filename = os.path.join(path.datapath, f"sub-{subject_id}", f"sub-{subject_id}.h5")
    mean_dict = {} # area --> neuron idx --> (norm, sub-area-idx)
    
    with h5py.File(filename, 'r') as file:

        session_names = list(file.keys())
        group = file[session_names[session_idx]]
        
        for area_name in area_names:
            # If it is a single area
            if isinstance(area_name, str):
                ds = group[f"area-{area_name}"]
                arr = ds[:]
                arr = np.swapaxes(arr, 1, 2) # shape = (batch, time, # neurons)
                means = np.mean(arr, axis=(0, 1))
                mean_dict[area_name] = {i: (means[i], 0) for i in range(len(means))}
                
            # If the area is comprised of multiple areas
            elif isinstance(area_name, list):
                mean_dict[area_name[0]] = {}
                base = 0
                for isub, sub_area_name in enumerate(area_name[1:]):
                    ds = group[f"area-{sub_area_name}"]
                    arr = ds[:]
                    arr = np.swapaxes(arr, 1, 2) # shape = (batch, time, # neurons)
                    means = np.mean(arr, axis=(0, 1))
                    mean_dict[area_name[0]].update( {base + i: (means[i], isub) for i in range(len(means))} )
                    base += len(means)
                    
            else:
                import pdb; pdb.set_trace()
                raise TypeError(f'area_name must be str or list.')
                
        return mean_dict

def process_licks(lick_lists, start_times, stop_times, time_dim=500):
    
    # Merge lick time lists by trial
    batch = len(lick_lists[0])
    merged_licks = []
    for b in range(batch):
        licks_b = []
        for src in lick_lists:
            if isinstance(src[b], list): 
                licks_b.extend(src[b])
            elif isinstance(src[b], float):
                licks_b.append(src[b])
            else:
                raise TypeError()
        merged_licks.append(licks_b)

    res = np.zeros((batch, time_dim, 1), dtype=np.float32)
    for b, lick_times_abs in enumerate(merged_licks):
        if len(lick_times_abs) == 0: continue
        
        start = start_times[b]
        stop = stop_times[b]
        total_time = stop - start

        lick_times = np.asarray(lick_times_abs, dtype=np.float64) - start
        bin_pos = lick_times / total_time * time_dim
        bin_idx = np.floor(bin_pos).astype(np.int64)

        # numerical safety: a lick extremely close to stop could map to time_dim
        bin_idx = np.clip(bin_idx, 0, time_dim - 1)
        res[b, np.unique(bin_idx), 0] = 1.0
    return res

def collect_nonselective_licks(group, hparams, **kwargs):
    kws = {'subject': '440959', 'session': 3}
    kws.update(kwargs)
    
    lefts = get_behavioral_events(kws['subject'], kws['session'], 'left_lick_times')
    rights = get_behavioral_events(kws['subject'], kws['session'], 'right_lick_times')
    starts = get_behavioral_events(kws['subject'], kws['session'], 'start_time', convert=False)
    stops = get_behavioral_events(kws['subject'], kws['session'], 'stop_time', convert=False)
    return process_licks(
        [lefts, rights],
        starts,
        stops,
        time_dim=hparams.time_dim,
    )

def collect_behavior(group, hparams, **kwargs):
    kws = {'subject': '440959', 'session': 3, 'target': 'tongue'}
    kws.update(kwargs)
    return get_behavior(kws['subject'], kws['session'], kws['target'])