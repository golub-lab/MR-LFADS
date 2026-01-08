import os
import re
import math
from pathlib import Path
from typing import Dict, Any, Optional
from collections.abc import Mapping, Sequence

import torch
import numpy as np
import torch.nn.functional as F
import mrlfads.paths as path
from torch import nn
from collections import namedtuple 
from dataclasses import dataclass
from distutils.dir_util import copy_tree

class HParams:
    def __init__(self, hparams):
        for key, value in hparams.items(): setattr(self, key, value)
        
    def add(self, key, value): setattr(self, key, value)

class HoldoutNeuron:
    def __init__(self, hparams):
        self.hn_dict = {}
        self.hn_mask = {}
        self.hparams = hparams

    def preprocess(self, batch):
        encod_data, ext_inp = batch
        
        # If holdout neurons are not implemented
        if len(self.hparams.hn_indices) == 0:
            for area_name, arr in encod_data.items():
                batch_size, time_size, _ = arr.shape
                self.hn_dict[area_name] = np.zeros(batch_size, time_size, 0)
                self.hn_mask[area_name] = torch.from_numpy(np.zeros(0))
            return Batch(encod_data, ext_inp)
        
        encod_data_modified = {}
        for area_name, arr in encod_data.items():
            hn_mask = torch.tensor(self.hparams.hn_indices[area_name])[0] # uses the 1st session
            
            # Store heldout neurons, replace with zeros
            if len(hn_mask) > 0:
                hn_arr = arr[..., hn_mask]
                arr[..., hn_mask] = torch.zeros_like(hn_arr)
            else:
                hn_arr = torch.zeros(0)
                
            encod_data_modified[area_name] = arr
            self.hn_dict[area_name] = hn_arr
            self.hn_mask[area_name] = hn_mask
            
        return Batch(encod_data_modified, ext_inp)
    
    def postprocess_holdin(self, recon_loss, area_name):
        # If holdout neurons are not implemented
        if len(self.hn_mask[area_name]) == 0: return recon_loss
    
        hn_mask = self.hn_mask[area_name]
        idxs = np.arange(recon_loss.shape[-1])
        holdin_idxs = np.setdiff1d(idxs, hn_mask.cpu().detach().numpy())
        return recon_loss[..., holdin_idxs]
    
    def postprocess_holdout(self, area_name, pred, compute_loss_func):
        # If holdout neurons are not implemented
        if len(self.hn_mask[area_name]) == 0: return 0
    
        truth = self.hn_dict[area_name][:, self.hparams.ic_enc_seq_len:]
        return compute_loss_func(truth, pred)

Batch = namedtuple(
    "Batch",
    [
        "encod_data",
        "ext_input",
    ],
)
    
@dataclass
class SaveVariables:
    states: torch.Tensor = torch.empty(0)
    inputs: torch.Tensor = torch.empty(0)
    ext_inputs: torch.Tensor = torch.empty(0)
    ic_params: torch.Tensor = torch.empty(0)
    co_params: torch.Tensor = torch.empty(0)
    com_params: torch.Tensor = torch.empty(0)
    gv_params: torch.Tensor = torch.empty(0)
    
def replace_hps_str(string):
    hp, val = string.split("=")
    return hp.replace(".", "_") + "=" + val

def flatten_params(source):
    flat = {}
    stack = [((), source)]

    while stack:
        path, current = stack.pop()
        for k, v in current.items():
            new_path = path + (k,)
            if isinstance(v, dict):
                stack.append((new_path, v))
            else:
                flat[".".join(new_path)] = v
    return flat

def find_directories(base_path, *patterns):
    matching_directories = []

    for root, dirs, files in os.walk(base_path):
        for directory in dirs:
            if all(pattern in directory for pattern in patterns):
                matching_directories.append(os.path.join(root, directory))

    return matching_directories

def extract_numbers_after_equal(string):
    lists = re.split('[,_]', string)
    lists = [float(elem.split("=")[1]) for elem in lists if "=" in elem]
    return lists

def relocate(folder, dest="scratch"): # Copy config to results to avoid read-only issues specific to CodeOcean
    CUR_DIR = os.getcwd()
    copy_tree(os.path.join(path.datapath, folder), os.path.join(path.homepath, dest, folder))
    RUN_DIR = Path(os.path.join(path.homepath, dest)) / project_str
    return RUN_DIR

def _dir_matches_overrides(dir_name: str, overrides: Dict[str, Any], tol: float = 1e-8):
    """
    Check if a Ray Tune-style directory name matches all overrides.
    Keys in overrides may contain '.' but are stored as '_' in the dir name.
    Floats like 0.1 vs 0.1000 are treated as equal.
    """
    for key, value in overrides.items():
        # Ray-style dir key ('.' -> '_')
        key_folder = str(key).replace(".", "_")

        # Capture the value:   ...<boundary>key=value...
        # boundary = not an alphanumeric char (so '_' is allowed before key)
        pattern = rf"(?<![A-Za-z0-9]){re.escape(key_folder)}=([A-Za-z0-9.\-]+)"
        m = re.search(pattern, dir_name)
        if not m:
            return False

        val_str = m.group(1)

        # Try numeric comparison first (handles 0.1 vs 0.1000)
        try:
            v_float = float(value)
            folder_float = float(val_str)
        except (ValueError, TypeError):
            # Not both floats -> fall back to string equality
            if str(value) != val_str:
                return False
        else:
            if math.isfinite(v_float) and math.isfinite(folder_float):
                if abs(v_float - folder_float) > tol:
                    return False
            else:
                if str(value) != val_str:
                    return False

    return True

def deep_clone_tensors(x):
    # Base case
    if torch.is_tensor(x): return x.clone()

    # Copy for containers
    if isinstance(x, Mapping):
        return {k: deep_clone_tensors(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(deep_clone_tensors(v) for v in x)
    if isinstance(x, list):
        return [deep_clone_tensors(v) for v in x]
    return x

def apply_along_axis(arr, func):
    return [func(item) for item in arr]

def convert_byte(byte_string):
    decoded_str = byte_string.decode('utf-8')
    
    try:
        return eval(decoded_str)
    except (SyntaxError, ValueError, NameError):
        pass

    try:
        return float(decoded_str)
    except ValueError:
        pass
    
    return decoded_str