import os
import re
import math
import time
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from torch import nn
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections.abc import Mapping, Sequence
from collections import namedtuple 
from dataclasses import dataclass
from distutils.dir_util import copy_tree
from omegaconf import DictConfig

import mrlfads.paths as path


# ===== Classes ===== #
class HoldoutNeuron:
    def __init__(self, hparams):
        self.hn_dict = {} # session --> area_name --> store heldout neuron spike counts
        self.hn_mask = {} # session --> area_name --> store heldout neuron indices
        self.hparams = hparams

    def mask_data(self, batch, s):
        """
        Preprocess Batch data to mask out holdout neurons.
        """
        encod_data, ext_inp = batch
        self.hn_dict[s] = {}
        self.hn_mask[s] = {}
        
        # If holdout neurons are not implemented for all areas
        # No operation performed, return original batch
        if len(self.hparams.hn_indices) == 0:
            for area_name, arr in encod_data.items():
                batch_size, time_size, _ = arr.shape
                self.hn_dict[s][area_name] = np.zeros((batch_size, time_size, 0))
                self.hn_mask[s][area_name] = torch.from_numpy(np.zeros(0))
            return Batch(encod_data, ext_inp)
        
        # If there are holdout neurons, modify data in-place
        encod_data_modified = {}
        for area_name, arr in encod_data.items():
            hn_mask = torch.tensor(self.hparams.hn_indices[area_name][s])
            
            # Store heldout neurons, replace with zeros
            if len(hn_mask) > 0:
                hn_arr = arr[..., hn_mask]
                arr[..., hn_mask] = torch.zeros_like(hn_arr)
            else:
                batch_size, time_size, _ = arr.shape
                hn_arr = arr.new_zeros(batch_size, time_size, 0)
                
            encod_data_modified[area_name] = arr
            self.hn_dict[s][area_name] = hn_arr
            self.hn_mask[s][area_name] = hn_mask
            
        return Batch(encod_data_modified, ext_inp)
    
    def mask_holdout(self, recon_loss, area_name, s):
        """
        Postprocess reconstruction loss for holdin neurons per area per session.
        """
        # If holdout neurons are not implemented
        # No operation performed, return original loss
        if len(self.hn_mask[s][area_name]) == 0: return recon_loss
    
        # Remove (not zero out) holdout neuron reconstruction loss
        hn_mask = self.hn_mask[s][area_name]
        idxs = np.arange(recon_loss.shape[-1]) # idxs = all neuron indices
        holdin_idxs = np.setdiff1d(idxs, hn_mask.cpu().detach().numpy())
        return recon_loss[..., holdin_idxs]
    
    def compute_holdout(self, area_name, pred, compute_loss_func, s):
        # If holdout neurons are not implemented
        # Return nothing
        if len(self.hn_mask[s][area_name]) == 0:
            return pred.new_zeros(*pred.shape[:-1], 0)
    
        # Retrieve holdout neuron spike counts from this class to compute loss
        truth = self.hn_dict[s][area_name][:, self.hparams.ic_enc_seq_len:]
        return compute_loss_func(truth, pred)

class HParams:
    def __init__(self, hparams):
        for key, value in hparams.items(): setattr(self, key, value)
        
    def add(self, key, value): setattr(self, key, value)
    
    def update(self, dic):
        for key, value in dic.items(): self.add(key, value)
        
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
            
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

    
# ===== Helper Functions ===== #
    
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

def dir_matches_overrides(dir_name: str, overrides: Dict[str, Any], tol: float = 1e-8):
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

def get_insert_func(sizes, return_slice=False):
    data_ends = np.cumsum(sizes)
    data_starts = np.insert(data_ends, 0, 0)[:-1]
    
    def insert_tensor(tensor, data, index):
        start, end = data_starts[index], data_ends[index]
        tensor[..., start:end] = data.clone()

    def exclude_tensor(tensor, index):
        start, end = data_starts[index], data_ends[index]
        indices_to_include = torch.tensor(list(range(start)) + list(range(end, data_ends[-1]))).to(torch.int64)
        sliced_tensor = torch.index_select(tensor, dim=-1, index=indices_to_include.to(tensor.device))
        return sliced_tensor
    
    def slice_tensor(tensor, index):
        start, end = data_starts[index], data_ends[index]
        indices_to_include = torch.tensor(list(range(start, end))).to(torch.int64)
        sliced_tensor = torch.index_select(tensor, dim=-1, index=indices_to_include.to(tensor.device))
        return sliced_tensor

    if not return_slice:
        return insert_tensor, exclude_tensor
    else:
        return insert_tensor, exclude_tensor, slice_tensor

def pad_by_index(x, counts, n_params):
    
    def inner(x, idx):
        left = sum(counts[:idx]) * n_params
        right = sum(counts[idx+1:]) * n_params
        return F.pad(x, (left, right), mode='constant', value=0)
    
    res = [inner(x[idx], idx) for idx in range(len(counts))]
    res = torch.cat(res, dim=0) # join across batch dimension
    
    # Re-order because now last dim is (s0_mean, s0_std, s1_mean, s1_std, ...)
    n_sess = len(counts)
    tt_counts = np.array([[count] * n_params for count in counts]).flatten()
    res_split = torch.split(res, tuple(tt_counts), dim=-1)
    
    order = np.arange(n_sess * n_params).reshape(n_sess, n_params).flatten('F')
    return torch.cat([res_split[od] for od in order], dim=-1)

def batch_smoothing_func(x):
    smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
    return np.apply_along_axis(smoothing_func, axis=1, arr=x)

def flatten(arr): return arr.reshape(-1, arr.shape[-1])

class PolyRegression:
    def __init__(self, degree, alpha=0.0, tpe="ridge"):
        self.degree = degree
        self.alpha = alpha
        self.poly_features = PolynomialFeatures(degree=degree)
        
        if tpe == "ridge":
            self.reg = Ridge(alpha=alpha)
        elif tpe == "lasso":
            self.reg = Lasso(alpha=alpha)
        else:
            raise ValueError()

    def fit(self, X, y):
        X_poly = self.poly_features.fit_transform(X)
        self.reg.fit(X_poly, y)
        
    def ffit(self, X, y):
        self.fit(flatten(X), flatten(y))

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return self.reg.predict(X_poly)
    
    def fpredict(self, X):
        X_poly = self.poly_features.transform(flatten(X))
        return self.reg.predict(X_poly).reshape(*X.shape[:2], -1)

    def score(self, X, y): # X: prediction, y: true
        X_poly = self.poly_features.transform(X)
        return self.reg.score(X_poly, y)
    
    def fscore(self, X, y):
        pred = self.fpredict(X)
        return r2_score(flatten(y), flatten(pred))
    
