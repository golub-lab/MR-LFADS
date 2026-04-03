import os
import re
import torch
import fnmatch
import pickle
import numpy as np
import torch.nn.functional as F

from collections.abc import Mapping
from collections import namedtuple 
from dataclasses import dataclass

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

def extract_numbers_after_equal(string):
    lists = re.split('[,_]', string)
    lists = [float(elem.split("=")[1]) for elem in lists if "=" in elem]
    return lists

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

def det(x):
    return x.detach() if torch.is_tensor(x) else x

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

def flatten(arr): return arr.reshape(-1, arr.shape[-1])

def mkfile(*args):
    os.makedirs(os.path.join(*args), exist_ok=True)

def npsave(item, *args):
    if len(args) > 1: mkfile(*args[:-1])
    np.save(os.path.join(*args), item)

def npload(*args): return np.load(os.path.join(*args))

def pklsave(item, *args):
    if len(args) > 1: mkfile(*args[:-1])
    f = open(os.path.join(*args), "wb")
    pickle.dump(item, f)
    f.close()

def pklload(*args):
    f = open(os.path.join(*args), "rb")
    data = pickle.load(f)
    f.close()
    return data

def write_logfile(f, message):
    with open(f, 'a') as logfile: 
        logfile.write(message + '\n')

def find_files(root_dir, pattern):
    matches = []

    for current_dir, _, files in os.walk(root_dir):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                full_path = os.path.join(current_dir, file)
                matches.append(full_path)

    return matches

def find_directories(base_path, *patterns):
    matching_directories = []

    for root, dirs, _ in os.walk(base_path):
        for directory in dirs:
            if all(pattern in directory for pattern in patterns):
                matching_directories.append(os.path.join(root, directory))

    return matching_directories