import os
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from copy import deepcopy
from scipy.spatial.distance import cosine, jensenshannon
from mrlfads.evals.utils import PolyRegression

def r_squared_rel_null(model):
    r2, r2_holdout = {}, {}
    for area_name in model.area_names:
        ic_enc_seq_len = model.hparams.ic_enc_seq_len
        data = model.current_batch[0].encod_data[area_name][:, ic_enc_seq_len:]
        r2_std = model.areas[area_name].output_dist.rsquared(data, model.outputs[area_name][0])
        r2_mcf = model.areas[area_name].output_dist.rsquared(data, model.outputs[area_name][0], mode='mcfadden')
        r2[area_name] = (r2_std, r2_mcf)
        
        data = model.raw_batch[0].encod_data[area_name][:, ic_enc_seq_len:]
        hn_idx = model.hparams.hn_indices[area_name][0]
        r2_std = model.areas[area_name].output_dist.rsquared(data[..., hn_idx], model.preds[area_name][0])
        r2_mcf = model.areas[area_name].output_dist.rsquared(data[..., hn_idx], model.preds[area_name][0], mode='mcfadden')
        r2_holdout[area_name] = (r2_std, r2_mcf)
    return r2, r2_holdout

def fit_metrics(model, metrics, hps):
    """Extracts reconstruction/fit related metrics."""
    r2, r2_holdout = r_squared_rel_null(model)
    res = {
        'total': (metrics['valid/recon'] + metrics['valid/kl/u'] * hps.kl_co_scale + metrics['valid/kl/m'] * hps.kl_com_scale).item(),
        'recon': metrics['valid/recon'].item(),
        'hn': metrics['valid/hn'].item(),
        'kl(u)': metrics['valid/kl/u'].item() / hps.kl_co_scale,
        'kl(m)': metrics['valid/kl/m'].item() / hps.kl_com_scale,
        'r2(standard)': np.mean([val[0] for val in r2.values()]),
        'r2(standard; holdout)': np.mean([val[0] for val in r2_holdout.values()]),
        'r2(mcfadden)': np.mean([val[1] for val in r2.values()]),
        'r2(mcfadden; holdout)': np.mean([val[1] for val in r2_holdout.values()]),
    }
    return res

def cosine_similarity(model, true_m, true_u):
    """Compute cosine similarity between inferred and ground-truth connectivity.

    This function compares the model-inferred communication and inferred-input
    effectomes against ground-truth counterparts using cosine similarity.
    Diagonal (self-to-self) communication terms are excluded from the comparison.

    Args:
        model: Trained model instance used to compute inferred volumes.
        true_m: Ground-truth communication connectivity matrix.
        true_u: Ground-truth inferred-input effectome.

    Returns:
        Tuple containing:
            - Cosine similarity between inferred and ground-truth communication
              connectivity (excluding diagonal terms).
            - Cosine similarity between inferred and ground-truth input
              effectomes.
    """
    pred_m, pred_u = volume(model, reduction=[-1, -2, -3])
    pred_m_flat, true_flat = [], []
    
    # Exclude diagonal terms
    for i in range(len(pred_u)):
        for j in range(len(pred_u)):
            if i != j:
                pred_m_flat.append(pred_m[i, j])
                true_flat.append(true_m[i, j])
                
    return 1-cosine(pred_m_flat, true_flat), 1-cosine(pred_u, true_u)

def volume(model, reduction=None):
    '''Extract communication and inferred-input volumes from a model over batch and time.
    
    Args:
        reduction: Optional iterable of axis indices to reduce over when computing
            L2 norms. If provided, the function returns L2 norms over those axes
            for both volumes. If None, returns raw volumes. Defaults to None.
            
    Returns:
        Tuple `(comm, inputs)` where:
            - comm: Communication volume with shape
              (num_target_areas, num_source_areas, batch, time, com_dim) if
              `reduction` is None, otherwise the reduced L2 norm array.
            - inputs: Inferred-input volume with shape
              (num_areas, batch, time, co_dim) if `reduction` is None, otherwise
              the reduced L2 norm array.
    '''
    hps = model.hparams
    num_areas = hps.num_other_areas + 1
    batch, time = model.save_var[model.area_names[0]].inputs.shape[:2]
    
    # Note: this function only works when all communication channels have equal dimensions
    assert len(list(set([model.areas[model.area_names[i]].hparams.com_dim for i in range(num_areas)]))) == 1
    co_dim = model.areas[model.area_names[0]].hparams.co_dim
    com_dim = model.areas[model.area_names[0]].hparams.com_dim
    
    # Locate and append communication, inferred inputs
    volume, uvolume = np.zeros((num_areas, num_areas, batch, time, com_dim)), np.zeros((num_areas, batch, time, co_dim))
    for ia, (area_name, area) in enumerate(model.areas.items()):
        ahps = area.hparams
        num_other_areas_name = list(model.areas.keys())
        num_other_areas_name.pop(ia)

        inputs = model.save_var[area_name].inputs.detach().cpu()
        ci_size, com_dim, co_dim = ahps.ci_size, ahps.com_dim, ahps.co_dim
        _, com, co = torch.split(inputs, [ci_size, com_dim * hps.num_other_areas, co_dim], dim=2)

        for ioa in range(hps.num_other_areas):
            idx = ioa + 1 if ioa >= ia else ioa
            volume[ia, idx] = com[..., com_dim * ioa: com_dim * (ioa + 1)]
        uvolume[ia] = co

    # Reduce dimensions of volumes
    if reduction is not None: 
        axes = tuple(reduction)
        return np.sqrt(np.sum(np.square(volume), axis=axes)), np.sqrt(np.sum(np.square(uvolume), axis=axes))
    else: return volume, uvolume

def r_squared(
    model,
    dim_map_m = None,
    dim_map_u = None,
    var_name = 'inp',
    transform = None,
    delay = None,
):
    mscores = r_squared_m(model, dim_map_m, var_name, transform, delay) if isinstance(dim_map_m, dict) else None
    uscores = r_squared_u(model, dim_map_u, var_name, transform, delay) if isinstance(dim_map_u, dict) else None
    return mscores, uscores

def r_squared_m(
    model,
    dim_map: dict,
    var_name = 'inp',
    transform: callable = None,
    delay: int = None,
):
    hps = model.hparams
    inps = model.current_info[0][var_name][:, hps.ic_enc_seq_len:].cpu().detach().numpy()
    if not isinstance(transform, type(None)): inps = transform(inps)
    plag = delay
    nlag = -delay if delay != None else None 
    
    na = hps.num_other_areas + 1
    scores = np.zeros((na, na))
    
    reg = PolyRegression(1)
    for ia, area_name in enumerate(model.area_names): # target area
        
        ahps = model.areas[area_name].hparams
        msgs = model.save_var[area_name].inputs[..., ahps.ci_size:-ahps.co_dim].cpu().detach().numpy()
        msg_split = np.split(msgs, na-1, axis=-1)
        
        for ioa, other_area_name in enumerate(model.area_names): # source area
            
            if ia == ioa: continue # no self recurrence
            assert (other_area_name, area_name) in dim_map.keys(), 'Missing keys in dimension map!'
            
            dims = dim_map[(other_area_name, area_name)]
            if isinstance(dims, type(None)): continue # if none, the message doesn't exist in ground truth
            
            true = inps[..., dims][:, :-1][..., np.newaxis] # true signal
            pred = msg_split.pop(0)[:, 1:] # predicted signal
            reg.ffit(pred[:, :nlag], true[:, plag:])
            scores[ia, ioa] = reg.fscore(pred, true)
            
    return scores
            
def r_squared_u(
    model,
    dim_map: dict,
    var_name = 'inp',
    transform: callable = None,
    delay: int = None,
):
    hps = model.hparams
    inps = model.current_info[0][var_name][:, hps.ic_enc_seq_len:].cpu().detach().numpy()
    if not isinstance(transform, type(None)): inps = transform(inps)
    scores = np.zeros(hps.num_other_areas + 1)
    
    reg = PolyRegression(1)
    for ia, area_name in enumerate(model.area_names):
        
        ahps = model.areas[area_name].hparams
        pred = model.save_var[area_name].inputs[..., -ahps.co_dim:].cpu().detach().numpy()
        
        assert area_name in dim_map.keys(), 'Missing keys in dimension map!'    
        dims = dim_map[area_name]
        if isinstance(dims, type(None)): continue # if none, the message doesn't exist in ground truth
            
        true = inps[..., dims][..., np.newaxis] # true signal
        reg.ffit(pred, true)
        scores[ia] = reg.fscore(pred, true)
            
    return scores         
            
            