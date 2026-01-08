"""
Utilities for evaluating model fit and inferred connectivity.

Main functions
---------------------
fit_metrics:
    Aggregate reconstruction and other metrics into a summary dictionary.

cosine_similarity:
    Compute cosine similarity between model-inferred connectivity and 
    their ground-truth counterparts.
"""

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

def fit_metrics(model, metrics):
    """Aggregate reconstruction and fit metrics into a summary dictionary.

    Extracts total/reconstruction/holdout loss, normalized KL terms,
    and R² across areas for held-in and held-out neurons.

    Args:
        model: `MRLFADS` class.
        metrics: Training/validation metrics dictionary.

    Returns:
        res: Dict containing scalar loss components and mean R² metrics.
    """

    hps = model.hparams
    r2, r2_holdout = r_squared_rel_null(model) # calculates r-squared values
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

    Compares model-inferred communication and inferred-input effectomes against
    ground-truth counterparts using cosine similarity. Diagonal (self-to-self)
    communication terms are excluded.

    Args:
        model: `MRLFADS` class.
        true_m: Ground-truth communication connectivity matrix.
        true_u: Ground-truth inferred-input effectome.

    Returns:
        sim_m: Cosine similarity between inferred and ground-truth communication
            connectivity (excluding diagonal terms).
        sim_u: Cosine similarity between inferred and ground-truth inferred-input
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

def r_squared_rel_null(model):
    """Compute R² metrics relative to a null model for each area.

    For each area, computes standard and McFadden R² for held-in and held-out neurons.

    Args:
        model: `MRLFADS` class.

    Returns:
        r2: Dict mapping area name to (r2_standard, r2_mcfadden) for held-in neurons.
        r2_holdout: Dict mapping area name to (r2_standard, r2_mcfadden) for held-out neurons.
    """
    r2, r2_holdout = {}, {}
    for area_name in model.area_names:
        ic_enc_seq_len = model.hparams.ic_enc_seq_len
        data = model.raw_batch[0].encod_data[area_name][:, ic_enc_seq_len:]
        hn_idx = model.hparams.hn_indices[area_name][0]
        
        # R-squared values for heldin neurons
        r2_std = model.areas[area_name].output_dist.rsquared(data, model.outputs[area_name][0], mode='standard')
        r2_mcf = model.areas[area_name].output_dist.rsquared(data, model.outputs[area_name][0], mode='mcfadden')
        r2[area_name] = (r2_std, r2_mcf)
        
        # R-squared values for heldout neurons
        r2_std = model.areas[area_name].output_dist.rsquared(data[..., hn_idx], model.preds[area_name][0], mode='standard')
        r2_mcf = model.areas[area_name].output_dist.rsquared(data[..., hn_idx], model.preds[area_name][0], mode='mcfadden')
        r2_holdout[area_name] = (r2_std, r2_mcf)
    return r2, r2_holdout

def volume(model, reduction=None):
    '''Extract communication and inferred-input volumes from a model over batch and time.
    
    Returns per-area communication volumes and inferred-input volumes. If `reduction`
    is provided, computes L2 norms over the specified axes instead of returning the
    raw volumes.
    
    Args:
        model: `MRLFADS` class.
        reduction: Optional iterable of axis indices to reduce over when computing
            L2 norms. If provided, the function returns L2 norms over those axes
            for both volumes. If None, returns raw volumes. Defaults to None.
            
    Returns:
        comm: Communication volume with shape
            (num_target_areas, num_source_areas, batch, time, com_dim), or the
            reduced L2 norm array if `reduction` is provided.
        inputs: Inferred-input volume with shape
            (num_areas, batch, time, co_dim), or the reduced L2 norm array if
            `reduction` is provided.
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

