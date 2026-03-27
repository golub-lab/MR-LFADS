"""
Utilities for visualizing model inferences.

Main functions
--------------
plot_reconstruction:
    Plot reconstructed and ground-truth activity traces for selected neurons
    across model areas.
"""

import io
import os
import numpy as np
import torch
import torch.nn
import seaborn as sns
import networkx as nx
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

import mrlfads.utils.visualization_utils as vis

def plot_reconstruction(model, indices=None, b=0, color='b', smooth=False, end=-1):
    """Plot reconstructed and ground-truth activity traces for selected neurons.

    Args:
        model: `MRLFADS` class.
        indices: Optional neuron indices to plot. May be:
            - None: plot the first five neurons in each area,
            - list: use the same neuron indices for all areas,
            - dict: map area name to a list of neuron indices.
        b: Batch index to visualize.
        color: Line color for reconstructed traces.
        smooth: If True, apply temporal smoothing to the true traces.
        end: Final time index (exclusive) to plot.
    """
    
    hps = model.hparams
    ic_enc_seq_len = hps.ic_enc_seq_len
    
    # Neuron indices to plot
    if isinstance(indices, type(None)):
        indices = np.zeros(5)
        indices_dict = {an: np.arange(5).astype(int) for an in model.area_names}
    elif isinstance(indices, list):
        indices_dict = {an: indices for an in model.area_names}
    elif isinstance(indices, dict):
        indices_dict = indices
    
    num_rows = len(indices)
    num_cols = len(model.areas)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 2 * num_rows), squeeze=False)
    for ia, (area_name, area) in enumerate(model.areas.items()):
        true = model.raw_batch[0].encod_data[area_name][:, ic_enc_seq_len:].cpu().detach().numpy()
        recon = model.outputs[area_name][0].cpu().detach().numpy()
        
        # Transform and smoothing
        transform = area.output_dist.mean
        if smooth:
            smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        else:
            smoothing_func = lambda x: x
            
        indices = indices_dict[area_name]
        for j, nn in enumerate(indices):
            true_processed = smoothing_func(true[b, :end, nn])
            recon_processed = transform(torch.from_numpy(recon[b, :end, nn]))
            axs[j, ia].plot(true_processed, color='k')
            axs[j, ia].plot(recon_processed, color=color, linestyle='--')
            vis.set_invisible(axs[j, ia])
            axs[j, ia].set_ylabel(f'Neuron {nn}')
            
    vis.common_col_title(fig, model.area_names, axs.shape)

def plot_anatomy(model, ax=None, save=None, last=None, co_scale=1):   
    kl_weight_dict = {}
    num_areas = len(model.areas)
    if num_areas < 2: return
    
    model.to("cuda")
    for area_name, area in model.areas.items():
        if area.hparams.use_con:
            co_mean, co_std = torch.split(model.save_var[area_name].co_params, area.hparams.co_dim, dim=2)
            co_kl = np.log(area.co_prior(co_mean, co_std).item()) * co_scale # scale is already multiplied
        else:
            co_kl = 0
            
        com_mean, com_std = torch.split(model.save_var[area_name].com_params, area.hparams.com_dim * (num_areas-1), dim=2)
        com_kl = np.log(np.array(area.com_prior.kl_divergence_by_component(com_mean, com_std, area.hparams.com_dim, tpe='mean')))

        other_areas = model.area_names.copy()
        other_areas.remove(area_name)
        for io, other_area in enumerate(other_areas):
            kl_weight_dict[(other_area[:last], area_name[:last])] = com_kl[io]
            
        # Inferred inputs
        kl_weight_dict[("Other", area_name[:last])] = co_kl
        
        # Global area
        if model.hparams.gv_dim > 0:
            gv_mean, gv_std = torch.split(model.save_var[area_name].gv_params, model.hparams.gv_dim, dim=2)
            gv_kl = np.log(model.global_area.gv_prior(gv_mean, gv_std).item())
            kl_weight_dict[("Global", area_name[:last])] = gv_kl
        
    # Create Digraph
    color_choice = ["black", "red", "green"]
    G = nx.DiGraph()
    for edge, weight in kl_weight_dict.items():
        G.add_edge(edge[0], edge[1], weight=weight)
    pos = nx.circular_layout(G)
    edges = G.edges()
    weights = np.array([kl_weight_dict[edge] for edge in edges])
    colors = np.array([color_choice[
        int(("Other" in edge) or ("Global" in edge)) * (int("Global" in edge) + 1)
    ] for edge in edges])
    
    # Normalize and scale weights
    weights -= np.min(weights)
    weights /= np.max(weights)
    weights += 0.1
    weights *= 5

    if not ax: fig, ax = plt.subplots(1, 1)
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue', font_weight='bold', connectionstyle='arc3, rad = 0.1', width=weights, edge_color=colors, ax=ax)
    plt.tight_layout()
    
    if save: plt.savefig(f"/root/capsule/results/{save}.png")
    return kl_weight_dict

def plot_holdout_reconstruction(model, s=0, b=0, color='b', smooth=False, end=-1):
    hps = model.hparams
    ic_enc_seq_len = hps.ic_enc_seq_len
    
    idx_len = []
    for area_name in model.area_names:
        idx_len.append( len(model.hparams.hn_indices[area_name][s]) )
    
    num_rows = max(idx_len)
    num_cols = len(model.areas)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 2 * num_rows), squeeze=False)
    for ia, (area_name, area) in enumerate(model.areas.items()):
        true = model.raw_batch[0].encod_data[area_name][:, ic_enc_seq_len:].cpu().detach().numpy()
        recon = model.preds[area_name][0].cpu().detach().numpy()
        
        # Transform and smoothing
        transform = area.output_dist.mean
        if smooth:
            smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        else:
            smoothing_func = lambda x: x
            
        indices = model.hparams.hn_indices[area_name][s]
        for j, nn in enumerate(indices):
            true_processed = smoothing_func(true[b, :end, nn])
            recon_processed = transform(torch.from_numpy(recon[b, :end, j]))
            axs[j, ia].plot(true_processed, color='k')
            axs[j, ia].plot(recon_processed, color=color, linestyle='--')
            vis.set_invisible(axs[j, ia])
            axs[j, ia].set_ylabel(f'Neuron {nn}')
            
    vis.common_col_title(fig, model.area_names, axs.shape)