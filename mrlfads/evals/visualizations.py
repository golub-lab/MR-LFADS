import os
import numpy as np
import torch
import torch.nn
import seaborn as sns
import matplotlib.pyplot as plt
import tools.vistools as vis
from scipy.ndimage import gaussian_filter1d

def plot_reconstruction(model, indices=None, b=0, color='b', smooth=False, end=75):
    hps = model.hparams
    ic_enc_seq_len = hps.ic_enc_seq_len
    
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
        true = model.current_batch[0].encod_data[area_name][:, ic_enc_seq_len:].cpu().detach().numpy()
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
            
    vis.common_col_title(fig, model.area_names, axs.shape)
    vis.common_row_ylabel(fig, [f'Neuron {nn}' for nn in indices], axs.shape)