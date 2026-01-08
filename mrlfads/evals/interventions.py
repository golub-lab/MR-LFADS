"""
Utilities for performing ablation analysis.

Main functions
---------------------
ablate_by_channels:
    Ablate selected generator input channels and measure rate impact.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def ablate_by_channels(
    model,
    area_name: str,
    channels: list,
    t_mask: int = 100,
    t_dur: int = 50,
):
    """Ablate generator input channels over a time window and measure rate impact.

    Runs the area generator in two configurations:
        (1) With the original generator input,
        (2) With specified input channels zeroed during `[t_mask, t_mask + t_dur)`.
    Returns the unablated hidden trajectory, masked and unmasked readout rates,
    and their absolute difference.

    Args:
        model: `MRLFADS` class.
        channels: Generator input indices to zero.
        t_mask: Start time index of the ablation window.
        t_dur: Duration of the ablation window.

    Returns:
        h_true_arr: Unablated generator hidden states.
        r_mask_arr: Readout rates from masked hidden states.
        d_mask_arr: Absolute difference `|r_true - r_mask|`.
        r_true_arr: Readout rates from unablated hidden states.

    Notes:
        - `h_true_arr` is equal to model.save_var[area_name].states[:, 1:].
    """

    name_to_idx = {name: idx for idx, name in enumerate(model.area_names)}

    # Get area components: generator, linear layers, hyperparameters
    area = model.areas[area_name]
    gen = area.decoder.gen_cell
    lin = area.decoder.gen_map
    readout = area.readout[0].to(model.device) 
    hps = model.hparams
    ahps = model.areas[area_name].hparams
    time = hps.seq_len - hps.ic_enc_seq_len # total time to unroll over
    try: # deprecated feature
        scalar = area.decoder.gv_scalar
    except AttributeError:
        scalar = torch.ones(1).to(model.device)

    # Get source areas, i.e. all areas except the current one
    other_area_names = deepcopy(model.area_names)
    other_area_names.remove(area_name)

    # Get saved hidden states
    states = model.save_var[area_name].states[..., ahps.con_size:-ahps.fac_dim].to(model.device)
    batch = len(states)

    # Define storage array
    h_true_arr = np.zeros((batch, time, ahps.gen_size))
    r_true_arr = np.zeros((batch, time, ahps.num_neurons[0]))
    r_mask_arr = np.zeros((batch, time, ahps.num_neurons[0]))
    d_mask_arr = np.zeros((batch, time, ahps.num_neurons[0]))

    # Re organize generator input into inferred then communication, then external inputs
    # (Because it is saved as communication first then inferred input, opposite of how the generator receives it)
    gen_inp = model.save_var[area_name].inputs[..., ahps.ci_size:].to(model.device)
    ext_inp = model.save_var[area_name].ext_inputs.to(model.device)
    ext_inp = scalar.reshape(1, 1, -1) * ext_inp
    gen_inp = torch.cat([gen_inp[..., -ahps.co_dim:], gen_inp[..., :-ahps.co_dim], ext_inp], dim=-1) # inferred input, then communication

    # Mask generator input by channel index during a time interval
    gen_inp_mask = gen_inp.clone()
    gen_inp_mask[:, t_mask:t_mask+t_dur][..., channels] = torch.zeros_like(gen_inp[:, t_mask:t_mask+t_dur][..., channels]).to(model.device)

    # Main loop
    state = states[:, 0]
    for t in range(0, time):

        # Put hidden states, masked input into the generator
        h_true = gen(gen_inp[:, t], state)
        h_mask = gen(gen_inp_mask[:, t], state)

        # Get corresponding rates by putting generator hidden states through the linear layers
        r_true = readout(lin(h_true)).cpu().detach().numpy()
        r_mask = readout(lin(h_mask)).cpu().detach().numpy()

        # Normalize the difference between the true rates and ablated rates
        cut = ahps.num_neurons[0]
        h_true_arr[:, t] = h_true.cpu().detach().numpy()
        r_true_arr[:, t] = r_true[..., :cut]
        r_mask_arr[:, t] = r_mask[..., :cut]
        d_mask_arr[:, t] = np.abs(r_true[..., :cut] - r_mask[..., :cut])
        state = h_true

    return h_true_arr, r_mask_arr, d_mask_arr, r_true_arr