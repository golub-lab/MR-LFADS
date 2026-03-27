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
    data_type: str = 'rates',
    compound: bool = False,
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
        compound: Whether to compound ablation effects.

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
    gen_inp = torch.cat([gen_inp[..., -ahps.co_dim:], gen_inp[..., :-ahps.co_dim], ext_inp], dim=-1) # inferred input, then communication, then external inputs

    # Mask generator input by channel index during a time interval
    gen_inp_mask = gen_inp.clone()
    gen_inp_mask[:, t_mask:t_mask+t_dur][..., channels] = torch.zeros_like(gen_inp[:, t_mask:t_mask+t_dur][..., channels]).to(model.device)

    # Main loop
    state_true = state_mask = states[:, 0]
    for t in range(0, time):

        # Put hidden states, masked input into the generator
        h_true = gen(gen_inp[:, t], state_true)
        h_mask = gen(gen_inp_mask[:, t], state_mask)

        # Get corresponding rates by putting generator hidden states through the linear layers
        if data_type == 'rates':
            r_true = readout(lin(h_true)).cpu().detach().numpy()
            r_mask = readout(lin(h_mask)).cpu().detach().numpy()
        elif data_type == 'factors':
            r_true = lin(h_true).cpu().detach().numpy()
            r_mask = lin(h_mask).cpu().detach().numpy()

        # Normalize the difference between the true rates and ablated rates
        cut = ahps.num_neurons[0]
        h_true_arr[:, t] = h_true.cpu().detach().numpy()
        r_true_arr[:, t] = r_true[..., :cut]
        r_mask_arr[:, t] = r_mask[..., :cut]
        d_mask_arr[:, t] = np.abs(r_true[..., :cut] - r_mask[..., :cut])
        
        if not compound:
            state_true = state_mask = h_true
        else:
            state_true = h_true
            state_mask = h_mask

    return h_true_arr, r_mask_arr, d_mask_arr, r_true_arr

def ablate(model, t_start, ablate_ii_to=[], ablate_comm_from=[]):
    hps = model.hparams
    time = hps.seq_len - hps.ic_enc_seq_len # total time to unroll over
    name_to_idx = {name: idx for idx, name in enumerate(model.area_names)}
    
    # ===== Save all pre-defined variables first ===== #
    states= {}    # states of all areas
    comps = {}    # componenets required to unroll
    inps = {}     # inputs to all areas
    rctrls = {}   # rate of controls
    rpertbs = {}  # rate of perturbs
    
    for area_name, area in model.areas.items():
        ahps = model.area.hparams
        
        # Get area components: generator, linear layers, hyperparameters
        comps[area_name] = {}
        comps[area_name]["gen"] = area.decoder.gen_cell
        comps[area_name]["fac_lin"] = area.decoder.fac_map # generator to factor
        comps[area_name]["readout"] = area.readout[0].to(model.device) # factor to rates
        comps[area_name]["con"] = area.decoder.con_cell
        comps[area_name]["co_lin"] = area.decoder.co_map
        
        # Get saved hidden states
        states[area_name] = {}
        states[area_name]["gen"] = model.save_var[area_name].states[..., ahps.con_dim:-ahps.fac_dim].to(model.device)
        states[area_name]["fac"] = model.save_var[area_name].states[..., -ahps.fac_dim:].to(model.device)
        states[area_name]["con"] = model.save_var[area_name].states[..., :ahps.con_dim].to(model.device)
        batch = len(states[area_name]["gen"])
        
        # Define storage
        rctrls[area_name] = np.zeros((batch, time, ahps.num_neurons[0]))
        rpertbs[area_name] = np.zeros((batch, time, ahps.num_neurons[0]))
        
        # Re organize generator input into inferred then communication
        # (Because it is saved as communication first then inferred input, opposite of how the generator receives it)
        inps[area_name] = {}
        gen_inp = model.save_var[area_name].inputs[..., ahps.ci_enc_dim:].to(model.device)
        gen_inp = torch.cat([gen_inp[..., -ahps.co_dim:], gen_inp[..., :-ahps.co_dim]], dim=-1) # inferred input, then communication
        inps[area_name]["gen_inp"] = gen_inp
        
        # Mask: comms from parea
        other_area_names = deepcopy(model.area_names)
        other_area_names.remove(area_name)
        gen_ptrb = gen_inp.clone()
        for parea in ablate_comm_from:
            if parea == area_name: continue # No communication from self
            chan_start = ahps.co_dim + other_area_names.index(parea) * ahps.com_dim
            channels = list(range(chan_start, chan_start + ahps.com_dim))
            gen_ptrb[:, t_start:][..., channels] = torch.zeros_like(gen_ptrb[:, t_start:][..., channels])
        inps[area_name]["gen_ptrb"] = gen_ptrb
        
    # ===== Ablate inferred input ===== #
    for parea, _ in areas_to_ablate:
        if parea in ablate_ii_to:
            parea_hps = model.areas[parea].hparams
            gen_ptrb_parea = inps[parea]["gen_ptrb"].clone()
            gen_ptrb_parea[:, t_start:][..., :parea_hps.co_dim] = torch.zeros_like(gen_ptrb_parea[:, t_start:][..., :parea_hps.co_dim])
            inps[parea]["gen_ptrb"] = gen_ptrb_parea
        
    # ===== Process first timestep information ===== #
    # Here, we use t_start instead of t_start - 1 because of the way ``states`` is set up:
    # At time t, ``decoder`` takes states[t] and outputs states[t+1]
    # Also, using either doesn't make too much of a difference
    h_trues, h_ptrbs, c_ptrbs, f_ptrbs = {}, {}, {}, {}
    for ia, (area_name, area) in enumerate(areas_to_ablate):
        h_trues[area_name] = states[area_name]["gen"][:, t_start]
        h_ptrbs[area_name] = states[area_name]["gen"][:, t_start]
        c_ptrbs[area_name] = states[area_name]["con"][:, t_start]
        f_ptrbs[area_name] = states[area_name]["fac"][:, t_start]
        
    # ===== Rates to Messages ===== #
    def to_mesg(rates, tar_area_name, src_area_name):
        ahps = model.areas[tar_area_name].hparams
        comm = model.areas[tar_area_name].communicator.areas_linear[src_area_name]
        pertb_activity = torch.log(torch.clamp(torch.from_numpy(rates), 1e-7)).to(model.device).float()
        pertb_activity = comm(pertb_activity)[:, :ahps.com_dim] # just get mean, not std
        return pertb_activity
    
    # ===== Ablation loop ===== #
    for t in range(t_start, time-1):
        
        for ia, (area_name, area) in enumerate(areas_to_ablate):
            ahps = area.hparams
            
            # Process controller state; compute ii but only write it if this area is NOT ablated
            ci_enc = model.save_var[area_name].inputs[:, t, :ahps.ci_enc_dim].to(model.device)
            f = f_ptrbs[area_name]
            c_ptrb = comps[area_name]["con"](torch.cat([ci_enc, f], axis=-1), c_ptrbs[area_name])

            # compute inferred input but only overwrite gen_ptrb if this area is NOT ablated
            if area_name not in ablate_ii_to:
                ii = comps[area_name]["co_lin"](c_ptrb)
                inps[area_name]["gen_ptrb"][:, t, :ahps.co_dim] = ii[..., :ahps.co_dim]
            # otherwise leave gen_ptrb zeros (ablation)
            c_ptrbs[area_name] = c_ptrb
            
            # Put hidden states, masked input into the generator for inf-step ablation
            # And store it
            h_true = comps[area_name]["gen"](inps[area_name]["gen_inp"][:, t], h_trues[area_name])
            h_ptrb = comps[area_name]["gen"](inps[area_name]["gen_ptrb"][:, t], h_ptrbs[area_name])
            h_trues[area_name] = h_true
            h_ptrbs[area_name] = h_ptrb

            # Get corresponding rates by putting generator hidden states through the linear layers
            f_true = comps[area_name]["fac_lin"](h_true)
            f_ptrb = comps[area_name]["fac_lin"](h_ptrb)
            # f_ptrbs[area_name] = f_ptrb # Depends on full roll-out is desired
            
            cut = ahps.num_neurons[0]
            r_true = np.exp(comps[area_name]["readout"](f_true[..., :cut]).cpu().detach().numpy())
            r_ptrb = np.exp(comps[area_name]["readout"](f_ptrb[..., :cut]).cpu().detach().numpy())

            # Calculate the difference between the true rates and ablated rates
            # Double checked that np.exp(f_true[..., :cut]) matches model.outputs
            rctrls[area_name][:, t] = r_true
            rpertbs[area_name][:, t] = r_ptrb
            
            # Process input and store
            other_area_names = deepcopy(model.area_names)
            if area_name in ablate_comm_from: continue
            if area_name in other_area_names:
                other_area_names.remove(area_name)

            for ioa, oan in enumerate(other_area_names):
                oahps = model.areas[oan].hparams
                m_oan = to_mesg(rpertbs[area_name][:, t], oan, area_name) # message from this to other areas

                # Calculate istart for message from area_name to oan
                copy_area_names = deepcopy(model.area_names)
                copy_area_names.remove(oan)
                i_start = oahps.co_dim + oahps.com_dim * copy_area_names.index(area_name)
                i_end = i_start + oahps.com_dim
                inps[oan]["gen_ptrb"][:, t+1, i_start: i_end] = m_oan
            
    return rctrls, rpertbs, inps