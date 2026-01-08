import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

def ablate_by_channels(model, area_name, channels, t_mask=100, t_dur=50):
    """Ablates messages based on channel index.
    
    Note: Have verified that h_true_arr == states[:, 1:]
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
    try:
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