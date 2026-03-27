import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd.functional import jacobian

from mrlfads.utils.torch_utils import ScaledLinear, NormLinear, GRUCellBase, fanin_normal_init_

class SRDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hps = hparams
        self.num_other_areas = len(hps.area_names) - 1
        
        # Build Generator
        self.gen_cell = GRUCellBase(
            hps.co_dim + hps.com_dim * self.num_other_areas + hps.ext_input_dim + hps.gv_dim,
            hps.gen_dim,
            clip_value=hps.cell_clip,
        )
        
        # Build Controller (optional)
        if hps.use_con:
            # Create the controller
            self.con_cell = GRUCellBase(
                hps.ci_enc_dim + hps.fac_dim,
                hps.con_dim,
                clip_value=hps.cell_clip,
            ).float()
            
            # Mapping from controller hidden states to inferred inputs parameters
            self.co_map = ScaledLinear(hps.con_dim, hps.co_dim * 2)
            
        # Mapping to factors
        self.fac_map = NormLinear(hps.gen_dim, hps.fac_dim, bias=True)
        fanin_normal_init_(self.fac_map)
        
        # Dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)

    def forward(self, input, h_0, sample:bool=True, eps=None):
        hps = self.hparams
        
        # Previous hidden state
        con_state, gen_state, factor = torch.split(
            h_0.float(),
            [hps.con_dim, hps.gen_dim, hps.fac_dim],
            dim=1
        )
        
        # Input
        ci_step, com_step, _, ext_input_step = torch.split(
            input.float(),
            [hps.ci_enc_dim, hps.com_dim * self.num_other_areas, hps.co_dim, hps.ext_input_dim+hps.gv_dim],
            dim=1
        )

        if hps.use_con:
            # Controller forward pass
            con_input = torch.cat([ci_step, factor], dim=1)
            con_input_drop = self.dropout(con_input)
            con_state = self.con_cell(con_input_drop, con_state)
            
            # Map to inferred input parameters
            co_params = self.co_map(con_state)
            co_mean, co_logvar = torch.split(co_params, hps.co_dim, dim=1)
            co_std = torch.sqrt(torch.exp(co_logvar) + hps.co_post_var_min)
            co_post = self.hparams.co_prior.make_posterior(co_mean, co_std)
            # con_output = co_post.rsample() if sample else co_mean
            if sample:
                con_output = co_mean + co_std * eps
            else:
                con_output = co_mean
            
            # Generator input
            gen_input = torch.cat([con_output, com_step, ext_input_step], dim=1)
        else:
            gen_input = torch.cat([com_step, ext_input_step], dim=1)
            con_output = None
            co_mean = torch.zeros(input.shape[0], hps.co_dim)
            co_std = torch.zeros(input.shape[0], hps.co_dim)
            
        # Generator forward pass
        gen_state = self.gen_cell(gen_input, gen_state)
        gen_state_drop = self.dropout(gen_state)
        
        # Map to factors
        factor = self.fac_map(gen_state_drop)
        
        hidden = torch.cat([con_state, gen_state, factor], dim=1)
        return hidden, torch.cat([co_mean, co_std], dim=1), con_output
    
