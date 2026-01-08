import torch
import torch.nn.functional as F

from torch import nn
from ..blocks import GRUCellBase, NormLinear, ScaledLinear

class SRDecoder(nn.Module):
    """Decoder for a SRLFADS module."""
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hps = hparams
        self.num_other_areas = len(hps.area_names) - 1
        
        # Controller
        if hps.use_con:
            self.con_cell = GRUCellBase(
                hps.ci_size + hps.fac_dim, hps.con_size, clip=5.0
            ).float()
            self.con_map = ScaledLinear(hps.con_size, hps.co_dim * 2) # h_con -> co_params
            
        # Generator
        self.gen_cell = GRUCellBase(
            hps.co_dim + hps.com_dim * self.num_other_areas + hps.ext_input_dim + hps.gv_dim,
            hps.gen_size,
            clip=5.0,
        )
        self.gen_map = NormLinear(hps.gen_size, hps.fac_dim, bias=True, init_weights=True)
        
        # Other layers
        self.dropout = nn.Dropout(0.3)
        self.input_dims = [2 * hps.ci_size, hps.ext_input_dim]
        if hps.gv_dim > 0: # deprecated feature
            self.gv_scalar = nn.Parameter(torch.ones(hps.gv_dim), requires_grad=False)

    def forward(self, input, h_0, sample=True):
        hps = self.hparams
        
        h_con, h_gen, f = torch.split(h_0.float(), [hps.con_size, hps.gen_size, hps.fac_dim], dim=1)
        ci_inp, com_inp, _, ext_inp, gv_inp = torch.split(input.float(), [
            hps.ci_size, hps.com_dim * self.num_other_areas, hps.co_dim, hps.ext_input_dim, hps.gv_dim,
        ], dim=1)
        
        if hps.gv_dim > 0:
            gv_inp = self.gv_scalar.unsqueeze(0).tile(len(gv_inp), 1) * gv_inp
            ext_inp = torch.cat([ext_inp, gv_inp], dim=-1)

        if hps.use_con:
            con_inp = torch.cat([ci_inp, f], dim=1)
            con_inp = self.dropout(con_inp)
            
            h_con = self.con_cell(con_inp, h_con)
            co_params = self.con_map(h_con)
            co_mean, co_logvar = torch.split(co_params, hps.co_dim, dim=1)
            co_std = torch.sqrt(torch.exp(co_logvar) + 1.0e-4) # minimum variance to avoid zero error
            con_out = self.hparams.co_prior.psample(co_mean, co_std) if sample else co_mean
            gen_inp = torch.cat([con_out, com_inp, ext_inp], dim=1)
            
        else:
            gen_inp = torch.cat([com_inp, ext_inp], dim=1)
            con_out = None
            co_mean = torch.zeros(input.shape[0], hps.co_dim)
            co_std = torch.zeros(input.shape[0], hps.co_dim)
            
        h_gen = self.gen_cell(gen_inp, h_gen)
        h_gen = self.dropout(h_gen)
        f = self.gen_map(h_gen)
        h = torch.cat([h_con, h_gen, f], dim=1)
        return h, torch.cat([co_mean, co_std], dim=1), con_out
    