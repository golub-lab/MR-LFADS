import torch
import torch.nn.functional as F
from torch import nn

from mrlfads.utils.torch_utils import ScaledLinear, NormLinear, GRUBase, BiGRUBase

class SREncoder(nn.Module):
    def __init__(self, hparams: dict, ic_prior):
        super().__init__()
        self.hparams = hps = hparams
        self.prior = ic_prior

        # Build IC encoder
        self.ic_enc = BiGRUBase(
            input_size=hps.data_dim,
            hidden_size=hps.ic_enc_dim,
            clip_value=hps.cell_clip,
        )
        self.ic_enc_h0 = nn.Parameter(
            torch.zeros((2, 1, hps.ic_enc_dim), requires_grad=True)
        )
        
        # Build map from ic_enc hidden state --> ic parameters
        self.ic_map = ScaledLinear(hps.ic_enc_dim * 2, hps.ic_dim * 2)
        
        # Build CI encoder (optional)
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Build CI encoder
            self.ci_enc = GRUBase(
                input_size=hps.data_dim,
                hidden_size=hps.ci_enc_dim,
                clip_value=hps.cell_clip,
            )
            self.ci_enc_h0 = nn.Parameter(
                torch.zeros((1, hps.ci_enc_dim), requires_grad=True)
            )
        
        # Map for ic parameters --> ic sample --> factor, generator initial states
        self.ic_to_g0 = ScaledLinear(hps.ic_dim, hps.gen_dim)
        self.g0_to_f0 = NormLinear(hps.gen_dim, hps.fac_dim, bias=False)
        self.c0 = nn.Parameter(torch.zeros((1, hps.con_dim), requires_grad=True))
        
        # Dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)

    def forward(self, data: torch.Tensor, sample:bool=True):
        hps = self.hparams
        batch_size = data.shape[0]
        data = self.dropout(data)
        
        # Separate data segments for IC encoding and CI encoding (optional)
        if hps.ic_enc_seq_len > 0:
            ic_enc_data = data[:, : hps.ic_enc_seq_len, :]
            ci_enc_data = data[:, hps.ic_enc_seq_len :, :]
        else:
            ic_enc_data = data
            ci_enc_data = data
            
        # IC encoder forward pass
        ic_enc_h0 = torch.tile(self.ic_enc_h0, (1, batch_size, 1))
        _, h_n = self.ic_enc(ic_enc_data, ic_enc_h0)
        h_n = torch.cat([*h_n], dim=1)
        h_n = self.dropout(h_n)
        
        # Map to IC parameters
        ic_params = self.ic_map(h_n)
        ic_mean, ic_logvar = torch.split(ic_params, hps.ic_dim, dim=1)
        ic_std = torch.sqrt(torch.exp(ic_logvar) + hps.ic_post_var_min)
        
        # CI encoder forward pass
        if self.use_con:
            ci_enc_h0 = torch.tile(self.ci_enc_h0, (batch_size, 1))
            ci, _ = self.ci_enc(ci_enc_data, ci_enc_h0)
            ci = F.pad(ci, (0, 0, hps.ci_lag, 0, 0, 0))
            ci = ci[:, :hps.seq_len - hps.ic_enc_seq_len, :]
        else:
            ci = torch.zeros(data.shape[0], hps.seq_len - hps.ic_enc_seq_len, 0).to(data.device)

        # Sample ic, map to factor, generator initial state
        ic_post = self.prior.make_posterior(ic_mean, ic_std)
        ic_samp = ic_post.rsample() if sample else ic_mean
        gen_init = self.ic_to_g0(ic_samp)
        gen_init = self.dropout(gen_init)
        factor_init = self.g0_to_f0(gen_init)
        ic_params = torch.cat([ic_mean, ic_std], dim=1)
        return (ic_params, ci), (self.c0, gen_init, factor_init)
    
class BiEncoder(nn.Module):
    def __init__(self, hparams: dict, ic_prior):
        super().__init__()
        self.hparams = hps = hparams
        self.prior = ic_prior

        # Build IC encoder
        self.ic_enc = BiGRUBase(
            input_size=hps.data_dim,
            hidden_size=hps.ic_enc_dim,
            clip_value=hps.cell_clip,
        )
        self.ic_enc_h0 = nn.Parameter(
            torch.zeros((2, 1, hps.ic_enc_dim), requires_grad=True)
        )
        
        # Build map from ic_enc hidden state --> ic parameters
        self.ic_map = ScaledLinear(hps.ic_enc_dim * 2, hps.ic_dim * 2)
        
        # Build CI encoder (optional)
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Build CI encoder
            self.ci_enc = BiGRUBase(
                input_size=hps.data_dim,
                hidden_size=hps.ci_enc_dim//2,
                clip_value=hps.cell_clip,
            )
            self.ci_enc_h0 = nn.Parameter(
                torch.zeros((2, 1, hps.ci_enc_dim//2), requires_grad=True)
            )
        
        # Map for ic parameters --> ic sample --> factor, generator initial states
        self.ic_to_g0 = ScaledLinear(hps.ic_dim, hps.gen_dim)
        self.g0_to_f0 = NormLinear(hps.gen_dim, hps.fac_dim, bias=False)
        self.c0 = nn.Parameter(torch.zeros((1, hps.con_dim), requires_grad=True))
        
        # Dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)

    def forward(self, data: torch.Tensor, sample:bool=True, eps=None):
        hps = self.hparams
        batch_size = data.shape[0]
        data = self.dropout(data)
        
        # Separate data segments for IC encoding and CI encoding (optional)
        if hps.ic_enc_seq_len > 0:
            ic_enc_data = data[:, : hps.ic_enc_seq_len, :]
            ci_enc_data = data[:, hps.ic_enc_seq_len :, :]
        else:
            ic_enc_data = data
            ci_enc_data = data
            
        # IC encoder forward pass
        ic_enc_h0 = torch.tile(self.ic_enc_h0, (1, batch_size, 1))
        _, h_n = self.ic_enc(ic_enc_data, ic_enc_h0)
        h_n = torch.cat([*h_n], dim=1)
        h_n = self.dropout(h_n)
        
        # Map to IC parameters
        ic_params = self.ic_map(h_n)
        ic_mean, ic_logvar = torch.split(ic_params, hps.ic_dim, dim=1)
        ic_std = torch.sqrt(torch.exp(ic_logvar) + hps.ic_post_var_min)
        
        # CI encoder forward pass
        if self.use_con:
            ci_enc_h0 = torch.tile(self.ci_enc_h0, (1, batch_size, 1))
            ci, _ = self.ci_enc(ci_enc_data, ci_enc_h0)
            ci_fwd, ci_bwd = torch.split(ci, hps.ci_enc_dim//2, dim=2)
            ci_fwd = F.pad(ci_fwd, (0, 0, hps.ci_lag, 0, 0, 0))
            ci_bwd = F.pad(ci_bwd, (0, 0, 0, hps.ci_lag, 0, 0))
            ci_len = hps.seq_len - hps.ic_enc_seq_len
            ci = torch.cat([ci_fwd[:, :ci_len, :], ci_bwd[:, -ci_len:, :]], dim=2)
        else:
            ci = torch.zeros(data.shape[0], hps.seq_len - hps.ic_enc_seq_len, 0).to(data.device)

        # Sample ic, map to factor, generator initial state
        ic_post = self.prior.make_posterior(ic_mean, ic_std)
        ic_samp = ic_post.rsample() if sample else ic_mean
        gen_init = self.ic_to_g0(ic_samp)
        gen_init = self.dropout(gen_init)
        factor_init = self.g0_to_f0(gen_init)
        
        ic_params = torch.cat([ic_mean, ic_std], dim=1)
        return (ic_params, ci), (self.c0, gen_init, factor_init)