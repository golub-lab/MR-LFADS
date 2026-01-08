import torch
import torch.nn.functional as F
from torch import nn
from ..blocks import GRUBase, BiGRUBase, NormLinear, ScaledLinear
    
class ICSampler(nn.Module):
    def __init__(self, hparams, prior):
        super().__init__()
        self.hparams = hps = hparams
        
        self.prior = prior
        self.dropout = nn.Dropout(0.3)
        self.ic_to_g0 = ScaledLinear(hps.ic_dim, hps.gen_size)
        self.g0_to_f0 = NormLinear(hps.gen_size, hps.fac_dim, bias=False)
        self.c0 = nn.Parameter(torch.zeros((1, hps.con_size), requires_grad=True))
        
    def forward(
        self,
        ic_mean,
        ic_std,
        sample: bool = True,
    ):
        ic_out = self.prior.psample(ic_mean, ic_std) if sample else ic_mean
        g0 = self.ic_to_g0(ic_out)
        g0 = self.dropout(g0)
        f0 = self.g0_to_f0(g0)
        return self.c0, g0, f0

class SREncoder(nn.Module):
    def __init__(self, hparams: dict, prior):
        super().__init__()
        self.hparams = hps = hparams

        # IC encoder
        self.ic_enc_h0 = nn.Parameter(
            torch.zeros((2, 1, hps.ic_enc_size), requires_grad=True)
        )
        self.ic_enc = BiGRUBase(
            input_size=hps.data_dim,
            hidden_size=hps.ic_enc_size,
            clip=5.0,
        )
        self.ic_map = ScaledLinear(hps.ic_enc_size * 2, hps.ic_dim * 2)
        
        # Con encoder
        if hps.use_con:
            self.ci_enc_h0 = nn.Parameter(
                torch.zeros((1, hps.ci_size), requires_grad=True)
            )
            self.ci_enc = GRUBase(
                input_size=hps.data_dim,
                hidden_size=hps.ci_size,
                clip=5.0,
            )
        self.dropout = nn.Dropout(0.3)
        
        # IC sample and map
        self.ic_sampler = ICSampler(hps, prior)

    def forward(self, data: torch.Tensor, sample=True):
        hps = self.hparams
        data = self.dropout(data)
        assert data.shape[1] == hps.seq_len
        
        # Optionally split time series for IC encoder and controller encoder
        if hps.ic_enc_seq_len > 0:
            ic_enc_data = data[:, : hps.ic_enc_seq_len, :]
            ci_enc_data = data[:, hps.ic_enc_seq_len :, :]
        else:
            ic_enc_data = data
            ci_enc_data = data
            
        ic_enc_h0 = torch.tile(self.ic_enc_h0, (1, data.shape[0], 1))
        _, h_n = self.ic_enc(ic_enc_data, ic_enc_h0)
        h_n = torch.cat([*h_n], dim=1)
        h_n = self.dropout(h_n)
        ic_params = self.ic_map(h_n)
        ic_mean, ic_logvar = torch.split(ic_params, hps.ic_dim, dim=1)
        ic_std = torch.sqrt(torch.exp(ic_logvar) + 1e-4)
        
        if hps.use_con:
            ci_enc_h0 = torch.tile(self.ci_enc_h0, (data.shape[0], 1))
            ci, _ = self.ci_enc(ci_enc_data, ci_enc_h0)
        else:
            ci = torch.zeros(data.shape[0], hps.seq_len - hps.ic_enc_seq_len, 0).to(data.device)
            
        c0, g0, f0 = self.ic_sampler(ic_mean, ic_std, sample)
        return (ic_mean, ic_std, ci), (c0, g0, f0)