import torch
import torch.nn as nn
from copy import deepcopy
        
class SRCommunicator(nn.Module):
    """Communicator for a SRLFADS module."""
    def __init__(self, hparams, com_prior, self_name):
        super().__init__()
        self.name = self_name
        self.hparams = hparams
        self.other_area_names = deepcopy(hparams.area_names)
        self.other_area_names.remove(self.name)
        self.com_dim_total = hparams.com_dim * len(self.other_area_names)
        self.com_prior = com_prior
        self._build_areas()
        
    def forward(
        self,
        factor_state,
        sample: bool = True,
        mask_hn: bool = False,
    ):
        hps = self.hparams
        batch_size = factor_state[self.name].size(0)
        m_means = torch.zeros(batch_size, self.com_dim_total)
        m_stds = torch.zeros(batch_size, self.com_dim_total)
        
        base = hps.com_dim * len(self.other_area_names)
        for ia, area_name in enumerate(self.other_area_names):
            
            # heldout neuron mask
            if mask_hn:
                f = factor_state[area_name]
                hn_idx = torch.Tensor(hps.hn_indices[area_name][0]).to(int) # use hn_indices of the first session only
                mask = torch.zeros(f.shape[-1])
                mask[hn_idx] = 1
                mask = mask.to(bool).unsqueeze(0).to(f.device)
                factor_masked = torch.where(mask, torch.zeros_like(f), f)
            else:
                factor_masked = factor_state[area_name]
                
            # factor mask
            num_factor_included = hps.total_ems_dim_dict[area_name]
            factor_cut = factor_masked[..., :num_factor_included]

            m_params = self.areas_linear[area_name](factor_cut)
            m_mean, m_logvar = torch.split(m_params, hps.com_dim, dim=1)
            m_std = torch.sqrt(torch.exp(m_logvar) + 1e-4)
            m_means[:, hps.com_dim * ia: hps.com_dim * (ia+1)] = m_mean
            m_stds[:, hps.com_dim * ia: hps.com_dim * (ia+1)] = m_std
            
        com_samp = self.com_prior.psample(m_means, m_stds) if sample else m_means
        com_params = torch.cat([m_means, m_stds], dim=1)
        return com_samp, com_params
    
    def _build_areas(self):
        hps = self.hparams
        self.areas_linear = nn.ModuleDict()
        for ia, area_name in enumerate(self.other_area_names):
            self.areas_linear[area_name] = nn.Linear(hps.total_ems_dim_dict[area_name], 2 * hps.com_dim)