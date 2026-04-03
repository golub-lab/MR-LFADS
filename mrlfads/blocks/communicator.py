import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

class Communicator(nn.Module):
    def __init__(self, hparams, com_prior, self_name):
        super().__init__()

        self.name = self_name
        self.hparams = dict(hparams.__dict__)

        self.area_names = list(self.hparams["area_names"])
        self.other_area_names = deepcopy(self.area_names)
        self.other_area_names.remove(self.name)

        self.hparams["com_dim_total"] = self.hparams["com_dim"] * len(self.other_area_names)
        self.com_prior = com_prior

        self._build_areas()

        for area_name in self.other_area_names:
            hn_idx = []
            other_num_neurons = list(self.hparams["other_area_hps_dict"][area_name]["num_neurons"].values())
            offset = 0
            for isess, hn_idx_sess in enumerate(self.hparams["hn_indices"][area_name]):
                hn_idx_adj = np.array(hn_idx_sess) + offset
                hn_idx += list(hn_idx_adj)
                offset += other_num_neurons[isess]
            hn_idx = torch.tensor(hn_idx, dtype=torch.long)
            mask = torch.zeros(sum(other_num_neurons), dtype=torch.bool)
            if hn_idx.numel() > 0:
                mask[hn_idx] = True
            self.register_buffer(f"mask_{area_name}", mask.unsqueeze(0))

    def forward(
        self,
        factor_state,
        sample: bool = True,
        mask_hn: bool = False,
        eps = None,
    ):
        hps = self.hparams

        batch_size = factor_state[0].size(0)
        device = factor_state[0].device
        dtype = factor_state[0].dtype
        m_means = torch.zeros(batch_size, hps["com_dim_total"], device=device, dtype=dtype)
        m_stds = torch.zeros(batch_size, hps["com_dim_total"], device=device, dtype=dtype)

        add_one = 0
        for ia, area_name in enumerate(self.area_names):

            # Adjust indexing
            if area_name == self.name:
                add_one = 1
                continue
            idx = ia - add_one

            if mask_hn:
                f = factor_state[ia]
                mask = getattr(self, f"mask_{area_name}")
                factor_masked = f.masked_fill(mask, 0)
            else:
                factor_masked = factor_state[ia]

            num_factor_included = hps["total_ems_dim_dict"][area_name]
            factor_cut = factor_masked[..., :num_factor_included]

            m_params = self.areas_linear[area_name](factor_cut)
            m_mean, m_logvar = torch.split(m_params, hps["com_dim"], dim=1)
            m_std = torch.sqrt(torch.exp(m_logvar) + hps["com_post_var_min"])

            start = hps["com_dim"] * idx
            end = hps["com_dim"] * (idx + 1)
            m_means[:, start:end] = m_mean
            m_stds[:, start:end] = m_std

        if sample:
            com_samp = m_means + m_stds * eps
        else:
            com_samp = m_means
        com_params = torch.cat([m_means, m_stds], dim=1)
        return com_samp, com_params

    def _build_areas(self):
        hps = self.hparams
        self.areas_linear = nn.ModuleDict()
        for area_name in self.other_area_names:
            self.areas_linear[area_name] = nn.Linear(
                hps["total_ems_dim_dict"][area_name],
                2 * hps["com_dim"],
            )