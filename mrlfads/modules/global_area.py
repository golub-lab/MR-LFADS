import torch
import numpy as np
import torch.nn as nn
from mrlfads.blocks import GRUBase, ScaledLinear
from mrlfads.utils import HParams

class EmptyGlobalVar(nn.Module):
    """Placeholder global area for a MRLFADS module."""
    def __init__(self, params, *args, **kwargs):
        super().__init__()
        self.params = params
        self.hparams = HParams({'gv_dim': 0})
        self.area_names = list(self.params.keys())
        self.variational = False
        self.gv_prior = None
        
    def build(self, info, batch):
        sessions = sorted(batch.keys()) # session structure is provided but not used
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        self.batch_size = sum(batch_sizes)
        
    def forward(self, area_name, t, sample=True, kwargs={}):
        gv_params = torch.zeros(self.batch_size, 0).to('cuda')
        gv_samp = torch.zeros(self.batch_size, 0).to('cuda')
        return gv_params, gv_samp
