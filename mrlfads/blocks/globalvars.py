import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mrlfads.utils.common_utils import HParams
from mrlfads.utils.torch_utils import GRUBase, BiGRUBase, ScaledLinear

class EmptyGlobalVar(nn.Module):
    """Placeholder global area for a MRLFADS module."""
    def __init__(self, params, *args, **kwargs):
        super().__init__()
        self.params = params # area_params from MRLFADS
        self.hparams = HParams({'gv_dim': 0})
        self.area_names = list(self.params.keys())
        self.gv_prior = None
        self.variational = False
        
    def build(self, info, batch, *args, **kwargs):
        sessions = sorted(batch.keys()) 
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        self.batch_size = sum(batch_sizes)
        self.device = batch[0].encod_data[self.area_names[0]].device
        
    def forward(self, t, sample=True, **kwargs):
        gv_params = torch.zeros(self.batch_size, 0, device=self.device)
        gv_samp = torch.zeros(self.batch_size, 0, device=self.device)
        return gv_params, gv_samp

class LinearGlobalVar(nn.Module):
    """Global module is linear readout of regional spike counts.
    Notes:
        - Take encod_data as input, which already masks holdout neurons.
    """
    def __init__(self, params, global_area_params):
        super().__init__()
        self.params = params # area related params
        self.gv_prior = None
        self.variational = False
        
        assert 'gv_dim' in global_area_params.keys()
        if 'lag' not in global_area_params.keys(): global_area_params['lag'] = 0
        hps = self.hparams = HParams(global_area_params)
        self.area_names = list(self.params.keys())
        
        # Build linear readin per session
        self.readin = nn.ModuleList()
        nn_per_sess = []
        for area_name in self.area_names:
            num_neurons = self.params[area_name]
            nn_per_sess.append( num_neurons )
            
        nn_per_sess = np.sum(nn_per_sess, axis=0) # total neurons across areas per session
        for nneu in nn_per_sess:
            self.readin.append( nn.Linear(nneu, hps.gv_dim) )
        
    def build(self, info, batch, *args, **kwargs):
        sessions = sorted(batch.keys()) 
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        self.batch_size = sum(batch_sizes)
        self.device = batch[0].encod_data[self.area_names[0]].device
        
        data = []
        for s in sessions:
            arr = [batch[s].encod_data[area_name] for area_name in self.area_names]
            arr = torch.cat(arr, axis=2) # shape = (batch, time, total neurons)
            data.append( self.readin[s](arr.float()) )
        self.data = torch.cat(data, axis=0)
        
    def forward(self, t, sample=True, **kwargs):
        hps = self.hparams
        gv_params = torch.zeros(self.batch_size, hps.gv_dim * 2, device=self.device)
        gv_samp = self.data[:, t-hps.lag]
        return gv_params, gv_samp

class NoFeedbackController(nn.Module):
    """
    Notes:
        - Takes post-readin data, which already masked holdout neurons
    """
    def __init__(self, params, global_area_params):
        super().__init__()
        self.params = params # area related params from MRLFADS
        self.area_names = list(self.params.keys())
        
        default_hps = {
            'lag': 0,
            'cell_clip': 5.0,
            'dropout_rate': 0.3,
            'gv_post_var_min': 1e-4,
        }
        default_hps.update(global_area_params)
        
        self.gv_prior = default_hps['gv_prior']
        del default_hps['gv_prior']
        self.hparams = hps = HParams(default_hps)
        
        assert hasattr(self.hparams, 'con_dim')
        assert hasattr(self.hparams, 'gv_dim')
        self.variational = True
        
        # Build an BI-directional controller
        # It receives all area encoded data as input
        # Optionally has a layer to reduce dimensionality
        data_size = sum([val['data_dim'] for val in self.params.values()])
        if hasattr(self.hparams, 'gv_enc_dim'):
            self.data_map = ScaledLinear(data_size, hps.gv_enc_dim)
            con_inp_size = hps.gv_enc_dim
        else:
            self.data_map = nn.Identity()
            con_inp_size = data_size
        
        self.con_gru = BiGRUBase(
            input_size=con_inp_size,
            hidden_size=hps.con_dim//2,
            clip_value=hps.cell_clip,
        ).float()#.to('cuda')
        self.con_h0 = nn.Parameter(
                torch.zeros((2, 1, hps.con_dim//2), requires_grad=True)
            )
        self.con_map = ScaledLinear(hps.con_dim, hps.gv_dim * 2)#.to('cuda') # h_con -> gv_params
        self.dropout = nn.Dropout(hps.dropout_rate)#.to('cuda')
        
    def build(self, info, batch, area_data_dict, *args, **kwargs):
        data = torch.cat(
            [area_data_dict[an] for an in self.area_names],
            dim = 2,
        )
        data = self.data_map(data)
        data = self.dropout(data)
        
        batch_size, time_size = data.shape[:2]
        con_h0 = torch.tile(self.con_h0, (1, batch_size, 1))
        out, _ = self.con_gru(data, con_h0)
        self.output = out
        
    def forward(self, t, sample=True, **kwargs):
        hps = self.hparams
        out = self.output[:, t-hps.lag]
        gv_params = self.con_map(out)
        gv_mean, gv_logvar = torch.split(gv_params, hps.gv_dim, dim=1)
        gv_std = torch.sqrt(torch.exp(gv_logvar) + hps.gv_post_var_min)
        
        gv_post = self.gv_prior.make_posterior(gv_mean, gv_std)
        gv_samp = gv_post.rsample() if sample else gv_mean
        return torch.cat([gv_mean, gv_std], dim=1), gv_samp
    
