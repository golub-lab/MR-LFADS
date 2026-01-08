import torch
import numpy as np
import torch.nn as nn
from mrlfads.blocks import GRUBase, ScaledLinear
from mrlfads.utils import HParams

class EmptyGlobalVar(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super().__init__()
        self.params = params
        self.hparams = HParams({'gv_dim': 0})
        self.area_names = list(self.params.keys())
        self.variational = False
        self.gv_prior = None
        
    def build(self, info, batch):
        sessions = sorted(batch.keys()) # session structure is provided but not used currently
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        self.batch_size = sum(batch_sizes)
        
    def forward(self, area_name, t, sample=True, kwargs={}):
        gv_params = torch.zeros(self.batch_size, 0).to('cuda')
        gv_samp = torch.zeros(self.batch_size, 0).to('cuda')
        return gv_params, gv_samp

class NoFeedbackController(nn.Module):
    def __init__(self, params, global_area_params):
        super().__init__()
        self.params = params # are area_params
        self.area_names = list(self.params.keys())
        self.variational = True
        
        default_hps = {
            'use_scalar': False,
        }
        default_hps.update(global_area_params)
        
        self.gv_prior = default_hps['gv_prior']
        del default_hps['gv_prior']
        self.hparams = hps = HParams(default_hps)
        assert hasattr(self.hparams, 'con_size')
        assert hasattr(self.hparams, 'gv_dim')
        assert hasattr(self.hparams, 'ic_enc_seq_len')
        
        # Build an uni-directional controller that receives all area data as input
        data_size = sum([val['num_neurons']['n0'] for val in self.params.values()])
        self.con_cell = GRUBase(
            data_size, hps.con_size, clip=5.0
        ).float().to('cuda')
        self.con_map = ScaledLinear(hps.con_size, hps.gv_dim * 2).to('cuda') # h_con -> gv_params
        self.dropout = nn.Dropout(0.3).to('cuda')
        
        # Build an scalar mask per area
        self.areas_scalar = nn.ParameterDict()
        for ia, area_name in enumerate(params.keys()):
            self.areas_scalar[area_name] = nn.Parameter(torch.ones(hps.gv_dim * 2), requires_grad=hps.use_scalar)
        
    def build(self, info, batch):
        # Session not implemented
        sess_idx = 0
        hps = self.hparams
        
        data = torch.cat([batch[sess_idx].encod_data[area_name] for area_name in self.area_names], dim=2).to('cuda')
        batch_size, _, data_size = data.shape
        h0 = torch.zeros(batch_size, hps.con_size).to('cuda')
        self.output = self.con_cell(data, h0)[0]
        
    def forward(self, area_name, t, sample=True, kwargs={}):
        hps = self.hparams
        h_con = self.output[:, t + hps.ic_enc_seq_len]
        gv_params = self.con_map(h_con)
        gv_mean, gv_logvar = torch.split(gv_params, hps.gv_dim, dim=1)
        
        # Scale by area-specific scalar
        sc_mean, sc_logvar = torch.split(self.areas_scalar[area_name], hps.gv_dim)
        gv_mean = gv_mean * sc_mean.reshape(1, -1)
        gv_logvar = gv_logvar * sc_logvar.reshape(1, -1)
        
        gv_std = torch.sqrt(torch.exp(gv_logvar) + 1.0e-4) # minimum variance to avoid zero error
        gv_out = self.gv_prior.psample(gv_mean, gv_std) if sample else gv_mean
        return torch.cat([gv_mean, gv_std], dim=1), gv_out
    
class DataGlobalVar(nn.Module):
    def __init__(self, params, global_area_params):
        super().__init__()
        self.params = params # are area_params
        self.area_names = list(self.params.keys())
        self.variational = True
        
        default_hps = {
            'use_scalar': False,
        }
        default_hps.update(global_area_params)
        
        self.gv_prior = default_hps['gv_prior']
        del default_hps['gv_prior']
        self.gv_generator = default_hps['gv_generator']
        del default_hps['gv_generator']
        self.hparams = hps = HParams(default_hps) 
        assert hasattr(self.hparams, 'gv_dim')
        assert hasattr(self.hparams, 'data_size')
        
        # Build a map from data to area
        self.data_map = ScaledLinear(hps.data_size, hps.gv_dim * 2).to('cuda') # h_con -> gv_params
        self.dropout = nn.Dropout(0.3).to('cuda')
        
        # Build an scalar mask per area
        self.areas_scalar = nn.ParameterDict()
        for ia, area_name in enumerate(params.keys()):
            self.areas_scalar[area_name] = nn.Parameter(torch.ones(hps.gv_dim * 2), requires_grad=hps.use_scalar)
            
    def build(self, info, batch):
        self.output = self.gv_generator(batch, info)
        
    def forward(self, area_name, t, sample=True, kwargs={}):
        hps = self.hparams
        h_con = self.output[:, t]
        gv_params = self.data_map(h_con)
        gv_mean, gv_logvar = torch.split(gv_params, hps.gv_dim, dim=1)
        
        # Scale by area-specific scalar
        sc_mean, sc_logvar = torch.split(self.areas_scalar[area_name], hps.gv_dim)
        gv_mean = gv_mean * sc_mean.reshape(1, -1)
        gv_logvar = gv_logvar * sc_logvar.reshape(1, -1)
        
        gv_std = torch.sqrt(torch.exp(gv_logvar) + 1.0e-4) # minimum variance to avoid zero error
        gv_out = self.gv_prior.psample(gv_mean, gv_std) if sample else gv_mean # same param, diff sample per area
        return torch.cat([gv_mean, gv_std], dim=1), gv_out
    
class JointGlobalVar(nn.Module):
    def __init__(self, params, global_area_params):
        super().__init__()
        self.params = params # are area_params
        self.area_names = list(self.params.keys())
        self.variational = True
        
        default_hps = {
            'use_scalar': False,
        }
        default_hps.update(global_area_params)
        
        self.gv_prior = default_hps['gv_prior']
        del default_hps['gv_prior']
        self.hparams = hps = HParams(default_hps)
        assert hasattr(self.hparams, 'gv_dim')
        assert hasattr(self.hparams, 'data_size')
        
        # Define name
        assert hasattr(self.hparams, 'name')
        self.name = hps.name
        
        # Build a map from data to area
        self.data_map = ScaledLinear(hps.data_size, hps.gv_dim * 2).to('cuda') # h_gv -> gv_params
        self.dropout = nn.Dropout(0.3).to('cuda')
        
        # Build an scalar mask per area
        self.areas_scalar = nn.ParameterDict()
        for ia, area_name in enumerate(params.keys()):
            self.areas_scalar[area_name] = nn.Parameter(torch.ones(hps.gv_dim * 2), requires_grad=hps.use_scalar)
            
    def build(self, info, batch):
        self.output = None
        
    def forward(self, area_name, t, sample=True, kwargs={}):
        emission_state_dict = kwargs['emission_state_dict']
        if area_name == self.name:
            self.output = h_con = emission_state_dict[self.name] # (batch, neuron)
        
        hps = self.hparams
        h_con = self.output[..., :self.hparams.data_size]
        gv_params = self.data_map(h_con)
        gv_mean, gv_logvar = torch.split(gv_params, hps.gv_dim, dim=1)
        
        # Scale by area-specific scalar
        sc_mean, sc_logvar = torch.split(self.areas_scalar[area_name], hps.gv_dim)
        gv_mean = gv_mean * sc_mean.reshape(1, -1)
        gv_logvar = gv_logvar * sc_logvar.reshape(1, -1)
        gv_std = torch.sqrt(torch.exp(gv_logvar) + 1.0e-4) # minimum variance to avoid zero error
        gv_out = self.gv_prior.psample(gv_mean, gv_std) if sample else gv_mean
        return torch.cat([gv_mean, gv_std], dim=1), gv_out
    
class RawDataGlobalVar(nn.Module):
    def __init__(self, params, global_area_params):
        super().__init__()
        self.params = params # are area_params
        self.area_names = list(self.params.keys())
        self.variational = True
        
        default_hps = {
            'use_scalar': False,
        }
        default_hps.update(global_area_params)
        
        self.gv_prior = default_hps['gv_prior']
        del default_hps['gv_prior']
        self.hparams = hps = HParams(default_hps)
        assert hasattr(self.hparams, 'gv_dim')
        assert hasattr(self.hparams, 'data_size')
        assert hasattr(self.hparams, 'ic_enc_seq_len')
        
        # Build a map from data to area
        self.data_map = ScaledLinear(hps.data_size, hps.gv_dim * 2).to('cuda') # h_gv -> gv_params
        self.dropout = nn.Dropout(0.3).to('cuda')
        
        # Build an scalar mask per area
        self.areas_scalar = nn.ParameterDict()
        for ia, area_name in enumerate(params.keys()):
            self.areas_scalar[area_name] = nn.Parameter(torch.ones(hps.gv_dim * 2), requires_grad=hps.use_scalar)
            
    def build(self, info, batch):
        sess_idx = 0
        self.output = torch.cat([batch[sess_idx].encod_data[area_name] for area_name in self.area_names], dim=2).to('cuda')
        
    def forward(self, area_name, t, sample=True, kwargs={}):
        hps = self.hparams
        h_con = self.output[:, t + hps.ic_enc_seq_len]
        gv_params = self.data_map(h_con)
        gv_mean, gv_logvar = torch.split(gv_params, hps.gv_dim, dim=1)
        
        # Scale by area-specific scalar
        sc_mean, sc_logvar = torch.split(self.areas_scalar[area_name], hps.gv_dim)
        gv_mean = gv_mean * sc_mean.reshape(1, -1)
        gv_logvar = gv_logvar * sc_logvar.reshape(1, -1)
        gv_std = torch.sqrt(torch.exp(gv_logvar) + 1.0e-4) # minimum variance to avoid zero error
        gv_out = self.gv_prior.psample(gv_mean, gv_std) if sample else gv_mean
        return torch.cat([gv_mean, gv_std], dim=1), gv_out