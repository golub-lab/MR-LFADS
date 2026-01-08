import os
import gc
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions import Independent, Normal, StudentT, kl_divergence

import mrlfads.paths as path
from mrlfads.run import load
from mrlfads.utils import Batch

class ScaledLinear(nn.Linear):
    """Linear layer with scaled weight initialization.

    This layer initializes weights from a zero-mean normal distribution with
    standard deviation 1 / sqrt(in_features), and initializes biases to zero.
    """
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class NormLinear(ScaledLinear):
    """Linear layer with row-normalized weights.

    This layer applies L2 normalization to the weight matrix along the output
    dimension at each forward pass. Optionally, it can reuse the scaled
    initialization defined in `ScaledLinear`.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        init_weights: bool = False,
    ):
        """
        Args:
            init_weights: If True, initialize weights using the scaled
                initialization from `ScaledLinear`. If False, use the default
                `nn.Linear` initialization. Defaults to False.
        """
        if init_weights:
            super().__init__(in_features, out_features, bias=bias)
        else:
            nn.Linear.__init__(self, in_features, out_features, bias=bias)

    def forward(self, x):
        return F.linear(x, F.normalize(self.weight, dim=1), self.bias)
    
class MLPBase(nn.Module):
    """Base class for constructing a multilayer perceptron (MLP)."""
    def __init__(self, features_list):
        """
        Args:
            features_list: List of layer specifications. Each element is a
                three-tuple of the form (input_dim, output_dim, nonlinearity),
                defining the dimensions and activation function for a layer.
        """
        super(MLPBase, self).__init__()
        self.features_list = features_list
        
        # Define self.model
        layers = OrderedDict({})
        for i, (in_features, out_features, activation_type) in enumerate(self.features_list):
            layers[f"linear{i}"] = nn.Linear(in_features=in_features, out_features=out_features)
            if activation_type:
                layers[f"{activation_type}{i}"] = get_activation_type(activation_type)
        self.model = nn.Sequential(layers)
        
    def forward(self, *inp):
        return self.model(*inp)
    
class GRUCellBase(nn.GRUCell):
    """GRU cell with custom initialization and optional weight scaling."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip: float = 5.0,
        scale: bool = False,
    ):
        super().__init__(input_size, hidden_size, bias=True)

        self.bias_hh.requires_grad = False
        self.clip = clip

        # Scale input, hidden matrices
        if scale:
            scale_dim_ih = (input_size + hidden_size) / float(input_size)
            scale_dim_hh = (input_size + hidden_size) / float(hidden_size)
        else:
            scale_dim_ih, scale_dim_hh = 1, 1

        with torch.no_grad():
            # weight_ih ~ N(0, 1/sqrt(input_size * scale_dim_ih))
            ih_scale = input_size * scale_dim_ih
            ih_std = 1.0 / torch.sqrt(
                torch.tensor(ih_scale, dtype=self.weight_ih.dtype, device=self.weight_ih.device)
            )
            nn.init.normal_(self.weight_ih, std=ih_std)

            # weight_hh ~ N(0, 1/sqrt(hidden_size * scale_dim_hh))
            hh_scale = hidden_size * scale_dim_hh
            hh_std = 1.0 / torch.sqrt(
                torch.tensor(hh_scale, dtype=self.weight_hh.dtype, device=self.weight_hh.device)
            )
            nn.init.normal_(self.weight_hh, std=hh_std)

            # Biases: bias_ih = 1, last hidden_size entries set to 0; bias_hh = 0
            nn.init.ones_(self.bias_ih)
            self.bias_ih.data[-hidden_size:] = 0.0  # reset gate
            nn.init.zeros_(self.bias_hh)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        # x @ W_ih^T + b_ih
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)

        # Split hidden-to-hidden params
        split_dims = [2 * self.hidden_size, self.hidden_size]
        weight_hh_zr, weight_hh_n = torch.split(self.weight_hh, split_dims)
        bias_hh_zr, bias_hh_n = torch.split(self.bias_hh, split_dims)

        h_all = hidden @ weight_hh_zr.T + bias_hh_zr
        h_z, h_r = torch.chunk(h_all, chunks=2, dim=1)

        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_n = (r * hidden) @ weight_hh_n.T + bias_hh_n
        n = torch.tanh(x_n + h_n)

        hidden = z * hidden + (1 - z) * n
        hidden = torch.clamp(hidden, -self.clip, self.clip)
        return hidden


class GRUBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip: float = 5.0,
    ):
        super().__init__()
        self.inner = GRUCellBase(
            input_size=input_size,
            hidden_size=hidden_size,
            clip=clip,
            scale=True,
        )

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        h = h_0
        hs = torch.empty(*input.shape[:2], h_0.shape[-1], dtype=h.dtype, device=input.device)
        for t in range(input.shape[1]):
            h = self.inner(input[:, t], h)
            hs[:, t] = h
        return hs, h

class BiGRUBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip: float = 5.0,
    ):
        super().__init__()
        self.fwd = GRUBase(input_size, hidden_size, clip=clip)
        self.bwd = GRUBase(input_size, hidden_size, clip=clip)

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        h_fwd, h_bwd = h_0
        hs_fwd, h_fwd = self.fwd(input, h_fwd)
        hs_bwd, h_bwd = self.bwd(torch.flip(input, [1]), h_bwd)
        hs = torch.cat([hs_fwd, torch.flip(hs_bwd, [1])], dim=2)
        h = torch.stack([h_fwd, h_bwd])
        return hs, h
    
class MultivariateNormal(nn.Module):
    def __init__(
        self,
        mean: float,
        variance: float,
        shape: int,
        var_requires_grad: bool = True,
    ):
        super().__init__()
        # Create distribution parameter tensors
        means = torch.ones(shape) * mean
        logvars = torch.log(torch.ones(shape) * variance)
        self.mean = nn.Parameter(means, requires_grad=False)
        self.logvar = nn.Parameter(logvars, requires_grad=var_requires_grad)
        self.posterior = None
        
    def make_posterior(self, post_mean, post_std):
        post_mean = torch.nan_to_num(post_mean, posinf=1e3, neginf=-1e3)
        post_std = torch.nan_to_num(post_std, posinf=1e3, neginf=-1e3) + 1e-2
        return self.make_dist(post_mean, post_std)

    def psample(self, post_mean, post_std):
        posterior = self.make_posterior(post_mean, post_std)
        self.posterior = posterior
        return self.posterior.rsample()

    def forward(self, post_mean, post_std, reduction='none', dim=None):
        """
        Args:
            reduction: Controls the aggregation of KL values. Defaults to "none".
                Supported values:
                - "none": Return a scalar KL averaged across all dimensions.
                - "mean": Return a list of length `num_other_areas`, containing a
                  scalar KL per area.
                - "seq": Return a list of length `num_other_areas`, containing a
                  1D tensor of KL values per area across time.
        """
        posterior = self.make_posterior(post_mean, post_std)
        self.posterior = posterior
        
        prior_std = torch.exp(0.5 * self.logvar)
        prior = self.make_dist(self.mean, prior_std)
        
        if reduction == 'none':
            kl_batch = kl_divergence(self.posterior, prior) # shape = posterior.shape[:-1]

            # If kl_batch.shape has a time component, sum across that dimension
            # For initial conditions, there is no time component; for others there is
            if len(kl_batch.shape) > 1: kl_batch = torch.sum(kl_batch, dim=1)
            return torch.mean(kl_batch)
        
        # Other reductions (mean, seq, batch)
        assert reduction in ('mean', 'seq', 'batch'), f'`reduction` cannot be {reduction}, must be none, mean, seq or batch.'
        post_mean = torch.split(post_mean, dim, dim=2)
        post_std = torch.split(post_std, dim, dim=2)
        prior_mean = torch.split(self.mean, dim)
        prior_std = torch.split(torch.exp(0.5 * self.logvar), dim)
            
        kls = []
        for i in range(len(post_mean)):
            posterior = self.make_posterior(post_mean[i], post_std[i])
            prior = self.make_dist(prior_mean[i], prior_std[i])
            kl_batch = kl_divergence(posterior, prior)
            
            if reduction == 'mean': kls.append(kl_batch.mean().item())
            elif reduction == 'seq': kls.append(kl_batch.mean(dim=0)) 
            else: kls.append(kl_batch) # reduction='batch', returns list with elements of shape (batch, time)
        return kls
    
    @staticmethod
    def make_dist(mean, std): return Independent(Normal(mean, std), 1)
    
class Poisson:
    def __init__(self):
        self.name = "poisson"
        self.n_params = 1
        
    def __call__(self, data, params):
        params = self.unbind(params)
        return F.poisson_nll_loss(
            params,
            data,
            full=True,
            reduction="none",
        )
    
    def pseudo_rsquared(self, data, params):
        raise NotImplementedError('Not done debugging')
        nll_model = self(data, params).mean()
        
        # Get neuronal mean across time (per batch per neuron)
        means = data.mean(dim=1, keepdim=True).tile(1, data.shape[1], 1)
        nll_null = self(data, means).mean()
        
        return (1 - nll_model / nll_null).item()

    def unbind(self, params): return params

    def mean(self, params): return torch.exp(params)

    def rsquared(self, data, params, mode: str = "mcfadden"):
        """Computes a reconstruction R sqaured score."""
        data = data.float()
        means_model = self.unbind(params)

        # Null mean + variance across time (batch, 1, neurons)
        T = data.shape[1]
        mean_null = data.mean(dim=1, keepdim=True).expand(-1, T, -1)

        if mode == "mcfadden":
            # Use learned variances (full NLL)
            nll_model = self(data, params).sum()
            nll_null  = F.poisson_nll_loss(
                mean_null,
                data,
                full=True,
                reduction="none",
            ).sum()
            return (1.0 - nll_model / nll_null).item()
        
        elif mode == 'standard':
            return 0

        else:
            raise ValueError(f"Unknown R squared mode: {mode}. Choose from "
                             "['mcfadden']")

class Gaussian:
    def __init__(self, fix_logvar: bool = False, detach_logvar: bool = False):
        self.name = "gaussian"
        self.n_params = 2

        assert not (fix_logvar and detach_logvar), "fix_logvar and detach_logvar cant both be True"
        self.fix_logvar = fix_logvar
        self.detach_logvar = detach_logvar

    def __call__(self, data: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        means, logvars = self.unbind(params)
        if self.fix_logvar:
            # σ² = 1 → log σ² = 0; detach so gradients don't flow through this constraint
            logvars = torch.zeros_like(means).detach()

        recon_all = self.gaussian_nll_loss(
            input=means,
            target=data,
            logvar=logvars,
            reduction="none",
            detach=self.detach_logvar,
        )
        return recon_all

    def rsquared(self, data, params, mode: str = "standard"):
        """Computes a reconstruction R sqaured score."""
        means_model, logvars_model = self.unbind(params)

        # Null mean + variance across time (batch, 1, neurons)
        T = data.shape[1]
        mean_null = data.mean(dim=1, keepdim=True).expand(-1, T, -1)
        var_null = data.var(dim=1, keepdim=True, unbiased=False).clamp_min(1e-12)
        logvar_null = var_null.log().expand(-1, T, -1)

        if mode == "standard":
            # standard R squared: 1 - SSE_model / SSE_null
            sse_model = ((data - means_model) ** 2).sum()
            sse_null = ((data - mean_null) ** 2).sum()
            return (1.0 - sse_model / sse_null).item()

        elif mode == "mcfadden":
            # Use learned variances (full NLL)
            nll_model = self(data, params).sum()
            nll_null  = self.gaussian_nll_loss(
                input=mean_null,
                target=data,
                logvar=logvar_null,
                reduction="sum",
                detach=self.detach_logvar,
            )
            return (1.0 - nll_model / nll_null).item()

        else:
            raise ValueError(f"Unknown R squared mode: {mode}. Choose from "
                             "['standard', 'mcfadden']")

    def unbind(self, params: torch.Tensor):
        means, logvars = torch.chunk(params, 2, dim=-1)
        return means, logvars

    def gaussian_nll_loss(
        self,
        input: torch.Tensor,      # μ
        target: torch.Tensor,     # y
        logvar: torch.Tensor,     # log sigma^2
        *,
        detach: bool = False,
        reduction: str = "mean",  # 'none' | 'mean' | 'sum'
        eps: float = 1e-6,
        beta: float = 1.0,        # coefficient on log sigma^2 term (keep at 1.0 for true NLL)
    ):
        var = logvar.exp().clamp_min(eps)          # sigma^2
        var_resid = var.detach() if detach else var
        sqerr = (target - input) ** 2
        # log sigma^2 (already numerically safe via var clamp)
        logvar_term = torch.log(var)

        per_elem = 0.5 * (sqerr / var_resid + beta * logvar_term) + 0.5 * math.log(2 * math.pi)

        if reduction == "none":
            return per_elem
        elif reduction == "mean":
            return per_elem.mean()
        elif reduction == "sum":
            return per_elem.sum()
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    def mean(self, params):
        return self.unbind(params)[0] # return mean only
    
class MRLFADS_DataGenerator:
    def __init__(self, filename):
        # Load corresponding model
        try: # Fix for CodeOcean read only issues
            state_dict = load(
                os.path.join(path.homepath, 'scratch', filename, 'configs', 'main.yaml'),
                validate=True, # helps initialize some things like current_info
            )
        except:
            state_dict = load(
                os.path.join(path.datapath, filename, 'configs', 'main.yaml'),
                validate=True, # helps initialize some things like current_info
            )
        self.model = state_dict['model'].to('cuda')
        self.area_name = self.model.area_names[0] # assumes there is only 1 area
        del state_dict
        gc.collect()
        
    def __call__(self, batch, info):
        # Pre-process batch
        mod = {}
        for field, val in batch[0]._asdict().items(): # session=0
            arr = torch.cat(list(val.values()), dim=2).to('cuda')
            mod[field] = {self.area_name: arr} 
        
        batch = {0: Batch(**mod)}
        self.model.current_batch = {0: self.model.holdout.preprocess(batch[0])} # need to reset current_batch from the initial validation run
        outputs = self.model(batch)
        output_dist = self.model.areas[self.area_name].output_dist
        return output_dist.mean(outputs[self.area_name][0]).detach() # gives mean only