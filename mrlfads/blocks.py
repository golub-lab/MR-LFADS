import os
import gc
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import mrlfads.paths as path

from collections import OrderedDict
from torch.distributions import Independent, Normal, StudentT, kl_divergence
from mrlfads.run import load
from mrlfads.utils import Batch

class ScaledLinear(nn.Linear):
    """Linear layer with scaled weight initialization."""
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class NormLinear(ScaledLinear):
    """Linear layer with row-normalized weights."""
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
            self.bias_ih.data[-hidden_size:] = 0.0
            nn.init.zeros_(self.bias_hh)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)

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
        assert dim is not None, '`dim` cannot be none when using `reduction`=mean, seq or batch.'
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
            log_input=True, # default should already be True
        )

    def unbind(self, params): return params

    def mean(self, params): return torch.exp(params)

    def rsquared(self, data, params, mode="mcfadden", eps=1e-8):
        """Compute reconstruction R² relative to a null model."""
        data = data.float()
        B, T, Fdim = data.shape

        # Dynamic threshold: require ~10 expected events total per feature
        # This prevents near-zero values for inflating null model performance
        rate_threshold = 10.0 / (B * T)
        feature_mean = data.mean(dim=(0, 1))
        active = feature_mean > rate_threshold # mask out near-silent features
        if not active.any(): # if no active features, return NaN
            return float("nan")
        data_a = data[..., active]
        params_a = params[..., active]

        B, T, _ = data_a.shape
        mean_null = data_a.mean(dim=(0, 1), keepdim=True).expand(B, T, -1)
        log_mean_null = torch.log(mean_null.clamp_min(eps))

        if mode == "mcfadden":
            nll_model = F.poisson_nll_loss(
                params_a,
                data_a,
                log_input=True,
                full=False,
                reduction="sum",
            )
            nll_null = F.poisson_nll_loss(
                log_mean_null,
                data_a,
                log_input=True,
                full=False,
                reduction="sum",
            )
            nll_null = torch.clamp(nll_null, min=1e-6)
            return (1.0 - nll_model / nll_null).item()

        elif mode == "standard":
            return 0.0

        else:
            raise ValueError(
                f"Unknown R squared mode: {mode}. Choose from ['mcfadden']"
            )

class Gaussian:
    def __init__(self):
        self.name = "gaussian"
        self.n_params = 2

    def __call__(self, data: torch.Tensor, params: torch.Tensor):
        means, logvars = self.unbind(params)
        var = logvars.exp()
        return F.gaussian_nll_loss(
            input=means,
            target=data,
            var=var,
            reduction="none",
            full=True,
            eps=1e-6,
        )

    def rsquared(self, data: torch.Tensor, params: torch.Tensor, mode: str = "standard"):
        """Compute reconstruction R² relative to a null model."""
        means_model, _ = self.unbind(params)

        # Null mean + variance across batch/time (B, T, neurons)
        B, T = data.shape[0], data.shape[1]
        mean_null = data.mean(dim=(0, 1), keepdim=True).expand(B, T, -1)
        var_null = (
            data.var(dim=(0, 1), keepdim=True, unbiased=False)
            .clamp_min(1e-12)
            .expand(B, T, -1)
        )

        if mode == "standard":
            # Standard R²: 1 - SSE_model / SSE_null
            sse_model = ((data - means_model) ** 2).sum()
            sse_null = ((data - mean_null) ** 2).sum()
            return (1.0 - sse_model / sse_null).item()

        if mode == "mcfadden":
            # McFadden-style pseudo-R² using full NLL
            nll_model = self(data, params).sum()
            nll_null = F.gaussian_nll_loss(
                input=mean_null,
                target=data,
                var=var_null,
                reduction="sum",
                full=True,
                eps=1e-6,
            )
            return (1.0 - nll_model / nll_null).item()

        raise ValueError(f"Unknown R² mode: {mode}. Choose from ['standard', 'mcfadden'].")

    def unbind(self, params: torch.Tensor):
        means, logvars = torch.chunk(params, 2, dim=-1)
        return means, logvars

    def mean(self, params: torch.Tensor):
        return self.unbind(params)[0]
