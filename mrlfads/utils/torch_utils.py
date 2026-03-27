import os
import gc
import abc
import math
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchmetrics import Metric
from torch.distributions import MultivariateNormal as FullRankMultivariateNormal
from torch.distributions.transforms import AffineTransform
from torch.distributions import Independent, Normal, StudentT, kl_divergence
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

import mrlfads.paths as path
from mrlfads.run import load
from mrlfads.utils.common_utils import Batch


# ===== Linear layers ===== #

class PreInitLinear(nn.Linear):
    def __init__(
        self,
        filepath,
        in_features,
        out_features,
        bias=True,
    ):
        super().__init__(in_features, out_features, bias=bias)

        weight = np.load(filepath)
        assert weight.shape == (out_features, in_features)

        w = torch.from_numpy(weight).to(self.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(w)

class ScaledLinear(nn.Linear):
    """Linear layer with scaled weight initialization."""

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class NormLinear(nn.Linear):
    def forward(self, x):
        weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, weight, self.bias)
    
def fanin_normal_init_(layer):
    nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.in_features))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

class LowRankLinear(nn.Module):
    """Linear layer with rank-r weight factorization: W = U @ V."""
    def __init__(
        self,
        in_features,
        out_features,
        rank: int = 1,
        bias: bool = True,
        update_scale_init: float = 1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.std = update_scale_init

        self.weights_u = nn.Parameter(torch.empty(out_features, rank))
        self.weights_v = nn.Parameter(torch.empty(rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weights_u, std=self.std)
        nn.init.normal_(self.weights_v, std=self.std)
        
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        weight = self.weights_u @ self.weights_v
        return F.linear(x, weight, self.bias)
    
    def get_weight(self):
        return self.weights_u @ self.weights_v
    
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
                layers[f"{activation_type}{i}"] = getattr(nn, activation_type)
        self.model = nn.Sequential(layers)
        
    def forward(self, *inp):
        return self.model(*inp)
    
    
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=0,   # important: pad manually for causality
            bias=bias
        )

    def forward(self, x):
        # x: (B, C, T)
        x = F.pad(x, (self.left_pad, 0))  # (pad_left, pad_right)
        return self.conv(x)

class Conv1dBase(nn.Module):
    def __init__(self, layer_spec):
        super().__init__()
        layers = []

        for in_ch, out_ch, k, act in layer_spec:
            layers.append(CausalConv1d(in_ch, out_ch, k))

            if act:
                layers.append(getattr(nn, act)())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, channels, time)
        return self.model(x)
    
# ===== Recurrent layers ===== #
    
class GRUCellBase(nn.GRUCell):
    def __init__(
        self,
        input_size,
        hidden_size,
        clip_value=float("inf"),
        shared_scale=False,
    ):
        super().__init__(input_size, hidden_size, bias=True)
        self.bias_hh.requires_grad = False
        self.clip_value = clip_value
        self._reset_parameters(shared_scale)

    def _reset_parameters(self, shared_scale):
        if shared_scale:
            ih_scale = self.input_size + self.hidden_size
            hh_scale = self.input_size + self.hidden_size
        else:
            ih_scale = self.input_size
            hh_scale = self.hidden_size

        nn.init.normal_(self.weight_ih, std=1 / math.sqrt(ih_scale))
        nn.init.normal_(self.weight_hh, std=1 / math.sqrt(hh_scale))

        nn.init.ones_(self.bias_ih)
        self.bias_ih.data[-self.hidden_size:] = 0.0
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
        hidden = torch.clamp(hidden, -self.clip_value, self.clip_value)
        return hidden

class GRUBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.cell = GRUCellBase(
            input_size, hidden_size, clip_value=clip_value, shared_scale=True
        )

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden

class BiGRUBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.fwd_gru = GRUBase(input_size, hidden_size, clip_value=clip_value)
        self.bwd_gru = GRUBase(input_size, hidden_size, clip_value=clip_value)

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        h0_fwd, h0_bwd = h_0
        input_fwd = input
        input_bwd = torch.flip(input, [1])
        output_fwd, hn_fwd = self.fwd_gru(input_fwd, h0_fwd)
        output_bwd, hn_bwd = self.bwd_gru(input_bwd, h0_bwd)
        output_bwd = torch.flip(output_bwd, [1])
        output = torch.cat([output_fwd, output_bwd], dim=2)
        h_n = torch.stack([hn_fwd, hn_bwd])
        return output, h_n
    
# ===== Distributions ===== #  
    
class MultivariateNormal(nn.Module):
    def __init__(
        self,
        mean: float,
        variance: float,
        shape: int,
        var_requires_grad: bool = True,
    ):
        super().__init__()
        means = torch.ones(shape) * mean
        logvars = torch.log(torch.ones(shape) * variance)
        self.mean = nn.Parameter(means, requires_grad=False)
        self.logvar = nn.Parameter(logvars, requires_grad=var_requires_grad)

    def make_posterior(self, post_mean, post_std):
        post_mean = torch.nan_to_num(post_mean, posinf=1e3, neginf=-1e3)
        post_std = torch.nan_to_num(post_std, posinf=1e3, neginf=-1e3) + 1e-2
        return Independent(Normal(post_mean, post_std), 1)

    def forward(self, post_mean, post_std):
        posterior = self.make_posterior(post_mean, post_std)
        
        prior_std = torch.exp(0.5 * self.logvar)
        prior = Independent(Normal(self.mean, prior_std), 1)
        
        kl_batch = kl_divergence(posterior, prior)
        # If kl_batch.shape has a time component, sum across that dimension
        if len(kl_batch.shape) > 1:
            kl_batch = torch.sum(kl_batch, dim=1)
        return torch.mean(kl_batch)
    
    def kl_divergence_by_component(self, post_mean, post_std, com_dim, tpe="mean"):
        post_mean = torch.split(post_mean, com_dim, dim=2)
        post_std = torch.split(post_std, com_dim, dim=2)
        prior_std = torch.split(torch.exp(0.5 * self.logvar), com_dim)
        prior_mean = torch.split(self.mean, com_dim)
        
        kls = []
        for i in range(len(post_mean)):
            posterior = self.make_posterior(post_mean[i], post_std[i])
            prior = Independent(Normal(prior_mean[i], prior_std[i]), 1)
            
            if tpe == "mean":
                kls.append(kl_divergence(posterior, prior).mean().item())
            elif tpe == "seq":
                kls.append(kl_divergence(posterior, prior).mean(dim=0))
        return kls

class Poisson:
    def __init__(self):
        self.n_params = 1
        self.name = "poisson"
        
    def __call__(self, data, output_params):
        """Performs reshape, then computes loss."""
        return self.compute_loss(data, self.unbind(output_params))

    def unbind(self, output_params):
        return torch.unsqueeze(output_params, dim=-1)
    
    def mean(self, x): return torch.exp(x)

    def compute_loss(self, data, output_params):
        return F.poisson_nll_loss(
            output_params[..., 0],
            data,
            full=True,
            reduction="none",
        )

    def compute_means(self, output_params):
        return torch.exp(output_params[..., 0])
    
    def pseudo_r2(self, data, output_params):
        nll_model = self(data, output_params)
        nll_null = F.poisson_nll_loss(
            data,
            torch.mean(
                data.to(torch.float), dim=(0, 1), keepdim=True).repeat(
                    *list(data.shape[:2]) + [1]
                ) + 1e-16,
            full=True,
            reduction="none",
        )
        return (1 - nll_model / nll_null).mean().item()
        
class Gaussian:
    def __init__(self):
        self.n_params = 2
        self.name = "gaussian"
        
    def __call__(self, data, output_params):
        """Performs reshape, then computes loss."""
        return self.compute_loss(data, self.unbind(output_params))

    def unbind(self, output_params):
        means, logvars = torch.chunk(output_params, 2, -1)
        return torch.stack([means, logvars], -1)

    def compute_loss(self, data, output_params):
        means, logvars = torch.unbind(output_params, axis=-1)
        recon_all = F.gaussian_nll_loss(
            input=means, target=data, var=torch.exp(logvars), reduction="none"
        )
        return recon_all

    def compute_means(self, output_params):
        return output_params[..., 0]
    
    def pseudo_r2(self, data, output_params):
        nll_model = self(data, output_params)
        mean_null = data.mean(dim=(0,1), keepdim=True)
        var_null = data.var(dim=(0,1), keepdim=True).clamp_min(1e-8)
        
        nll_null = F.gaussian_nll_loss(
            input=mean_null.expand_as(data),
            target=data,
            var=var_null.expand_as(data),
            full=True,
            reduction="none",
        )
        return (1 - nll_model / nll_null).mean().item()
    
# ===== Metrics ===== #

class EMAMetric(Metric):
    """
    Exponentially smoothed metric computed from batch-weighted averages.
    """
    def __init__(self, momentum: float = 0.9, **kwargs):
        super().__init__(**kwargs)

        self.momentum = momentum
        self.add_state("total", default=torch.tensor(0.0))
        self.add_state("n_samples", default=torch.tensor(0))
        self._ema_state = torch.tensor(float("nan"))

    def update(self, metric_value: torch.Tensor, batch_n: int) -> None:
        self.total += metric_value * batch_n
        self.n_samples += batch_n

    def compute(self) -> torch.Tensor:
        current_mean = self.total / self.n_samples
        if torch.isnan(self._ema_state):
            self._ema_state = current_mean

        smoothed_value = (
            self.momentum * self._ema_state
            + (1.0 - self.momentum) * current_mean
        )
        self._ema_state = smoothed_value
        return smoothed_value
    
def det(x):
    return x.detach() if torch.is_tensor(x) else x

