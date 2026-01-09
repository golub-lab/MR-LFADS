import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class RNNBase(nn.Module):
    """
    Base class for RNNs, where its type ``rnn_type`` and ``nonlinearity`` can be specified.
    The rest follows nn.RNNCell notations.
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        """
        Args:
            - input_size (int)
            - hidden_size (int)
            - rnn_type (str): type of RNN, can be rnncell, grucell, rnn or gru, default: "grucell"
            - nonlinearity (str): can be tanh or relu (note that it cannot be relu if it's gru), default: "tanh"
            - min (float): minimum value for clamp, default: None (not clamped)
            - max (float): maximum value for clamp, default: None (not clamped)
        """
        super(RNNBase, self).__init__()
        
        self.params = {
            "rnn_type": "grucell",
            "nonlinearity": "tanh",
            "min": None,
            "max": None,
            "ev_scale": 1,
        }
        self.params.update(kwargs)
        
        # Define attributes and self.model
        self.input_size, self.hidden_size = input_size, hidden_size
        self.model = get_rnn_type(self.input_size, self.hidden_size, self.params["rnn_type"], self.params["nonlinearity"])
        
    def forward(self, *inp):
        return self.model(*inp)
    
    def _init_spectral_weight(self):
        if self.params["rnn_type"] != "rnncell": print("WARNING: rnn_type is not RNNcell.")
        
        self.model.weight_hh.data = self.normalize_eigenvalues(self.model.weight_hh.data, self.params["ev_scale"])
    
    @staticmethod
    def normalize_eigenvalues(W, target_radius):
        # Get eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(W)
        spectral_radius = max(np.abs(eigenvalues))

        # Scale the eigenvalues to achieve the target spectral radius
        scaling_factor = target_radius / spectral_radius
        scaled_eigenvalues = eigenvalues * scaling_factor

        # Reconstruct the weight matrix
        scaled_W = eigenvectors @ np.diag(scaled_eigenvalues) @ np.linalg.inv(eigenvectors)
        return scaled_W
    
class RNNWrapper(nn.Module):
    """
    Wrapper class for RNN (GRU, LSTM) classes so that their output matches those of their cell counterparts.
    """
    def __init__(self, model):
        """
        Args:
            - model (nn.Module subclass): a RNNCell class
        """
        super(RNNWrapper, self).__init__()
        self.model = model
        
    def forward(self, *inp):
        return self.model(*inp)[0]
    
class MLPBase(nn.Module):
    """
    Base class for MLPs.
    """
    def __init__(self, features_list):
        """
        Args:
            - features_list (list): list containing specifications of each layer, (input_dim, output_dim, nonlinearity)
        """
        super(MLPBase, self).__init__()
        
        # Define attributes
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
    
class RNNChannel(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes, output_nonlinearity, rnn_nonlinearity="tanh", override_single=False, **kwargs):
        super(RNNChannel, self).__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.override_single = override_single
        self.rnn = RNNBase(input_size, hidden_size, nonlinearity=rnn_nonlinearity, **kwargs)
        self.output = MLPBase([[hidden_size, sum(output_sizes), output_nonlinearity]])

    def forward(self, *inp):
        h = self.rnn(*inp)
        m = self.output(h)
        if (len(self.output_sizes) > 1) or self.override_single:
            return h, torch.split(m, self.output_sizes, dim=-1)
        else: return h, m
       
def get_rnn_type(input_size, hidden_size, rnn_type, nonlinearity, **kwargs):
    if rnn_type == "grucell":
        return nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
    elif rnn_type == "rnncell":
        return nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
    elif rnn_type == "gru":
        model = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        return RNNWrapper(model)
    elif rnn_type == "rnn":
        model = nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity, batch_first=True)
        return RNNWrapper(model)
    elif rnn_type == "lowrankrnncell":
        model = LowRankRNNCell(input_size, hidden_size, kwargs["rank"], rnn_nonlinearity=nonlinearity)
    else:
        raise TypeError(f"``rnn_type`` cannot be {rnn_type}.")
    
def get_activation_type(activation_type):
    if activation_type == "tanh":
         return nn.Tanh()
    elif activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    else:
        raise TypeError(f"``activation_type`` cannot be {activation_type}.")
        
