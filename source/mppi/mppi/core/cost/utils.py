import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from typing import Callable, List


def resolve_nn_activation(act_name: str, lrelu_negative_slope: float = 0.01) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU(negative_slope=lrelu_negative_slope)
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def fcnn_factory(input_dim: int, hidden_dims: List[int],
                 activation: Callable, output_dim: int = 1,
                 squash_output: bool = False,
                 use_spectral_norm: bool = False,
                 use_dropout: bool = False,
                 use_layer_norm: bool = False,
                 use_batch_norm: bool = False,
                 batch_norm_track_running_stats: bool = True,
                 dropout_rate: float = 0.5,
                 lrelu_negative_slope: float = 0.01,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float32,
                 ) -> torch.nn.Sequential:
    activation = resolve_nn_activation(activation, lrelu_negative_slope=lrelu_negative_slope)
    layers = []

    def add_layer(layers, layer, use_spectral_norm, use_layer_norm, use_batch_norm, layer_dim=None):
        l = spectral_norm(layer) if use_spectral_norm else layer
        layers.append(l)
        if use_batch_norm and layer_dim is not None:
            layers.append(nn.BatchNorm1d(layer_dim, track_running_stats=batch_norm_track_running_stats))
        if use_layer_norm and layer_dim is not None:
            layers.append(nn.LayerNorm(layer_dim))

    # Input layer
    add_layer(
        layers, nn.Linear(input_dim, hidden_dims[0]),
        use_spectral_norm, use_layer_norm, use_batch_norm, layer_dim=hidden_dims[0]
    )
    layers.append(activation)
    if use_dropout:
        layers.append(nn.Dropout(dropout_rate))

    # Hidden layers
    for layer_index in range(len(hidden_dims)):
        is_last = layer_index == len(hidden_dims) - 1
        out_dim = output_dim if is_last else hidden_dims[layer_index + 1]
        # Don't apply normalization to output layer
        add_layer(
            layers, nn.Linear(hidden_dims[layer_index], out_dim), 
            use_spectral_norm, 
            use_layer_norm=use_layer_norm and not is_last, 
            use_batch_norm=use_batch_norm and not is_last,
            layer_dim=out_dim
        )
        if not is_last:
            layers.append(activation)
        if not is_last and use_dropout:
            layers.append(nn.Dropout(dropout_rate))

    # Optional sigmoid output
    if squash_output:
        layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)
    model.to(device=device, dtype=dtype)
    return model
