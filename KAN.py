import torch
import numpy as np
from spline import *
from kan_layer import *
import torch.nn as nn
import torch.nn.functional as F
import math

class KAN(nn.Module):
    def __init__(
            self,
            layers,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers, layers[1:]):
            self.layers.append(
                kan_layer(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    noise_scale=scale_noise,
                    scale_base=scale_base,
                    scale_sp=scale_spline,
                    base_func=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x, _ = layer(x)

        return x
    