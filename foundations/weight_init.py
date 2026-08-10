import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        weights = torch.randn(fan_out, fan_in) * ((2. / (fan_in + fan_out)) ** 0.5)
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        weights = torch.randn(fan_out, fan_in) * ((2. / fan_in) ** 0.5)
        return torch.round(weights, decimals=4).tolist()
    
    def rand_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        weights = torch.randn(fan_out, fan_in)
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights = []
        stds = []
        for num_layer in range(num_layers):
            curr_in = input_dim if num_layer == 0 else hidden_dim
            match init_type:
                case "xavier":
                    scaling_factor = (2. / (curr_in + hidden_dim)) ** 0.5
                case "kaiming":
                    print("kaiming")
                    scaling_factor = (2. / curr_in) ** 0.5
                case "random":
                    scaling_factor = 1.0
            weights.append(torch.randn(hidden_dim, curr_in) * scaling_factor)
        
        x = torch.randn(1, input_dim)
        for w in weights:
            x = torch.relu(x @ w.T)
            stds.append(round(x.std().item(), 2))

        return stds


