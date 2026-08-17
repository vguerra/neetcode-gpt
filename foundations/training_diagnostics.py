import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        ans = []
        with torch.no_grad():
            for name, module in model.named_children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    deads = (torch.sum(torch.where(x <= 0, 1.0, 0.), dim=0) == x.shape[0]).float()

                    stats = {
                        'mean': round(x.detach().mean().item(), 4),
                        'std': round(x.detach().std().item(), 4),
                        'dead_fraction': round(deads.mean().item(), 4)
                    }
                    ans.append(stats)
        
        return ans

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        y_hat = model(x)
        loss = nn.MSELoss()
        output = loss(y_hat, y)
        output.backward()

        ans = []
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                for param_name, param in module.named_parameters():
                    if param_name.endswith('weight'):
                        stats = {
                            'mean': round(param.grad.mean().item(), 4),
                            'std': round(param.grad.std().item(), 4),
                            'norm': round(param.grad.norm().item(), 4)
                        }
                        ans.append(stats)
        return ans


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        # `dead_neurons` if any layer has dead_fraction > 0.5
        if any([stat['dead_fraction'] > 0.5 for stat in activation_stats]):
            return "dead_neurons"
        # `exploding_gradients` if any layer gradient norm > 1000
        if any([stat['norm'] > 1000 for stat in gradient_stats]):
            return "exploding_gradients"
        # `vanishing_gradients` if any layer gradient norm < 1e-5
        if any([stat['norm'] < 1e-5 for stat in gradient_stats]):
            return "vanishing_gradients"
        if any([stat['std'] < 0.1 for stat in activation_stats]):
            return "vanishing_gradients"
        if any([stat['std'] > 10.0 for stat in activation_stats]):
            return "exploding_gradients"
        return 'healthy'


