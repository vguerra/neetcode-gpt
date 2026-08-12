import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x_arr = np.array(x)
        g = np.array(gamma)
        b = np.array(beta)
        r_mean = np.array(running_mean)
        r_var = np.array(running_var)

        if training:
            mean = np.mean(x_arr, axis=0, keepdims=True)
            var = np.mean((x_arr - mean)**2, axis=0, keepdims=True)
            r_mean = (1 - momentum) * r_mean + momentum * mean.squeeze()
            r_var = (1 - momentum) * r_var + momentum * var.squeeze()
        else:
            mean = r_mean
            var = r_var
        x = (x_arr - mean) / np.sqrt(var + eps)
        y = g * x + b

        return np.round(y, decimals=4).tolist(), np.round(r_mean, decimals=4).tolist(), np.round(r_var, decimals=4).tolist()
