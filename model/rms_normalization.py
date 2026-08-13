import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        
        x_arr = np.array(x)
        g = np.array(gamma)

        rms = np.sqrt(np.mean(x_arr**2) + eps)
        out = gamma * x_arr / rms

        return np.round(out, decimals=4).tolist()