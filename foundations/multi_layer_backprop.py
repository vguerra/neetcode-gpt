import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x_arr = np.array(x)
        N = 1
        y_true_arr = np.array(y_true)
        W1_arr = np.array(W1)
        W2_arr = np.array(W2)
        b1_arr = np.array(b1)
        b2_arr = np.array(b2)

        z1 =  x_arr[None, :] @ W1_arr.T + b1_arr[None, :]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ W2_arr.T + b2_arr[None, :]

        residuals = z2 - y_true_arr

        L = np.mean(residuals ** 2, axis=1)

        dz2 = 2 * residuals / N # (1, 1)
        dW2 = dz2 @ a1 # (1, 2)
        db2 = dz2[0] # (1, 1)

        da1 = dz2 @ W2_arr # (1, 2)
        dz1 = da1 * (z1 > 0).astype(np.float32) # (1, 2)

        dW1 = dz1.T @ x_arr[None, :]
        db1 = dz1[0]


        return {
            'loss': round(L.item(), 4),
            'dW2': np.round(dW2, decimals=4),
            'db2': np.round(db2, decimals=4),
            'dW1': np.round(dW1, decimals=4),
            'db1': np.round(db1, decimals=4),
        }
