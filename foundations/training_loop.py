import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        # weight init
        w = np.zeros(X.shape[1])
        b = 0.0
        n = X.shape[0]
        
        for _ in range(epochs):
            y_hat = X @ w + b
            residual = y_hat - y

            # L = (1/n) np.sum(residual ** 2)
            gradW = 2. * X.T @ residual / n
            gradB = 2. * np.mean(residual)
            

            w = w - lr * gradW
            b = b - lr * gradB

        return np.round(w, 5), np.round(b, 5)




