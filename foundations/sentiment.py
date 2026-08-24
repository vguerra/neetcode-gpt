import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.emb = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(in_features=16, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        x = self.emb(x) # (B, T, embed_dim)
        x = torch.mean(x, dim=-2)
        x = self.linear(x)
        return self.sigmoid(x)
