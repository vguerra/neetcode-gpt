import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.linear1 = nn.Linear(in_features=784, out_features=512)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        logits = self.linear2(self.dropout(torch.relu(self.linear1(images))))
        return torch.round(torch.sigmoid(logits), decimals=4)

