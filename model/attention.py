import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.WK = nn.Linear(out_features=attention_dim, in_features=embedding_dim, bias=False)
        self.WQ = nn.Linear(out_features=attention_dim, in_features=embedding_dim, bias=False)
        self.WV = nn.Linear(out_features=attention_dim, in_features=embedding_dim, bias=False)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.WK(embedded)
        Q = self.WQ(embedded)
        V = self.WV(embedded)

        d_model = K.size(2)
        attn_scores = Q@K.transpose(-2, -1) / math.sqrt(d_model)
        causal_mask = torch.tril(torch.ones_like(attn_scores), diagonal=0)
        attn_scores.masked_fill_(causal_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1) @ V

        return attn_weights



