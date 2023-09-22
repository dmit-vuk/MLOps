import torch
from torch import nn
import torchvision

from typing import List

class MLPNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: List[int]
    ):
        super(MLPNN, self).__init__()

        self.mlp_nn = torchvision.ops.MLP(in_features, hidden_sizes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_nn(x)