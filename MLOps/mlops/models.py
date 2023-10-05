from typing import List

import torch
import torchvision
from torch import nn


class MLPNN(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int]):
        super(MLPNN, self).__init__()

        self.flatten = nn.Flatten()
        self.mlp_nn = torchvision.ops.MLP(in_features, hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_nn(self.flatten(x))
