import math
import torch
from torch import nn


class ConditionEmbedding(nn.Module):
    def __init__(self, width: int, frequencies: int = 8):
        super().__init__()
        self.frequencies = frequencies
        in_features = 4 * (1 + 2 * frequencies)
        self.net = nn.Sequential(
            nn.Linear(in_features, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(
        self, r: torch.Tensor, t: torch.Tensor, start: torch.Tensor, end: torch.Tensor
    ) -> torch.Tensor:
        values = torch.cat([r, t, start, end], dim=-1)
        features = [values]
        for k in range(self.frequencies):
            frequency = (2.0**k) * math.pi
            features.extend(
                [torch.sin(values * frequency), torch.cos(values * frequency)]
            )
        return self.net(torch.cat(features, dim=-1))
