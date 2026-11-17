from dataclasses import dataclass

import torch

from .config import GrokConfig


@dataclass
class ModularAdditionData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor


def make_modular_addition(config: GrokConfig, split_seed: int) -> ModularAdditionData:
    """Create every ordered pair (a,b), then make a deterministic 30/70 split."""
    p = config.modulus
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    equals = torch.full_like(a, config.equals_token)
    x = torch.stack([a, b, equals], dim=1)
    y = (a + b) % p

    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(x), generator=generator)
    n_train = int(config.train_fraction * len(x))
    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:]
    return ModularAdditionData(x[train_idx], y[train_idx], x[val_idx], y[val_idx])
