from __future__ import annotations

import math

import torch

from .config import TaskConfig
from .data import ModularAdditionData


SUPPORTED_OPERATIONS = (
    "addition",
    "multiplication",
    "subtraction",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "division",
    "quadratic_symmetric",
    "quadratic_asymmetric",
    "cubic_asymmetric",
    "cosine_sum_quantized",
)


def _operation_labels(
    a: torch.Tensor, b: torch.Tensor, p: int, operation: str
) -> torch.Tensor:
    if operation == "addition":
        return (a + b) % p
    if operation == "multiplication":
        return (a * b) % p
    if operation == "subtraction":
        return (a - b) % p
    # Keep the original 113-class output head unchanged.  The raw bitwise OR
    # of two values in [0, 112] can reach 127, so all three logic operations
    # are reduced modulo p for a matched-vocabulary transfer benchmark.
    if operation == "bitwise_and":
        return torch.bitwise_and(a, b) % p
    if operation == "bitwise_or":
        return torch.bitwise_or(a, b) % p
    if operation == "bitwise_xor":
        return torch.bitwise_xor(a, b) % p
    if operation == "division":
        inverses = torch.tensor([0] + [pow(value, p - 2, p) for value in range(1, p)])
        return (a * inverses[b]) % p
    if operation == "quadratic_symmetric":
        return (a.square() + a * b + b.square()) % p
    if operation == "quadratic_asymmetric":
        return (a.square() + a * b + b.square() + a) % p
    if operation == "cubic_asymmetric":
        return (a.pow(3) + a * b.square() + b) % p
    if operation == "cosine_sum_quantized":
        angles_a = 2.0 * math.pi * a.float() / p
        angles_b = 2.0 * math.pi * b.float() / p
        normalized = (torch.cos(angles_a) + torch.cos(angles_b) + 2.0) / 4.0
        return torch.round(normalized * (p - 1)).long().clamp_(0, p - 1)
    raise ValueError(f"Unsupported operation: {operation}")


def make_binary_operation(
    config: TaskConfig, split_seed: int, operation: str
) -> ModularAdditionData:
    """Build a same-shape binary-operation task for cross-task optimization tests."""
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"operation must be one of {SUPPORTED_OPERATIONS}")
    p = config.modulus
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    if operation == "division":
        keep = b != 0
        a, b = a[keep], b[keep]
    equals = torch.full_like(a, config.equals_token)
    x = torch.stack([a, b, equals], dim=1)
    y = _operation_labels(a, b, p, operation)
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(x), generator=generator)
    n_train = int(config.train_fraction * len(x))
    train_idx, val_idx = permutation[:n_train], permutation[n_train:]
    return ModularAdditionData(x[train_idx], y[train_idx], x[val_idx], y[val_idx])
