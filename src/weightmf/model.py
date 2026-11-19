import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import TaskConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (
            self._heads(self.W_Q(x)),
            self._heads(self.W_K(x)),
            self._heads(self.W_V(x)),
        )
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.causal_mask, torch.finfo(scores.dtype).min)
        attended = F.softmax(scores, dim=-1) @ v
        attended = attended.transpose(1, 2).contiguous().view(x.shape)
        return self.W_O(attended)


class TaskTransformer(nn.Module):
    """The one-layer, no-LayerNorm transformer from Nanda et al. (2023)."""

    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(config.seq_len, config.d_model)
        )
        self.attention = MultiHeadSelfAttention(
            config.d_model, config.n_heads, config.seq_len
        )
        self.mlp_in = nn.Linear(config.d_model, config.d_mlp, bias=False)
        self.mlp_out = nn.Linear(config.d_mlp, config.d_model, bias=False)
        self.unembed = nn.Linear(config.d_model, config.modulus, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.token_embedding.weight, std=1 / math.sqrt(self.config.d_model)
        )
        nn.init.normal_(self.position_embedding, std=1 / math.sqrt(self.config.d_model))
        for module in [
            self.attention.W_Q,
            self.attention.W_K,
            self.attention.W_V,
            self.attention.W_O,
            self.mlp_in,
            self.mlp_out,
            self.unembed,
        ]:
            nn.init.xavier_normal_(module.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = self.token_embedding(tokens) + self.position_embedding
        residual = residual + self.attention(residual)
        residual = residual + self.mlp_out(F.relu(self.mlp_in(residual)))
        return self.unembed(residual[:, -1, :])
