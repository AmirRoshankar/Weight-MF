from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import ConditionEmbedding


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.d_head = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, query_length, width = query.shape
        context_length = context.shape[1]
        q = (
            self.q(query)
            .view(batch, query_length, self.heads, self.d_head)
            .transpose(1, 2)
        )
        k = (
            self.k(context)
            .view(batch, context_length, self.heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.v(context)
            .view(batch, context_length, self.heads, self.d_head)
            .transpose(1, 2)
        )
        weights = F.softmax((q @ k.transpose(-1, -2)) / self.d_head**0.5, dim=-1)
        output = (
            (weights @ v).transpose(1, 2).contiguous().view(batch, query_length, width)
        )
        return self.o(output)


class FeedForward(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class EmbeddingMeanFlowConfig:
    vocab_size: int = 114
    embedding_width: int = 128
    width: int = 256
    latent_tokens: int = 64
    latent_depth: int = 4
    heads: int = 8


class LatentRound(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.latent_norm = nn.LayerNorm(width)
        self.cross_attention = Attention(width, heads)
        self.self_attention = Attention(width, heads)
        self.feed_forward = FeedForward(width)

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        latents = latents + self.cross_attention(
            self.latent_norm(latents), self.token_norm(tokens)
        )
        normalized = self.latent_norm(latents)
        latents = latents + self.self_attention(normalized, normalized)
        return latents + self.feed_forward(self.latent_norm(latents))


class TokenEmbeddingMeanFlow(nn.Module):
    """Structured MeanFlow backbone operating only on the 114 x 128 token embedding matrix."""

    def __init__(self, config: EmbeddingMeanFlowConfig = EmbeddingMeanFlowConfig()):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.embedding_width, config.width)
        self.token_identity = nn.Parameter(
            torch.randn(config.vocab_size, config.width) * 0.02
        )
        self.condition = ConditionEmbedding(config.width)
        self.latents = nn.Parameter(
            torch.randn(config.latent_tokens, config.width) * 0.02
        )
        self.rounds = nn.ModuleList(
            [
                LatentRound(config.width, config.heads)
                for _ in range(config.latent_depth)
            ]
        )
        self.query_norm = nn.LayerNorm(config.width)
        self.latent_norm = nn.LayerNorm(config.width)
        self.decoder_attention = Attention(config.width, config.heads)
        self.output_projection = nn.Linear(
            config.width, config.embedding_width, bias=False
        )
        nn.init.zeros_(self.output_projection.weight)

    def forward_with_diagnostics(
        self,
        embedding: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        start_step: torch.Tensor,
        end_step: torch.Tensor,
        ablate_global: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        condition = self.condition(r, t, start_step, end_step)
        queries = (
            self.input_projection(embedding)
            + self.token_identity[None]
            + condition[:, None]
        )
        latents = (
            self.latents[None].expand(embedding.shape[0], -1, -1) + condition[:, None]
        )
        for round_module in self.rounds:
            latents = round_module(latents, queries)
        if ablate_global:
            latents = torch.zeros_like(latents)
        # No residual path: all predicted velocity must pass through global cross-attention.
        decoded = self.decoder_attention(
            self.query_norm(queries), self.latent_norm(latents)
        )
        velocity = self.output_projection(decoded)
        return velocity, {
            "latent_variance": latents.var(dim=(0, 1)).mean(),
            "velocity_rms": velocity.square().mean().sqrt(),
        }

    def forward(
        self,
        embedding: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        start_step: torch.Tensor,
        end_step: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(embedding, r, t, start_step, end_step)[0]


def embedding_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mse = F.mse_loss(prediction, target)
    row_cosine = 1 - F.cosine_similarity(prediction, target, dim=-1).mean()
    global_cosine = (
        1 - F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=-1).mean()
    )
    loss = mse + cosine_weight * (0.5 * row_cosine + 0.5 * global_cosine)
    return loss, {
        "mse": mse.detach(),
        "row_cosine_loss": row_cosine.detach(),
        "global_cosine_loss": global_cosine.detach(),
    }
