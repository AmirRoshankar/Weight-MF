from __future__ import annotations
import math
import torch
from torch import nn
from .embedding_meanflow import Attention, FeedForward


class ScalarCondition(nn.Module):
    """Fourier conditioning for time, checkpoint position, and task metric prompts."""

    def __init__(self, width: int, values: int = 6, frequencies: int = 16):
        super().__init__()
        self.frequencies = frequencies
        self.net = nn.Sequential(
            nn.Linear(values * (1 + 2 * frequencies), width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        scalars = torch.cat(values, dim=-1)
        features = [scalars]
        for exponent in range(self.frequencies):
            frequency = math.pi * (2.0**exponent)
            features.extend(
                (torch.sin(frequency * scalars), torch.cos(frequency * scalars))
            )
        return self.net(torch.cat(features, dim=-1))


class ConditionalLatentRound(nn.Module):
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


def _adaptive_loss(per_example: torch.Tensor, p: float, epsilon: float) -> torch.Tensor:
    weight = (per_example.detach() + epsilon).pow(p)
    return (per_example / weight).mean()


def improved_meanflow_loss(
    model: nn.Module,
    early: torch.Tensor,
    late: torch.Tensor,
    start_step: torch.Tensor,
    end_step: torch.Tensor,
    start_error: torch.Tensor,
    target_error: torch.Tensor,
    data_proportion: float = 0.5,
    logit_mean: float = -0.4,
    logit_std: float = 1.0,
    adaptive_p: float = 1.0,
    adaptive_epsilon: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = late.shape[0]
    first = torch.sigmoid(
        torch.randn(batch, 1, device=late.device) * logit_std + logit_mean
    )
    second = torch.sigmoid(
        torch.randn(batch, 1, device=late.device) * logit_std + logit_mean
    )
    t, r = torch.maximum(first, second), torch.minimum(first, second)
    boundary_count = int(batch * data_proportion)
    if boundary_count:
        r = torch.cat((t[:boundary_count], r[boundary_count:]), dim=0)
    expanded_t = t[:, :, None]
    noise = torch.randn_like(late)
    z = (1.0 - expanded_t) * late + expanded_t * noise
    target_v = noise - late

    _, predicted_v = model.predict_uv(
        z, early, r, t, start_step, end_step, start_error, target_error
    )

    def u_function(
        z_value: torch.Tensor, t_value: torch.Tensor, r_value: torch.Tensor
    ) -> torch.Tensor:
        return model.predict_u(
            z_value,
            early,
            r_value,
            t_value,
            start_step,
            end_step,
            start_error,
            target_error,
        )

    u, du_dt = torch.func.jvp(
        u_function,
        (z, t, r),
        (predicted_v.detach(), torch.ones_like(t), torch.zeros_like(r)),
    )
    compound_v = u + (t - r)[:, :, None] * du_dt.detach()
    u_squared = (compound_v - target_v).square().flatten(1).sum(dim=1)
    v_squared = (predicted_v - target_v).square().flatten(1).sum(dim=1)
    loss_u = _adaptive_loss(u_squared, adaptive_p, adaptive_epsilon)
    loss_v = _adaptive_loss(v_squared, adaptive_p, adaptive_epsilon)
    return loss_u + loss_v, {
        "loss_u_mse": (u_squared / late[0].numel()).mean().detach(),
        "loss_v_mse": (v_squared / late[0].numel()).mean().detach(),
    }


@torch.no_grad()
def sample_improved_meanflow(
    model: nn.Module,
    early: torch.Tensor,
    start_step: torch.Tensor,
    end_step: torch.Tensor,
    start_error: torch.Tensor,
    target_error: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    noise = torch.randn(
        early.shape, device=early.device, dtype=early.dtype, generator=generator
    )
    zeros = torch.zeros_like(start_step)
    ones = torch.ones_like(start_step)
    average_velocity = model.predict_u(
        noise, early, zeros, ones, start_step, end_step, start_error, target_error
    )
    return noise - average_velocity, 1
