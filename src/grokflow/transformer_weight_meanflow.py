from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .embedding_meanflow import Attention, FeedForward
from .meanflow_objective import ConditionalLatentRound, ScalarCondition
from .transformer_weight_dataset import TRANSFORMER_WEIGHT_NAMES


@dataclass(frozen=True)
class TransformerWeightGeneratorConfig:
    row_width: int = 128
    row_count: int = 1_536
    attention_heads: int = 4
    head_width: int = 32
    width: int = 240
    latent_tokens: int = 64
    latent_depth: int = 4
    heads: int = 8
    condition_frequencies: int = 16


def canonical_tensor_and_row_ids() -> tuple[torch.Tensor, torch.Tensor]:
    tensor_ids = []
    row_ids = []
    for tensor_id, name in enumerate(TRANSFORMER_WEIGHT_NAMES):
        count = 512 if name in {"mlp_in.weight", "mlp_out.weight"} else 128
        tensor_ids.extend([tensor_id] * count)
        row_ids.extend(range(count))
    return torch.tensor(tensor_ids, dtype=torch.long), torch.tensor(
        row_ids, dtype=torch.long
    )


class ConditionalTransformerWeightGenerator(nn.Module):
    """Permutation-aware iMF over coupled attention heads and MLP neurons.

    Four Q/K/V row blocks and their matching W_O column blocks form four
    unordered attention-head tokens.  Each MLP input row and matching output
    column form one of 512 unordered neuron tokens.  There are deliberately no
    learned head or neuron positions: permuting a functionally equivalent head
    or hidden neuron permutes the output in exactly the same way.
    """

    def __init__(
        self,
        config: TransformerWeightGeneratorConfig = TransformerWeightGeneratorConfig(),
    ):
        super().__init__()
        self.config = config
        self.attention_projection = nn.Linear(
            4 * config.head_width * config.row_width, config.width
        )
        self.mlp_projection = nn.Linear(2 * config.row_width, config.width)
        self.structure_identity = nn.Parameter(torch.randn(2, config.width) * 0.02)
        self.segment_identity = nn.Parameter(torch.randn(2, config.width) * 0.02)
        self.condition = ScalarCondition(
            config.width, values=6, frequencies=config.condition_frequencies
        )
        self.latents = nn.Parameter(
            torch.randn(config.latent_tokens, config.width) * 0.02
        )
        self.rounds = nn.ModuleList(
            [
                ConditionalLatentRound(config.width, config.heads)
                for _ in range(config.latent_depth)
            ]
        )
        self.query_norm = nn.LayerNorm(config.width)
        self.latent_norm = nn.LayerNorm(config.width)
        self.decoder_attention = Attention(config.width, config.heads)
        self.decoder_feed_forward = FeedForward(config.width)
        attention_features = 4 * config.head_width * config.row_width
        self.u_attention_head = nn.Linear(config.width, attention_features, bias=False)
        self.v_attention_head = nn.Linear(config.width, attention_features, bias=False)
        self.u_mlp_head = nn.Linear(config.width, 2 * config.row_width, bias=False)
        self.v_mlp_head = nn.Linear(config.width, 2 * config.row_width, bias=False)
        for head in (
            self.u_attention_head,
            self.v_attention_head,
            self.u_mlp_head,
            self.v_mlp_head,
        ):
            nn.init.normal_(head.weight, std=1e-3)

    def _pack_symmetry_tokens(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, o = [rows[:, offset : offset + 128] for offset in (0, 128, 256, 384)]
        mlp_in = rows[:, 512:1024]
        mlp_out_columns = rows[:, 1024:1536]
        head_tokens = []
        for head in range(self.config.attention_heads):
            start = head * self.config.head_width
            end = start + self.config.head_width
            # W_O is stored in ordinary [output, input] orientation, so its
            # matching input-channel columns are transposed into 32 row vectors.
            o_channels = o[:, :, start:end].transpose(1, 2)
            coupled = torch.cat(
                (q[:, start:end], k[:, start:end], v[:, start:end], o_channels), dim=1
            )
            head_tokens.append(coupled.flatten(1))
        attention = torch.stack(head_tokens, dim=1)
        mlp = torch.cat((mlp_in, mlp_out_columns), dim=-1)
        return attention, mlp

    def _unpack_symmetry_tokens(
        self, attention: torch.Tensor, mlp: torch.Tensor
    ) -> torch.Tensor:
        batch = attention.shape[0]
        coupled = attention.reshape(
            batch,
            self.config.attention_heads,
            4 * self.config.head_width,
            self.config.row_width,
        )
        q = coupled[:, :, 0:32].reshape(batch, 128, 128)
        k = coupled[:, :, 32:64].reshape(batch, 128, 128)
        v = coupled[:, :, 64:96].reshape(batch, 128, 128)
        o_channels = coupled[:, :, 96:128].reshape(batch, 128, 128)
        o = o_channels.transpose(1, 2)
        mlp_in, mlp_out_columns = mlp.split(self.config.row_width, dim=-1)
        return torch.cat((q, k, v, o, mlp_in, mlp_out_columns), dim=1)

    def _encode(
        self, rows: torch.Tensor, segment: int, condition: torch.Tensor
    ) -> torch.Tensor:
        attention, mlp = self._pack_symmetry_tokens(rows)
        attention_tokens = (
            self.attention_projection(attention)
            + self.structure_identity[0][None, None]
            + self.segment_identity[segment][None, None]
            + condition[:, None]
        )
        mlp_tokens = (
            self.mlp_projection(mlp)
            + self.structure_identity[1][None, None]
            + self.segment_identity[segment][None, None]
            + condition[:, None]
        )
        return torch.cat((attention_tokens, mlp_tokens), dim=1)

    def features(
        self,
        z: torch.Tensor,
        start_weights: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        start_step: torch.Tensor,
        end_step: torch.Tensor,
        start_error: torch.Tensor,
        target_error: torch.Tensor,
    ) -> torch.Tensor:
        if z.shape[1:] != (self.config.row_count, self.config.row_width):
            raise ValueError(f"Expected [batch, 1536, 128], found {tuple(z.shape)}")
        condition = self.condition(
            r, t, start_step, end_step, start_error, target_error
        )
        noised = self._encode(z, 0, condition)
        start = self._encode(start_weights, 1, condition)
        tokens = torch.cat((noised, start), dim=1)
        latents = self.latents[None].expand(z.shape[0], -1, -1) + condition[:, None]
        for block in self.rounds:
            latents = block(latents, tokens)
        decoded = self.decoder_attention(
            self.query_norm(noised), self.latent_norm(latents)
        )
        return decoded + self.decoder_feed_forward(self.query_norm(decoded))

    def _decode_velocity(
        self,
        features: torch.Tensor,
        attention_head: nn.Linear,
        mlp_head: nn.Linear,
    ) -> torch.Tensor:
        attention = attention_head(features[:, : self.config.attention_heads])
        mlp = mlp_head(features[:, self.config.attention_heads :])
        return self._unpack_symmetry_tokens(attention, mlp)

    def predict_uv(self, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(*args)
        return (
            self._decode_velocity(features, self.u_attention_head, self.u_mlp_head),
            self._decode_velocity(features, self.v_attention_head, self.v_mlp_head),
        )

    def predict_u(self, *args: torch.Tensor) -> torch.Tensor:
        return self._decode_velocity(
            self.features(*args), self.u_attention_head, self.u_mlp_head
        )
