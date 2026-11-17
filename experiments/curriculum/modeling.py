from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReasoningModelConfig:
    vocab_size: int = 128
    block_size: int = 1024
    n_layer: int = 5
    n_embd: int = 128
    n_head: int = 4
    intermediate_size: int = 512
    rotary_percentage: float = 0.25
    bias: bool = True
    norm_eps: float = 1e-5

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def rope_n_elem(self) -> int:
        return int(self.rotary_percentage * self.head_size)

    def to_dict(self) -> dict:
        return asdict(self)


def _rope_cache(length: int, width: int, device: torch.device | None = None):
    theta = 1.0 / (10_000 ** (torch.arange(0, width, 2, device=device).float() / width))
    positions = torch.arange(length, device=device)
    angles = torch.outer(positions, theta).repeat(1, 2)
    return angles.cos(), angles.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    width = x.size(-1)
    half = width // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos[None, None, :, :]) + (rotated * sin[None, None, :, :])


class PaperAttention(nn.Module):
    """LitGPT 0.4.12 default MHA used by the paper configuration."""

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        # LitGPT stores [Q,K,V] inside each query group/head.
        qkv = self.attn(x).view(batch, length, self.cfg.n_head, 3, self.cfg.head_size)
        q, k, v = qkv.unbind(dim=3)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        rope_width = self.cfg.rope_n_elem
        q = torch.cat(
            (_apply_rope(q[..., :rope_width], cos, sin), q[..., rope_width:]), dim=-1
        )
        k = torch.cat(
            (_apply_rope(k[..., :rope_width], cos, sin), k[..., rope_width:]), dim=-1
        )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, length, self.cfg.n_embd)
        return self.proj(y)


class PaperMLP(nn.Module):
    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=cfg.bias)
        self.proj = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class PaperBlock(nn.Module):
    """Pre-LN, parallel residual block matching LitGPT's default Config."""

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.norm_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.norm_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attn = PaperAttention(cfg)
        self.mlp = PaperMLP(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return x + self.attn(self.norm_1(x), cos, sin) + self.mlp(self.norm_2(x))


class PaperReasoningTransformer(nn.Module):
    """Dependency-light reproduction of the paper's 1M LitGPT architecture."""

    architecture_name = "paper_faithful_litgpt"

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(PaperBlock(cfg) for _ in range(cfg.n_layer))
        self.final_norm = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        cos, sin = _rope_cache(cfg.block_size, cfg.rope_n_elem)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        base_std = math.sqrt(2.0 / 5.0 / self.config.n_embd)
        residual_std = 1.0 / math.sqrt(self.config.n_embd) / self.config.n_layer
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=base_std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=residual_std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.config.block_size:
            raise ValueError(
                f"sequence length {length} exceeds {self.config.block_size}"
            )
        x = self.token_embedding(tokens)
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.final_norm(x))


class CompatibleAttention(nn.Module):
    """The exact matrix convention learned by the addition transformer iMF."""

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.n_heads = cfg.n_head
        self.d_head = cfg.head_size
        self.W_Q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.W_K = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.W_V = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.W_O = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        return x.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._heads(self.W_Q(x))
        k = self._heads(self.W_K(x))
        v = self._heads(self.W_V(x))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(x.shape)
        return self.W_O(y)


class IMFBridgeAttention(nn.Module):
    """Paper attention with separate Q/K/V modules for the addition-iMF adapter.

    Splitting LitGPT's fused QKV projection is a parameterization change only;
    LayerNorm, RoPE, biases, head ordering, and attention computation remain the
    paper defaults.
    """

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.W_K = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.W_V = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.W_O = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.cfg.n_head, self.cfg.head_size).transpose(
            1, 2
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q = self._heads(self.W_Q(x))
        k = self._heads(self.W_K(x))
        v = self._heads(self.W_V(x))
        rope_width = self.cfg.rope_n_elem
        q = torch.cat(
            (_apply_rope(q[..., :rope_width], cos, sin), q[..., rope_width:]), dim=-1
        )
        k = torch.cat(
            (_apply_rope(k[..., :rope_width], cos, sin), k[..., rope_width:]), dim=-1
        )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(x.shape)
        return self.W_O(y)


class IMFBridgeBlock(nn.Module):
    """Paper-faithful block exposing the six matrices expected by the iMF."""

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.norm_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.norm_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attention = IMFBridgeAttention(cfg)
        self.mlp_in = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=cfg.bias)
        self.mlp_out = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=cfg.bias)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return (
            x
            + self.attention(self.norm_1(x), cos, sin)
            + self.mlp_out(F.gelu(self.mlp_in(self.norm_2(x))))
        )


class IMFCompatiblePaperTransformer(nn.Module):
    """Paper-faithful forward pass with an iMF-addressable parameter layout."""

    architecture_name = "paper_faithful_imf_bridge"

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(IMFBridgeBlock(cfg) for _ in range(cfg.n_layer))
        self.final_norm = nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        cos, sin = _rope_cache(cfg.block_size, cfg.rope_n_elem)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        base_std = math.sqrt(2.0 / 5.0 / self.config.n_embd)
        residual_std = 1.0 / math.sqrt(self.config.n_embd) / self.config.n_layer
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=base_std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.attention.W_O.weight, std=residual_std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.config.block_size:
            raise ValueError(
                f"sequence length {length} exceeds {self.config.block_size}"
            )
        x = self.token_embedding(tokens)
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.final_norm(x))


class CompatibleBlock(nn.Module):
    """One no-LN, sequential, ReLU block matching the addition source model."""

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.attention = CompatibleAttention(cfg)
        self.mlp_in = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.mlp_out = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.mlp_out(F.relu(self.mlp_in(x)))


class CompatibleReasoningTransformer(nn.Module):
    """Five-layer paper-scale model whose individual blocks are iMF-compatible."""

    architecture_name = "exact_addition_source_block_ablation"

    def __init__(self, cfg: ReasoningModelConfig):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Parameter(torch.empty(cfg.block_size, cfg.n_embd))
        self.blocks = nn.ModuleList(CompatibleBlock(cfg) for _ in range(cfg.n_layer))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        width = self.config.n_embd
        nn.init.normal_(self.token_embedding.weight, std=1 / math.sqrt(width))
        nn.init.normal_(self.position_embedding, std=1 / math.sqrt(width))
        nn.init.xavier_normal_(self.lm_head.weight)
        for block in self.blocks:
            for module in (
                block.attention.W_Q,
                block.attention.W_K,
                block.attention.W_V,
                block.attention.W_O,
                block.mlp_in,
                block.mlp_out,
            ):
                nn.init.xavier_normal_(module.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.config.block_size:
            raise ValueError(
                f"sequence length {length} exceeds {self.config.block_size}"
            )
        x = self.token_embedding(tokens) + self.position_embedding[:length]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


IMF_WEIGHT_NAMES = (
    "attention.W_Q.weight",
    "attention.W_K.weight",
    "attention.W_V.weight",
    "attention.W_O.weight",
    "mlp_in.weight",
    "mlp_out.weight",
)


def compatible_block_rows(block: CompatibleBlock | IMFBridgeBlock) -> torch.Tensor:
    state = block.state_dict()
    rows = []
    for name in IMF_WEIGHT_NAMES:
        value = state[name].detach().float()
        rows.append(value.T if name == "mlp_out.weight" else value)
    return torch.cat(rows, dim=0)


def unpack_rows(rows: torch.Tensor) -> dict[str, torch.Tensor]:
    if rows.shape != (1536, 128):
        raise ValueError(f"expected [1536,128], got {tuple(rows.shape)}")
    q, k, v, o = rows[0:128], rows[128:256], rows[256:384], rows[384:512]
    mlp_in = rows[512:1024]
    mlp_out = rows[1024:1536].T
    return {
        "attention.W_Q.weight": q,
        "attention.W_K.weight": k,
        "attention.W_V.weight": v,
        "attention.W_O.weight": o,
        "mlp_in.weight": mlp_in,
        "mlp_out.weight": mlp_out,
    }


def copy_rows_to_compatible_block(
    block: CompatibleBlock | IMFBridgeBlock, rows: torch.Tensor
) -> None:
    parameters = dict(block.named_parameters())
    with torch.no_grad():
        for name, value in unpack_rows(rows).items():
            parameters[name].copy_(
                value.to(parameters[name].device, parameters[name].dtype)
            )


def copy_paper_to_bridge(
    paper: PaperReasoningTransformer, bridge: IMFCompatiblePaperTransformer
) -> None:
    """Copy a fused-QKV paper model into the mathematically equivalent bridge."""
    if paper.config != bridge.config:
        raise ValueError("paper and bridge configs must match")
    cfg = paper.config
    with torch.no_grad():
        bridge.token_embedding.weight.copy_(paper.token_embedding.weight)
        bridge.final_norm.load_state_dict(paper.final_norm.state_dict())
        bridge.lm_head.weight.copy_(paper.lm_head.weight)
        for source, target in zip(paper.blocks, bridge.blocks, strict=True):
            target.norm_1.load_state_dict(source.norm_1.state_dict())
            target.norm_2.load_state_dict(source.norm_2.state_dict())
            fused_w = source.attn.attn.weight.view(
                cfg.n_head, 3, cfg.head_size, cfg.n_embd
            )
            fused_b = source.attn.attn.bias.view(cfg.n_head, 3, cfg.head_size)
            for index, projection in enumerate(
                (target.attention.W_Q, target.attention.W_K, target.attention.W_V)
            ):
                projection.weight.copy_(
                    fused_w[:, index].reshape(cfg.n_embd, cfg.n_embd)
                )
                projection.bias.copy_(fused_b[:, index].reshape(cfg.n_embd))
            target.attention.W_O.load_state_dict(source.attn.proj.state_dict())
            target.mlp_in.load_state_dict(source.mlp.fc.state_dict())
            target.mlp_out.load_state_dict(source.mlp.proj.state_dict())


def paper_block_core_rows(block: PaperBlock) -> torch.Tensor:
    """Naively map a LitGPT block into the addition iMF's six-matrix coordinates."""
    cfg = block.attn.cfg
    fused = (
        block.attn.attn.weight.detach()
        .float()
        .view(cfg.n_head, 3, cfg.head_size, cfg.n_embd)
    )
    q, k, v = (fused[:, index].reshape(cfg.n_embd, cfg.n_embd) for index in range(3))
    return torch.cat(
        (
            q,
            k,
            v,
            block.attn.proj.weight.detach().float(),
            block.mlp.fc.weight.detach().float(),
            block.mlp.proj.weight.detach().float().T,
        ),
        dim=0,
    )


def copy_rows_to_paper_block(block: PaperBlock, rows: torch.Tensor) -> None:
    """Exploratory adapter: preserve norms/biases and overwrite only six core matrices."""
    tensors = unpack_rows(rows)
    cfg = block.attn.cfg
    q = tensors["attention.W_Q.weight"].view(cfg.n_head, cfg.head_size, cfg.n_embd)
    k = tensors["attention.W_K.weight"].view(cfg.n_head, cfg.head_size, cfg.n_embd)
    v = tensors["attention.W_V.weight"].view(cfg.n_head, cfg.head_size, cfg.n_embd)
    fused = torch.stack((q, k, v), dim=1).reshape(3 * cfg.n_embd, cfg.n_embd)
    with torch.no_grad():
        block.attn.attn.weight.copy_(fused.to(block.attn.attn.weight))
        block.attn.proj.weight.copy_(
            tensors["attention.W_O.weight"].to(block.attn.proj.weight)
        )
        block.mlp.fc.weight.copy_(tensors["mlp_in.weight"].to(block.mlp.fc.weight))
        block.mlp.proj.weight.copy_(tensors["mlp_out.weight"].to(block.mlp.proj.weight))


def block_parameters(model: nn.Module, indices: Iterable[int]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    for index in indices:
        result.extend(model.blocks[index].parameters())
    return result
