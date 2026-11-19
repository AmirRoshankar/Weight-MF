from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F

from .meanflow_objective import improved_meanflow_loss, sample_improved_meanflow
from .transformer_weight_dataset import TRANSFORMER_WEIGHT_NAMES
from .transformer_weight_meanflow import (
    ConditionalTransformerWeightGenerator,
    TransformerWeightGeneratorConfig,
    canonical_tensor_and_row_ids,
)


@dataclass(frozen=True)
class TransformerWeightTrainingConfig:
    train_runs: int = 500
    validation_runs: int = 51
    test_runs: int = 51
    train_steps: int = 10_000
    batch_size: int = 4
    runs_per_batch: int = 4
    learning_rate: float = 1e-4
    adam_beta2: float = 0.95
    gradient_clip: float = 1.0
    log_every: int = 100
    endpoint_eval_every: int = 500
    endpoint_eval_pairs: int = 24
    seed: int = 20_261_108
    max_checkpoint_step: float = 30_000.0
    ema_decay: float = 0.9999
    metric_prompt: str = "training_loss"
    width: int = 240
    latent_tokens: int = 64
    latent_depth: int = 4
    heads: int = 8


@lru_cache(maxsize=96)
def _load_shard(path: str) -> dict:
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def _split(
    rows: list[dict], config: TransformerWeightTrainingConfig
) -> tuple[list[dict], ...]:
    required = config.train_runs + config.validation_runs + config.test_runs
    if len(rows) < required:
        raise ValueError(f"Need {required} source runs, found {len(rows)}")
    shuffled = sorted(rows, key=lambda row: int(row["seed"]))
    random.Random(config.seed).shuffle(shuffled)
    test = shuffled[: config.test_runs]
    validation = shuffled[config.test_runs : config.test_runs + config.validation_runs]
    train = shuffled[config.test_runs + config.validation_runs : required]
    return train, validation, test


def _normalizers(
    dataset_dir: Path, rows: list[dict]
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(len(TRANSFORMER_WEIGHT_NAMES), dtype=torch.float64)
    squares = torch.zeros_like(sums)
    counts = torch.zeros_like(sums)
    for row in rows:
        shard = _load_shard(str(dataset_dir / row["file"]))
        for index, name in enumerate(TRANSFORMER_WEIGHT_NAMES):
            values = shard["weights"][name].to(torch.float64)
            sums[index] += values.sum()
            squares[index] += values.square().sum()
            counts[index] += values.numel()
    means = sums / counts
    scales = (squares / counts - means.square()).clamp_min(1e-8).sqrt()
    return means.float(), scales.float()


def _canonical_checkpoint(shard: dict, checkpoint_index: int) -> torch.Tensor:
    rows = []
    for name in TRANSFORMER_WEIGHT_NAMES:
        value = shard["weights"][name][checkpoint_index].float()
        rows.append(value.T if name == "mlp_out.weight" else value)
    return torch.cat(rows, dim=0)


def _normalize(
    rows: torch.Tensor, means: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    tensor_ids, _ = canonical_tensor_and_row_ids()
    return (rows - means[tensor_ids, None]) / scales[tensor_ids, None]


def denormalize_transformer_rows(
    rows: torch.Tensor, means: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    tensor_ids, _ = canonical_tensor_and_row_ids()
    ids = tensor_ids.to(rows.device)
    return rows * scales.to(rows.device)[ids, None] + means.to(rows.device)[ids, None]


def unpack_canonical_rows(rows: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {}
    offset = 0
    for name in TRANSFORMER_WEIGHT_NAMES:
        count = 512 if name in {"mlp_in.weight", "mlp_out.weight"} else 128
        value = rows[offset : offset + count]
        result[name] = (
            value.T.contiguous() if name == "mlp_out.weight" else value.contiguous()
        )
        offset += count
    return result


def _metric(shard: dict, index: int, prompt: str) -> float:
    if prompt == "training_loss":
        return math.log1p(float(shard["train_losses"][index])) / math.log1p(5.0)
    if prompt == "training_error":
        return 1.0 - float(shard["train_accuracies"][index])
    if prompt == "validation_error":
        return 1.0 - float(shard["validation_accuracies"][index])
    if prompt == "none":
        return 0.0
    raise ValueError(prompt)


def _pair(
    row: dict,
    pair_index: int,
    means: torch.Tensor,
    scales: torch.Tensor,
    config: TransformerWeightTrainingConfig,
) -> tuple[torch.Tensor, ...]:
    shard = _load_shard(row["_absolute_file"])
    start_index, end_index = [int(x) for x in shard["pair_indices"][pair_index]]
    early = _normalize(_canonical_checkpoint(shard, start_index), means, scales)
    late = _normalize(_canonical_checkpoint(shard, end_index), means, scales)
    start_step = torch.tensor(
        [float(shard["steps"][start_index]) / config.max_checkpoint_step]
    )
    end_step = torch.tensor(
        [float(shard["steps"][end_index]) / config.max_checkpoint_step]
    )
    start_metric = torch.tensor([_metric(shard, start_index, config.metric_prompt)])
    end_metric = torch.tensor([_metric(shard, end_index, config.metric_prompt)])
    return early, late, start_step, end_step, start_metric, end_metric


def _batch(
    rows: list[dict],
    config: TransformerWeightTrainingConfig,
    means: torch.Tensor,
    scales: torch.Tensor,
    rng: random.Random,
    device: str,
) -> tuple[torch.Tensor, ...]:
    chosen = rng.sample(rows, config.runs_per_batch)
    per_run = config.batch_size // config.runs_per_batch
    examples = []
    for row in chosen:
        count = int(_load_shard(row["_absolute_file"])["pair_indices"].shape[0])
        for pair_index in rng.sample(range(count), per_run):
            examples.append(_pair(row, pair_index, means, scales, config))
    return tuple(
        torch.stack(column).to(device, non_blocking=True) for column in zip(*examples)
    )


def _records(rows: list[dict], limit: int | None = None) -> list[tuple[dict, int]]:
    records = []
    for row in rows:
        count = int(_load_shard(row["_absolute_file"])["pair_indices"].shape[0])
        records.extend((row, index) for index in range(count))
    return records[:limit] if limit is not None else records


@torch.no_grad()
def _endpoint_metrics(
    model: ConditionalTransformerWeightGenerator,
    records: list[tuple[dict, int]],
    means: torch.Tensor,
    scales: torch.Tensor,
    config: TransformerWeightTrainingConfig,
    device: str,
    seed: int,
) -> dict[str, float]:
    squared = cosine = 0.0
    elements = examples = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for offset in range(0, len(records), 2):
        materialized = [
            _pair(row, index, means, scales, config)
            for row, index in records[offset : offset + 2]
        ]
        batch = [torch.stack(column).to(device) for column in zip(*materialized)]
        early, late, start, end, start_metric, end_metric = batch
        prediction, _ = sample_improved_meanflow(
            model, early, start, end, start_metric, end_metric, generator
        )
        difference = prediction - late
        squared += float(difference.square().sum())
        elements += difference.numel()
        cosine += float(
            F.cosine_similarity(
                (prediction - early).flatten(1), (late - early).flatten(1), dim=1
            ).sum()
        )
        examples += early.shape[0]
    return {"endpoint_mse": squared / elements, "delta_cosine": cosine / examples}


def _ema_update(
    ema: ConditionalTransformerWeightGenerator,
    model: ConditionalTransformerWeightGenerator,
    decay: float,
) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)


def _plot(rows: list[dict], output: Path) -> None:
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(frame.step, frame.train_loss)
    axes[0].set(xlabel="Meta-training step", ylabel="iMF loss", yscale="log")
    evaluated = frame.dropna(subset=["validation_endpoint_mse"])
    axes[1].plot(
        evaluated.step, evaluated.validation_endpoint_mse, label="endpoint MSE"
    )
    axes[1].plot(
        evaluated.step, evaluated.validation_delta_cosine, label="delta cosine"
    )
    axes[1].set(xlabel="Meta-training step", ylabel="Held-out metric")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def train_transformer_weight_meanflow(
    dataset_dir: Path,
    output_dir: Path,
    config: TransformerWeightTrainingConfig = TransformerWeightTrainingConfig(),
    device: str = "cuda",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        dict(row, _absolute_file=str(dataset_dir / row["file"]))
        for row in manifest["runs"]
    ]
    train_rows, validation_rows, test_rows = _split(rows, config)
    means, scales = _normalizers(dataset_dir, train_rows)
    validation_records = _records(validation_rows, config.endpoint_eval_pairs)
    test_records = _records(test_rows)
    split = {
        "train_seeds": [row["seed"] for row in train_rows],
        "validation_seeds": [row["seed"] for row in validation_rows],
        "test_seeds": [row["seed"] for row in test_rows],
        "policy": "complete held-out source runs; only retained within-run pairs",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    model_config = TransformerWeightGeneratorConfig(
        width=config.width,
        latent_tokens=config.latent_tokens,
        latent_depth=config.latent_depth,
        heads=config.heads,
    )
    model = ConditionalTransformerWeightGenerator(model_config).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(0.9, config.adam_beta2)
    )
    rng = random.Random(config.seed + 1)
    metrics = []
    best_endpoint = float("inf")
    best_state = None
    started = time.perf_counter()
    for step in range(1, config.train_steps + 1):
        model.train()
        batch = _batch(train_rows, config, means, scales, rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = improved_meanflow_loss(model, *batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        _ema_update(ema, model, config.ema_decay)
        if step == 1 or step % config.log_every == 0 or step == config.train_steps:
            val_mse = val_cosine = float("nan")
            if (
                step == 1
                or step % config.endpoint_eval_every == 0
                or step == config.train_steps
            ):
                ema.eval()
                endpoint = _endpoint_metrics(
                    ema,
                    validation_records,
                    means,
                    scales,
                    config,
                    device,
                    config.seed + 2,
                )
                val_mse, val_cosine = endpoint["endpoint_mse"], endpoint["delta_cosine"]
                if val_mse < best_endpoint:
                    best_endpoint = val_mse
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in ema.state_dict().items()
                    }
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "validation_endpoint_mse": val_mse,
                "validation_delta_cosine": val_cosine,
                "elapsed_seconds": time.perf_counter() - started,
                **{name: float(value) for name, value in diagnostics.items()},
            }
            metrics.append(row)
            pd.DataFrame(metrics).to_csv(
                output_dir / "training_metrics.csv", index=False
            )
            _plot(metrics, output_dir / "training_curves.png")
            print(
                f"transformer-iMF step={step} loss={row['train_loss']:.6f} val={val_mse:.6f}",
                flush=True,
            )
    assert best_state is not None
    ema.load_state_dict(best_state)
    ema.to(device).eval()
    test = _endpoint_metrics(
        ema, test_records, means, scales, config, device, config.seed + 3
    )
    artifact = {
        "method": "improved_meanflow",
        "scope": "transformer_block_weights",
        "state_dict": best_state,
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "normalizer_means": means,
        "normalizer_scales": scales,
        "weight_names": list(TRANSFORMER_WEIGHT_NAMES),
    }
    torch.save(artifact, output_dir / "best_model.pt")
    summary = {
        "method": "improved_meanflow",
        "scope": "transformer_block_weights",
        "config": asdict(config),
        "model_config": asdict(model_config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "run_counts": {"train": 500, "validation": 51, "test": 51},
        "strict_training_pairs": 5_000,
        "best_validation_endpoint_mse": best_endpoint,
        "test_endpoint_mse": test["endpoint_mse"],
        "test_delta_cosine": test["delta_cosine"],
        "training_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
