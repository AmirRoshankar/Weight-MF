from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F

from .embedding_meanflow import EmbeddingMeanFlowConfig, TokenEmbeddingMeanFlow


@dataclass(frozen=True)
class VelocityInvestigationConfig:
    train_steps: int = 10_000
    batch_size: int = 32
    runs_per_batch: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    cosine_weight: float = 0.05
    log_every: int = 100
    plot_every: int = 500
    train_eval_pairs: int = 256
    validation_pairs: int = 512
    test_pairs: int = 2_048
    max_normalizer_runs: int = 882
    max_step: float = 30_000.0
    seed: int = 20_261_105
    width: int = 384
    latent_tokens: int = 96
    latent_depth: int = 6
    heads: int = 8
    minimum_train_steps: int = 0
    stability_patience_steps: int = 0
    stability_relative_improvement: float = 0.002
    max_train_runs: int = 0
    validation_run_count: int = 0
    test_run_count: int = 0


@lru_cache(maxsize=96)
def _load_shard(path: str) -> dict:
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def possible_pair_count(rows: Iterable[dict]) -> int:
    return sum(
        int(row["unique_checkpoints"]) * (int(row["unique_checkpoints"]) - 1) // 2
        for row in rows
    )


def make_run_split(
    run_rows: list[dict],
    seed: int,
    validation_run_count: int = 0,
    test_run_count: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows = sorted(run_rows, key=lambda row: int(row["seed"]))
    random.Random(seed).shuffle(rows)
    if validation_run_count or test_run_count:
        validation_count = validation_run_count
        test_count = test_run_count
        if validation_count < 1 or test_count < 1:
            raise ValueError(
                "Explicit validation and test run counts must both be positive"
            )
        if validation_count + test_count >= len(rows):
            raise ValueError("Validation and test splits leave no training runs")
    else:
        test_count = max(1, round(len(rows) * 0.10))
        validation_count = max(1, round(len(rows) * 0.10))
    test = rows[:test_count]
    validation = rows[test_count : test_count + validation_count]
    train = rows[test_count + validation_count :]
    return train, validation, test


def _all_indices(endpoint_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(endpoint_count), 2))


def make_fixed_pairs(rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pairs: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    while len(pairs) < count:
        row = rng.choice(rows)
        candidates = _all_indices(int(row["unique_checkpoints"]))
        start_index, end_index = rng.choice(candidates)
        key = (row["file"], start_index, end_index)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "seed": int(row["seed"]),
                "file": row["file"],
                "start_index": start_index,
                "end_index": end_index,
            }
        )
    return pairs


def _normalizer(
    dataset_dir: Path, train_rows: list[dict], config: VelocityInvestigationConfig
) -> tuple[float, float]:
    rng = random.Random(config.seed)
    selected = rng.sample(train_rows, min(config.max_normalizer_runs, len(train_rows)))
    total, total_square, count = 0.0, 0.0, 0
    for row in selected:
        values = _load_shard(str(dataset_dir / row["file"]))["embeddings"].float()
        total += float(values.sum(dtype=torch.float64))
        total_square += float(values.square().sum(dtype=torch.float64))
        count += values.numel()
    mean = total / count
    variance = max(total_square / count - mean * mean, 1e-8)
    return mean, math.sqrt(variance)


def _materialize_pairs(
    dataset_dir: Path,
    pair_rows: list[dict],
    mean: float,
    scale: float,
    max_step: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    early, late, starts, ends = [], [], [], []
    for pair in pair_rows:
        shard = _load_shard(str(dataset_dir / pair["file"]))
        i, j = pair["start_index"], pair["end_index"]
        early.append((shard["embeddings"][i].float() - mean) / scale)
        late.append((shard["embeddings"][j].float() - mean) / scale)
        starts.append(float(shard["steps"][i]) / max_step)
        ends.append(float(shard["steps"][j]) / max_step)
    return (
        torch.stack(early),
        torch.stack(late),
        torch.tensor(starts, dtype=torch.float32).unsqueeze(1),
        torch.tensor(ends, dtype=torch.float32).unsqueeze(1),
    )


def _sample_training_batch(
    dataset_dir: Path,
    rows: list[dict],
    config: VelocityInvestigationConfig,
    mean: float,
    scale: float,
    rng: random.Random,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_runs = rng.sample(rows, config.runs_per_batch)
    per_run = config.batch_size // config.runs_per_batch
    early, late, starts, ends = [], [], [], []
    for row in chosen_runs:
        shard = _load_shard(str(dataset_dir / row["file"]))
        candidates = _all_indices(shard["embeddings"].shape[0])
        for i, j in rng.sample(candidates, per_run):
            early.append((shard["embeddings"][i].float() - mean) / scale)
            late.append((shard["embeddings"][j].float() - mean) / scale)
            starts.append(float(shard["steps"][i]) / config.max_step)
            ends.append(float(shard["steps"][j]) / config.max_step)
    return (
        torch.stack(early).to(device, non_blocking=True),
        torch.stack(late).to(device, non_blocking=True),
        torch.tensor(starts, dtype=torch.float32, device=device).unsqueeze(1),
        torch.tensor(ends, dtype=torch.float32, device=device).unsqueeze(1),
    )


def _direct_velocity_loss(
    model: TokenEmbeddingMeanFlow,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cosine_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    early, late, start, end = batch
    zeros, ones = torch.zeros_like(start), torch.ones_like(start)
    prediction = model(early, zeros, ones, start, end)
    target = early - late
    mse = F.mse_loss(prediction, target)
    cosine_loss = (
        1 - F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=-1).mean()
    )
    return mse + cosine_weight * cosine_loss, {
        "velocity_mse": float(mse.detach()),
        "velocity_cosine_loss": float(cosine_loss.detach()),
    }


@torch.no_grad()
def _evaluate(
    model: TokenEmbeddingMeanFlow,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    chunk_size: int = 32,
) -> dict[str, float]:
    squared_error, delta_cosine, velocity_rms, elements, examples = 0.0, 0.0, 0.0, 0, 0
    for offset in range(0, tensors[0].shape[0], chunk_size):
        early, late, start, end = [
            value[offset : offset + chunk_size].to(device) for value in tensors
        ]
        zeros, ones = torch.zeros_like(start), torch.ones_like(start)
        velocity = model(early, zeros, ones, start, end)
        prediction = early - velocity
        difference = prediction - late
        squared_error += float(difference.square().sum())
        elements += difference.numel()
        delta_cosine += float(
            F.cosine_similarity(
                (-velocity).flatten(1), (late - early).flatten(1), dim=-1
            ).sum()
        )
        velocity_rms += float(velocity.square().mean(dim=(1, 2)).sqrt().sum())
        examples += early.shape[0]
    return {
        "endpoint_mse": squared_error / elements,
        "delta_cosine": delta_cosine / examples,
        "velocity_rms": velocity_rms / examples,
    }


def _do_nothing_mse(tensors: tuple[torch.Tensor, ...]) -> float:
    return float(F.mse_loss(tensors[0], tensors[1]))


def _plot(rows: list[dict], output_file: Path) -> None:
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    axes[0].plot(frame["step"], frame["fixed_train_endpoint_mse"], label="Fixed train")
    axes[0].plot(frame["step"], frame["validation_endpoint_mse"], label="Validation")
    axes[0].axhline(
        frame["validation_do_nothing_mse"].iloc[0],
        color="black",
        linestyle="--",
        label="Validation do-nothing",
    )
    axes[0].set(
        xlabel="Optimizer step",
        ylabel="Endpoint MSE",
        yscale="log",
        title="Endpoint prediction error",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(frame["step"], frame["fixed_train_delta_cosine"], label="Fixed train")
    axes[1].plot(frame["step"], frame["validation_delta_cosine"], label="Validation")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        xlabel="Optimizer step",
        ylabel="Cosine similarity",
        title="Update-direction agreement",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def run_velocity_investigation(
    dataset_dir: Path,
    output_dir: Path,
    config: VelocityInvestigationConfig = VelocityInvestigationConfig(),
    device: str = "cuda",
    trained_model_callback: Callable[[TokenEmbeddingMeanFlow, float, float], None]
    | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    run_rows = manifest["runs"]
    all_train_rows, validation_rows, test_rows = make_run_split(
        run_rows,
        config.seed,
        validation_run_count=config.validation_run_count,
        test_run_count=config.test_run_count,
    )
    train_rows = (
        all_train_rows[: config.max_train_runs]
        if config.max_train_runs > 0
        else all_train_rows
    )
    if len(train_rows) < config.runs_per_batch:
        raise ValueError(
            f"Need at least {config.runs_per_batch} training runs, got {len(train_rows)}"
        )
    train_pairs = make_fixed_pairs(train_rows, config.train_eval_pairs, config.seed + 1)
    validation_pairs = make_fixed_pairs(
        validation_rows, config.validation_pairs, config.seed + 2
    )
    test_pairs = make_fixed_pairs(test_rows, config.test_pairs, config.seed + 3)
    split_manifest = {
        "seed": config.seed,
        "policy": "whole source runs are disjoint; pairs are forward in time and never cross runs",
        "train": {
            "runs": [
                {
                    "seed": row["seed"],
                    "file": row["file"],
                    "endpoints": row["unique_checkpoints"],
                }
                for row in train_rows
            ],
            "possible_forward_pairs": possible_pair_count(train_rows),
        },
        "validation": {
            "runs": [
                {
                    "seed": row["seed"],
                    "file": row["file"],
                    "endpoints": row["unique_checkpoints"],
                }
                for row in validation_rows
            ],
            "possible_forward_pairs": possible_pair_count(validation_rows),
            "fixed_pairs": validation_pairs,
        },
        "test": {
            "runs": [
                {
                    "seed": row["seed"],
                    "file": row["file"],
                    "endpoints": row["unique_checkpoints"],
                }
                for row in test_rows
            ],
            "possible_forward_pairs": possible_pair_count(test_rows),
            "fixed_pairs": test_pairs,
        },
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    mean, scale = _normalizer(dataset_dir, train_rows, config)
    fixed_train = _materialize_pairs(
        dataset_dir, train_pairs, mean, scale, config.max_step
    )
    fixed_validation = _materialize_pairs(
        dataset_dir, validation_pairs, mean, scale, config.max_step
    )
    fixed_test = _materialize_pairs(
        dataset_dir, test_pairs, mean, scale, config.max_step
    )
    validation_baseline = _do_nothing_mse(fixed_validation)
    test_baseline = _do_nothing_mse(fixed_test)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    torch.set_float32_matmul_precision("high")
    model_config = EmbeddingMeanFlowConfig(
        width=config.width,
        latent_tokens=config.latent_tokens,
        latent_depth=config.latent_depth,
        heads=config.heads,
    )
    model = TokenEmbeddingMeanFlow(model_config).to(device)
    torch.nn.init.normal_(model.output_projection.weight, std=1e-3)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = random.Random(config.seed + 4)
    rows: list[dict] = []
    best_validation = float("inf")
    best_state = None
    significant_train = float("inf")
    significant_validation = float("inf")
    last_significant_train_step = 0
    last_significant_validation_step = 0
    stopped_for_stability = False
    started = time.perf_counter()

    for step in range(1, config.train_steps + 1):
        model.train()
        batch = _sample_training_batch(
            dataset_dir, train_rows, config, mean, scale, rng, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
        ):
            loss, sampled = _direct_velocity_loss(model, batch, config.cosine_weight)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % config.log_every == 0 or step == config.train_steps:
            model.eval()
            fixed_train_metrics = _evaluate(model, fixed_train, device)
            validation_metrics = _evaluate(model, fixed_validation, device)
            row = {
                "step": step,
                "sampled_train_loss": float(loss.detach()),
                "sampled_velocity_mse": sampled["velocity_mse"],
                "sampled_velocity_cosine_loss": sampled["velocity_cosine_loss"],
                "fixed_train_endpoint_mse": fixed_train_metrics["endpoint_mse"],
                "fixed_train_delta_cosine": fixed_train_metrics["delta_cosine"],
                "validation_endpoint_mse": validation_metrics["endpoint_mse"],
                "validation_delta_cosine": validation_metrics["delta_cosine"],
                "validation_velocity_rms": validation_metrics["velocity_rms"],
                "validation_do_nothing_mse": validation_baseline,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            if validation_metrics["endpoint_mse"] < best_validation:
                best_validation = validation_metrics["endpoint_mse"]
                best_state = copy.deepcopy(
                    {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                )
            if fixed_train_metrics["endpoint_mse"] < significant_train * (
                1 - config.stability_relative_improvement
            ):
                significant_train = fixed_train_metrics["endpoint_mse"]
                last_significant_train_step = step
            if validation_metrics["endpoint_mse"] < significant_validation * (
                1 - config.stability_relative_improvement
            ):
                significant_validation = validation_metrics["endpoint_mse"]
                last_significant_validation_step = step
            pd.DataFrame(rows).to_csv(output_dir / "training_metrics.csv", index=False)
            if step == 1 or step % config.plot_every == 0 or step == config.train_steps:
                _plot(rows, output_dir / "training_curves.png")
            print(
                f"step={step} sampled={row['sampled_train_loss']:.6f} "
                f"train={row['fixed_train_endpoint_mse']:.6f} "
                f"val={row['validation_endpoint_mse']:.6f} "
                f"val_cos={row['validation_delta_cosine']:.4f} "
                f"baseline={validation_baseline:.6f}",
                flush=True,
            )
            last_significant_step = max(
                last_significant_train_step, last_significant_validation_step
            )
            if (
                config.stability_patience_steps > 0
                and step >= config.minimum_train_steps
                and step - last_significant_step >= config.stability_patience_steps
            ):
                stopped_for_stability = True
                print(
                    f"stability criterion reached at step={step}; "
                    f"last meaningful train/validation improvement={last_significant_step}",
                    flush=True,
                )
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device).eval()
    test_metrics = _evaluate(model, fixed_test, device)
    predictor_training_seconds = time.perf_counter() - started
    callback_seconds = 0.0
    if trained_model_callback is not None:
        callback_started = time.perf_counter()
        trained_model_callback(model, mean, scale)
        callback_seconds = time.perf_counter() - callback_started
    summary = {
        "config": asdict(config),
        "model_config": asdict(model_config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "normalizer_mean": mean,
        "normalizer_scale": scale,
        "run_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "possible_forward_pairs": {
            "train": possible_pair_count(train_rows),
            "validation": possible_pair_count(validation_rows),
            "test": possible_pair_count(test_rows),
        },
        "best_validation_endpoint_mse": best_validation,
        "validation_do_nothing_mse": validation_baseline,
        "test_endpoint_mse": test_metrics["endpoint_mse"],
        "test_do_nothing_mse": test_baseline,
        "test_delta_cosine": test_metrics["delta_cosine"],
        "training_seconds": predictor_training_seconds,
        "downstream_callback_seconds": callback_seconds,
        "total_experiment_seconds": predictor_training_seconds + callback_seconds,
        "actual_train_steps": rows[-1]["step"],
        "stopped_for_stability": stopped_for_stability,
        "last_significant_train_step": last_significant_train_step,
        "last_significant_validation_step": last_significant_validation_step,
        "weights_saved": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    del optimizer, model, best_state, fixed_train, fixed_validation, fixed_test
    _load_shard.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
