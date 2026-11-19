from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import TaskConfig
from .data import make_modular_addition
from .model import TaskTransformer
from .train import evaluate, seed_everything


@dataclass
class EmbeddingPairCollectionConfig:
    snapshot_every: int = 25
    pre_threshold_steps: int = 2_500
    post_threshold_steps: int = 2_000
    crossing_pairs: int = 4
    before_pairs: int = 3
    after_pairs: int = 3
    max_runs: int = 20_000
    max_dataset_gb: float = 30.0
    first_seed: int = 100
    phase_temperature_steps: float = 900.0
    sampling_strategy: str = "phase_focused"
    include_checkpoint_metrics: bool = False

    @property
    def pairs_per_run(self) -> int:
        return self.crossing_pairs + self.before_pairs + self.after_pairs


def _embedding_snapshot(model: TaskTransformer) -> torch.Tensor:
    return model.token_embedding.weight.detach().float().cpu().clone()


def _weighted_choice(
    candidates: list[int],
    steps: list[int],
    threshold_step: int,
    temperature: float,
    rng: random.Random,
) -> int:
    weights = [
        math.exp(-abs(steps[index] - threshold_step) / temperature) + 0.02
        for index in candidates
    ]
    return rng.choices(candidates, weights=weights, k=1)[0]


def sample_transition_pairs(
    steps: list[int], threshold_step: int, config: EmbeddingPairCollectionConfig, seed: int
) -> list[tuple[int, int, str]]:
    rng = random.Random(seed)
    before = [index for index, step in enumerate(steps) if step < threshold_step]
    after = [index for index, step in enumerate(steps) if step >= threshold_step]
    if len(before) < 2 or len(after) < 2:
        raise ValueError("Not enough checkpoints on both sides of the generalization threshold")
    requested = {
        "crossing": config.crossing_pairs,
        "before": config.before_pairs,
        "after": config.after_pairs,
    }
    pairs: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for _ in range(10_000):
        deficits = [
            kind
            for kind, count in requested.items()
            if sum(label == kind for _, _, label in pairs) < count
        ]
        if not deficits:
            break
        kind = rng.choice(deficits)
        if kind == "crossing":
            start = _weighted_choice(
                before, steps, threshold_step, config.phase_temperature_steps, rng
            )
            end = _weighted_choice(
                after, steps, threshold_step, config.phase_temperature_steps, rng
            )
        else:
            pool = before if kind == "before" else after
            start = _weighted_choice(
                pool[:-1], steps, threshold_step, config.phase_temperature_steps, rng
            )
            later = [candidate for candidate in pool if candidate > start]
            if not later:
                continue
            end = _weighted_choice(
                later, steps, threshold_step, config.phase_temperature_steps, rng
            )
        if start >= end or (start, end) in seen:
            continue
        seen.add((start, end))
        pairs.append((start, end, kind))
    if len(pairs) != config.pairs_per_run:
        raise RuntimeError("Could not sample ten unique transition pairs")
    rng.shuffle(pairs)
    return pairs


def sample_uniform_pairs(
    steps: list[int], config: EmbeddingPairCollectionConfig, seed: int
) -> list[tuple[int, int, str]]:
    if len(steps) < 2:
        raise ValueError("Need at least two checkpoints")
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < config.pairs_per_run:
        first, second = rng.sample(range(len(steps)), 2)
        start, end = sorted((first, second))
        seen.add((start, end))
    pairs = [(start, end, "uniform") for start, end in seen]
    rng.shuffle(pairs)
    return pairs


def _write_embedding_pair_shard(
    output_file: Path,
    seed: int,
    threshold_step: int,
    memorization_step: int | None,
    snapshots: list[tuple[int, torch.Tensor, float, float, float, float]],
    task_config: TaskConfig,
    pair_config: EmbeddingPairCollectionConfig,
    wall_seconds: float,
) -> dict:
    steps = [snapshot[0] for snapshot in snapshots]
    if pair_config.sampling_strategy == "phase_focused":
        sampled = sample_transition_pairs(
            steps, threshold_step, pair_config, seed + 9_000_000
        )
    elif pair_config.sampling_strategy == "uniform_within_run":
        sampled = sample_uniform_pairs(steps, pair_config, seed + 9_000_000)
    else:
        raise ValueError(f"Unknown sampling strategy: {pair_config.sampling_strategy}")
    used = sorted({index for start, end, _ in sampled for index in (start, end)})
    compact = {old: new for new, old in enumerate(used)}
    embeddings = torch.stack([snapshots[index][1] for index in used]).contiguous()
    kept_steps = torch.tensor([steps[index] for index in used], dtype=torch.int32)
    pair_indices = torch.tensor(
        [[compact[start], compact[end]] for start, end, _ in sampled], dtype=torch.int16
    )
    pair_types = [kind for _, _, kind in sampled]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 5 if pair_config.include_checkpoint_metrics else 4,
        "scope": "token_embedding.weight",
        "seed": seed,
        "threshold_step": threshold_step,
        "memorization_step": memorization_step,
        "steps": kept_steps,
        "embeddings": embeddings,
        "pair_indices": pair_indices,
        "pair_types": pair_types,
        "embedding_shape": list(embeddings.shape[1:]),
        "task_config": task_config.to_dict(),
        "pair_collection_config": asdict(pair_config),
    }
    if pair_config.include_checkpoint_metrics:
        payload.update(
            {
                "train_losses": torch.tensor([snapshots[index][2] for index in used]),
                "train_accuracies": torch.tensor(
                    [snapshots[index][3] for index in used]
                ),
                "validation_losses": torch.tensor(
                    [snapshots[index][4] for index in used]
                ),
                "validation_accuracies": torch.tensor(
                    [snapshots[index][5] for index in used]
                ),
            }
        )
    torch.save(payload, output_file)
    return {
        "seed": seed,
        "threshold_step": threshold_step,
        "memorization_step": memorization_step,
        "pairs": len(sampled),
        "unique_checkpoints": len(used),
        "first_kept_step": int(kept_steps.min()),
        "last_kept_step": int(kept_steps.max()),
        "bytes": output_file.stat().st_size,
        "wall_seconds": wall_seconds,
        "file": output_file.name,
        "pair_type_counts": {
            kind: pair_types.count(kind) for kind in sorted(set(pair_types))
        },
    }


def collect_one_embedding_pair_shard(
    task_config: TaskConfig,
    pair_config: EmbeddingPairCollectionConfig,
    seed: int,
    output_file: Path,
    device: str = "cuda",
) -> dict | None:
    seed_everything(seed)
    data = make_modular_addition(task_config, split_seed=seed)
    train_x, train_y = data.train_x.to(device), data.train_y.to(device)
    val_x, val_y = data.val_x.to(device), data.val_y.to(device)
    model = TaskTransformer(task_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=task_config.learning_rate,
        weight_decay=task_config.weight_decay,
    )
    detection_lag = task_config.log_every * (task_config.threshold_patience_logs - 1)
    rolling_count = (
        math.ceil(
            (pair_config.pre_threshold_steps + detection_lag) / pair_config.snapshot_every
        )
        + 4
    )
    rolling: deque[tuple[int, torch.Tensor, float, float, float, float]] = deque(
        maxlen=rolling_count
    )
    uniform_snapshots: list[tuple[int, torch.Tensor, float, float, float, float]] = []
    retained: list[tuple[int, torch.Tensor, float, float, float, float]] | None = None
    threshold_candidates: list[int] = []
    threshold_step = None
    memorization_step = None
    started = time.perf_counter()
    if pair_config.include_checkpoint_metrics:
        model.eval()
        initial_train_loss, initial_train_acc = evaluate(model, train_x, train_y)
        initial_val_loss, initial_val_acc = evaluate(model, val_x, val_y)
    else:
        initial_train_loss = initial_train_acc = initial_val_loss = initial_val_acc = (
            float("nan")
        )
    initial_snapshot = (
        0,
        _embedding_snapshot(model),
        initial_train_loss,
        initial_train_acc,
        initial_val_loss,
        initial_val_acc,
    )
    if pair_config.sampling_strategy == "uniform_within_run":
        uniform_snapshots.append(initial_snapshot)
    else:
        rolling.append(initial_snapshot)

    for step in range(1, task_config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        snapshot_due = step % pair_config.snapshot_every == 0
        log_due = step == 1 or step % task_config.log_every == 0
        metrics_due = log_due or (
            pair_config.include_checkpoint_metrics and snapshot_due
        )
        train_loss = train_acc = val_loss = val_acc = float("nan")
        if metrics_due:
            model.eval()
            train_loss, train_acc = evaluate(model, train_x, train_y)
            val_loss, val_acc = evaluate(model, val_x, val_y)
        if snapshot_due:
            snapshot = (
                step,
                _embedding_snapshot(model),
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            if pair_config.sampling_strategy == "uniform_within_run":
                uniform_snapshots.append(snapshot)
            elif retained is None:
                rolling.append(snapshot)
            else:
                retained.append(snapshot)
        if log_due:
            if memorization_step is None and train_acc >= 0.99:
                memorization_step = step
            if train_acc >= 0.99 and val_acc >= task_config.threshold_accuracy:
                threshold_candidates.append(step)
                if (
                    len(threshold_candidates) >= task_config.threshold_patience_logs
                    and threshold_step is None
                ):
                    threshold_step = threshold_candidates[-task_config.threshold_patience_logs]
                    if pair_config.sampling_strategy == "phase_focused":
                        retained = list(rolling)
            else:
                threshold_candidates.clear()
        if threshold_step is not None and step >= threshold_step + pair_config.post_threshold_steps:
            break
    if threshold_step is None:
        return None
    if pair_config.sampling_strategy == "uniform_within_run":
        retained = uniform_snapshots
    if retained is None:
        return None
    if pair_config.sampling_strategy == "phase_focused":
        low, high = (
            threshold_step - pair_config.pre_threshold_steps,
            threshold_step + pair_config.post_threshold_steps,
        )
        retained = [snapshot for snapshot in retained if low <= snapshot[0] <= high]
    if retained[-1][0] < step:
        model.eval()
        final_train_loss, final_train_acc = evaluate(model, train_x, train_y)
        final_val_loss, final_val_acc = evaluate(model, val_x, val_y)
        retained.append(
            (
                step,
                _embedding_snapshot(model),
                final_train_loss,
                final_train_acc,
                final_val_loss,
                final_val_acc,
            )
        )
    return _write_embedding_pair_shard(
        output_file,
        seed,
        threshold_step,
        memorization_step,
        retained,
        task_config,
        pair_config,
        time.perf_counter() - started,
    )


def collect_embedding_pair_dataset(
    task_config: TaskConfig,
    pair_config: EmbeddingPairCollectionConfig,
    dataset_dir: Path,
    device: str = "cuda",
) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "format_version": 5 if pair_config.include_checkpoint_metrics else 4,
            "scope": "token_embedding.weight",
            "task_config": task_config.to_dict(),
            "pair_collection_config": asdict(pair_config),
            "runs": [],
        }
    completed = {row["seed"] for row in manifest["runs"]}
    seed = pair_config.first_seed
    failures = manifest.get("failed_runs", 0)
    byte_limit = int(pair_config.max_dataset_gb * 1024**3)
    total_bytes = sum(row["bytes"] for row in manifest["runs"])
    while len(manifest["runs"]) < pair_config.max_runs and total_bytes < byte_limit:
        if seed in completed:
            seed += 1
            continue
        output_file = dataset_dir / f"embedding_pairs_seed_{seed:04d}.pt"
        row = collect_one_embedding_pair_shard(
            task_config, pair_config, seed, output_file, device
        )
        if row is None:
            failures += 1
            print(f"seed={seed} did not reach the generalization threshold; discarded", flush=True)
        else:
            manifest["runs"].append(row)
            total_bytes += row["bytes"]
            manifest["total_bytes"] = total_bytes
            manifest["total_gib"] = total_bytes / 1024**3
            manifest["failed_runs"] = failures
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"kept={len(manifest['runs'])} seed={seed} threshold={row['threshold_step']} "
                f"pairs={row['pairs']} unique={row['unique_checkpoints']} "
                f"dataset_gib={manifest['total_gib']:.3f}",
                flush=True,
            )
        seed += 1
    return manifest
