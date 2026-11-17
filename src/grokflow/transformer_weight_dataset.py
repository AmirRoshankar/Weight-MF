from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import GrokConfig
from .data import make_modular_addition
from .embedding_pair_dataset import sample_transition_pairs
from .model import GrokTransformer
from .train import evaluate, seed_everything


TRANSFORMER_WEIGHT_NAMES = (
    "attention.W_Q.weight",
    "attention.W_K.weight",
    "attention.W_V.weight",
    "attention.W_O.weight",
    "mlp_in.weight",
    "mlp_out.weight",
)


@dataclass
class TransformerWeightCollectionConfig:
    snapshot_every: int = 25
    pre_grok_steps: int = 2_500
    post_grok_steps: int = 2_000
    crossing_pairs: int = 4
    before_pairs: int = 3
    after_pairs: int = 3
    max_runs: int = 602
    max_dataset_gb: float = 10.0
    first_seed: int = 20_000
    phase_temperature_steps: float = 900.0

    @property
    def pairs_per_run(self) -> int:
        return self.crossing_pairs + self.before_pairs + self.after_pairs


Snapshot = tuple[int, dict[str, torch.Tensor], float, float, float, float]


def transformer_weight_snapshot(model: GrokTransformer) -> dict[str, torch.Tensor]:
    """Copy the six transformer-block matrices without flattening them."""
    state = model.state_dict()
    return {
        name: state[name].detach().to(device="cpu", dtype=torch.float16).clone()
        for name in TRANSFORMER_WEIGHT_NAMES
    }


def canonical_weight_rows(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack matrices as 1,536 width-128 rows while preserving tensor boundaries.

    ``mlp_out`` is transposed so both MLP tensors use the model dimension as
    their 128-wide feature axis.  The inverse representation is deterministic.
    """
    rows = []
    for name in TRANSFORMER_WEIGHT_NAMES:
        value = weights[name]
        rows.append(value.T if name == "mlp_out.weight" else value)
    return torch.cat(rows, dim=0)


def canonical_row_layout() -> list[dict]:
    result = []
    offset = 0
    for tensor_id, name in enumerate(TRANSFORMER_WEIGHT_NAMES):
        rows = 512 if name in {"mlp_in.weight", "mlp_out.weight"} else 128
        result.append(
            {
                "tensor_id": tensor_id,
                "name": name,
                "row_start": offset,
                "row_end": offset + rows,
                "canonical_shape": [rows, 128],
                "stored_shape": [128, 512] if name == "mlp_out.weight" else [rows, 128],
                "transpose_for_canonical": name == "mlp_out.weight",
            }
        )
        offset += rows
    return result


def _write_shard(
    output_file: Path,
    seed: int,
    grok_step: int,
    memorization_step: int | None,
    snapshots: list[Snapshot],
    grok_config: GrokConfig,
    pair_config: TransformerWeightCollectionConfig,
    wall_seconds: float,
) -> dict:
    steps = [snapshot[0] for snapshot in snapshots]
    sampled = sample_transition_pairs(steps, grok_step, pair_config, seed + 19_000_000)
    used = sorted({index for start, end, _ in sampled for index in (start, end)})
    compact = {old: new for new, old in enumerate(used)}
    weights = {
        name: torch.stack([snapshots[index][1][name] for index in used]).contiguous()
        for name in TRANSFORMER_WEIGHT_NAMES
    }
    kept_steps = torch.tensor([steps[index] for index in used], dtype=torch.int32)
    pair_indices = torch.tensor(
        [[compact[start], compact[end]] for start, end, _ in sampled], dtype=torch.int16
    )
    pair_types = [kind for _, _, kind in sampled]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "scope": "transformer_block_weights",
        "seed": seed,
        "grok_step": grok_step,
        "memorization_step": memorization_step,
        "steps": kept_steps,
        "weights": weights,
        "weight_names": list(TRANSFORMER_WEIGHT_NAMES),
        "canonical_row_layout": canonical_row_layout(),
        "pair_indices": pair_indices,
        "pair_types": pair_types,
        "train_losses": torch.tensor([snapshots[index][2] for index in used]),
        "train_accuracies": torch.tensor([snapshots[index][3] for index in used]),
        "validation_losses": torch.tensor([snapshots[index][4] for index in used]),
        "validation_accuracies": torch.tensor([snapshots[index][5] for index in used]),
        "grok_config": grok_config.to_dict(),
        "pair_collection_config": asdict(pair_config),
    }
    torch.save(payload, output_file)
    return {
        "seed": seed,
        "grok_step": grok_step,
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


def collect_one_transformer_weight_shard(
    grok_config: GrokConfig,
    pair_config: TransformerWeightCollectionConfig,
    seed: int,
    output_file: Path,
    device: str = "cuda",
) -> dict | None:
    seed_everything(seed)
    data = make_modular_addition(grok_config, split_seed=seed)
    train_x, train_y = data.train_x.to(device), data.train_y.to(device)
    val_x, val_y = data.val_x.to(device), data.val_y.to(device)
    model = GrokTransformer(grok_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=grok_config.learning_rate,
        weight_decay=grok_config.weight_decay,
    )
    detection_lag = grok_config.log_every * (grok_config.grok_patience_logs - 1)
    rolling_count = (
        math.ceil(
            (pair_config.pre_grok_steps + detection_lag) / pair_config.snapshot_every
        )
        + 4
    )
    rolling: deque[Snapshot] = deque(maxlen=rolling_count)
    retained: list[Snapshot] | None = None
    grok_candidates: list[int] = []
    grok_step = memorization_step = None
    started = time.perf_counter()

    model.eval()
    initial_train_loss, initial_train_acc = evaluate(model, train_x, train_y)
    initial_val_loss, initial_val_acc = evaluate(model, val_x, val_y)
    rolling.append(
        (
            0,
            transformer_weight_snapshot(model),
            initial_train_loss,
            initial_train_acc,
            initial_val_loss,
            initial_val_acc,
        )
    )

    for step in range(1, grok_config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        snapshot_due = step % pair_config.snapshot_every == 0
        log_due = step == 1 or step % grok_config.log_every == 0
        metrics_due = snapshot_due or log_due
        train_loss = train_acc = val_loss = val_acc = float("nan")
        if metrics_due:
            model.eval()
            train_loss, train_acc = evaluate(model, train_x, train_y)
            val_loss, val_acc = evaluate(model, val_x, val_y)
        if snapshot_due:
            snapshot = (
                step,
                transformer_weight_snapshot(model),
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            if retained is None:
                rolling.append(snapshot)
            else:
                retained.append(snapshot)
        if log_due:
            if memorization_step is None and train_acc >= 0.99:
                memorization_step = step
            if train_acc >= 0.99 and val_acc >= grok_config.grok_accuracy:
                grok_candidates.append(step)
                if (
                    len(grok_candidates) >= grok_config.grok_patience_logs
                    and grok_step is None
                ):
                    grok_step = grok_candidates[-grok_config.grok_patience_logs]
                    retained = list(rolling)
            else:
                grok_candidates.clear()
        if grok_step is not None and step >= grok_step + pair_config.post_grok_steps:
            break

    if grok_step is None or retained is None:
        return None
    low, high = (
        grok_step - pair_config.pre_grok_steps,
        grok_step + pair_config.post_grok_steps,
    )
    retained = [snapshot for snapshot in retained if low <= snapshot[0] <= high]
    if retained[-1][0] < step:
        model.eval()
        train_loss, train_acc = evaluate(model, train_x, train_y)
        val_loss, val_acc = evaluate(model, val_x, val_y)
        retained.append(
            (
                step,
                transformer_weight_snapshot(model),
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
        )
    return _write_shard(
        output_file,
        seed,
        grok_step,
        memorization_step,
        retained,
        grok_config,
        pair_config,
        time.perf_counter() - started,
    )
