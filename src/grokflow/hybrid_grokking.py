from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F

from .config import GrokConfig
from .data import make_modular_addition
from .embedding_meanflow import TokenEmbeddingMeanFlow
from .model import GrokTransformer
from .train import evaluate, seed_everything


@dataclass(frozen=True)
class HybridProtocol:
    name: str
    schedule: str = "none"
    target_mode: str = "fixed"
    fixed_target_step: int = 9_000
    horizon_steps: int = 2_000
    start_step: int = 500
    interval: int = 100
    norm_match: bool = False
    optimizer_weight: float = 0.5
    flow_weight: float = 0.5
    weight_schedule: str = "constant"
    ramp_up_end: int = 2_000
    hold_end: int = 4_000
    ramp_down_end: int = 6_000
    peak_flow_weight: float = 0.5


def mix_embedding_deltas(
    optimizer_delta: torch.Tensor,
    flow_delta: torch.Tensor,
    optimizer_weight: float = 0.5,
    flow_weight: float = 0.5,
    norm_match: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    optimizer_norm = optimizer_delta.norm()
    flow_norm = flow_delta.norm()
    cosine = F.cosine_similarity(optimizer_delta.flatten(), flow_delta.flatten(), dim=0)
    if norm_match:
        flow_delta = flow_delta * (optimizer_norm / flow_norm.clamp_min(1e-12))
    mixed = optimizer_weight * optimizer_delta + flow_weight * flow_delta
    return mixed, {
        "optimizer_delta_norm": float(optimizer_norm),
        "raw_flow_delta_norm": float(flow_norm),
        "mixed_delta_norm": float(mixed.norm()),
        "optimizer_flow_cosine": float(cosine),
    }


@torch.no_grad()
def predict_embedding_delta(
    velocity_model: TokenEmbeddingMeanFlow,
    embedding: torch.Tensor,
    current_step: int,
    target_step: int,
    normalizer_mean: float,
    normalizer_scale: float,
    max_condition_step: float = 30_000.0,
) -> torch.Tensor:
    normalized = (embedding - normalizer_mean) / normalizer_scale
    start = torch.tensor([[current_step / max_condition_step]], device=embedding.device)
    end = torch.tensor([[target_step / max_condition_step]], device=embedding.device)
    zeros, ones = torch.zeros_like(start), torch.ones_like(start)
    # The learned velocity is early - late, so the forward checkpoint delta is -velocity.
    velocity = velocity_model(normalized.unsqueeze(0), zeros, ones, start, end).squeeze(
        0
    )
    return -velocity * normalizer_scale


def _should_intervene(protocol: HybridProtocol, step: int) -> bool:
    if protocol.schedule == "none" or step < protocol.start_step:
        return False
    if protocol.weight_schedule == "trapezoid" and step > protocol.ramp_down_end:
        return False
    if protocol.schedule == "once":
        return step == protocol.start_step
    return (step - protocol.start_step) % protocol.interval == 0


def _mix_weights(protocol: HybridProtocol, step: int) -> tuple[float, float]:
    if protocol.weight_schedule != "trapezoid":
        return protocol.optimizer_weight, protocol.flow_weight
    if step <= protocol.ramp_up_end:
        width = max(protocol.ramp_up_end - protocol.start_step, 1)
        flow_weight = protocol.peak_flow_weight * (step - protocol.start_step) / width
    elif step <= protocol.hold_end:
        flow_weight = protocol.peak_flow_weight
    else:
        width = max(protocol.ramp_down_end - protocol.hold_end, 1)
        flow_weight = (
            protocol.peak_flow_weight * (protocol.ramp_down_end - step) / width
        )
    flow_weight = min(max(flow_weight, 0.0), protocol.peak_flow_weight)
    return 1.0 - flow_weight, flow_weight


def _target_step(protocol: HybridProtocol, current_step: int, max_steps: int) -> int:
    if protocol.target_mode == "rolling":
        return min(current_step + protocol.horizon_steps, max_steps)
    return max(current_step + 1, protocol.fixed_target_step)


def train_hybrid_run(
    velocity_model: TokenEmbeddingMeanFlow,
    normalizer_mean: float,
    normalizer_scale: float,
    protocol: HybridProtocol,
    seed: int,
    config: GrokConfig,
    device: str = "cuda",
) -> tuple[pd.DataFrame, dict]:
    seed_everything(seed)
    data = make_modular_addition(config, split_seed=seed)
    train_x, train_y = data.train_x.to(device), data.train_y.to(device)
    val_x, val_y = data.val_x.to(device), data.val_y.to(device)
    model = GrokTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rows, intervention_rows = [], []
    memorization_step, grok_step = None, None
    grok_candidates: list[int] = []
    flow_inference_seconds = 0.0
    started = time.perf_counter()

    for step in range(1, config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        embedding_before = model.token_embedding.weight.detach().clone()
        optimizer.step()
        if _should_intervene(protocol, step):
            embedding_after_optimizer = model.token_embedding.weight.detach().clone()
            optimizer_delta = embedding_after_optimizer - embedding_before
            target_step = _target_step(protocol, step, config.max_steps)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            flow_started = time.perf_counter()
            flow_delta = predict_embedding_delta(
                velocity_model,
                embedding_before,
                step,
                target_step,
                normalizer_mean,
                normalizer_scale,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            flow_inference_seconds += time.perf_counter() - flow_started
            optimizer_weight, flow_weight = _mix_weights(protocol, step)
            mixed_delta, diagnostics = mix_embedding_deltas(
                optimizer_delta,
                flow_delta,
                optimizer_weight,
                flow_weight,
                protocol.norm_match,
            )
            with torch.no_grad():
                model.token_embedding.weight.copy_(embedding_before + mixed_delta)
            intervention_rows.append(
                {
                    "step": step,
                    "target_step": target_step,
                    "optimizer_weight": optimizer_weight,
                    "flow_weight": flow_weight,
                    **diagnostics,
                }
            )

        if step == 1 or step % config.log_every == 0:
            model.eval()
            train_loss, train_acc = evaluate(model, train_x, train_y)
            val_loss, val_acc = evaluate(model, val_x, val_y)
            rows.append(
                {
                    "protocol": protocol.name,
                    "seed": seed,
                    "step": step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            if memorization_step is None and train_acc >= 0.99:
                memorization_step = step
            if train_acc >= 0.99 and val_acc >= config.grok_accuracy:
                grok_candidates.append(step)
                if (
                    len(grok_candidates) >= config.grok_patience_logs
                    and grok_step is None
                ):
                    grok_step = grok_candidates[-config.grok_patience_logs]
            else:
                grok_candidates.clear()
        if grok_step is not None and step >= grok_step + config.stabilization_steps:
            break

    final = rows[-1]
    summary = {
        "protocol": protocol.name,
        "protocol_config": asdict(protocol),
        "seed": seed,
        "memorization_step": memorization_step,
        "grok_step": grok_step,
        "final_step": final["step"],
        "final_train_acc": final["train_acc"],
        "final_val_acc": final["val_acc"],
        "interventions": len(intervention_rows),
        "mean_optimizer_flow_cosine": (
            sum(row["optimizer_flow_cosine"] for row in intervention_rows)
            / len(intervention_rows)
            if intervention_rows
            else None
        ),
        "mean_raw_flow_to_optimizer_norm_ratio": (
            sum(
                row["raw_flow_delta_norm"] / max(row["optimizer_delta_norm"], 1e-12)
                for row in intervention_rows
            )
            / len(intervention_rows)
            if intervention_rows
            else None
        ),
        "wall_seconds": time.perf_counter() - started,
        "flow_inference_seconds": flow_inference_seconds,
        "weights_saved": False,
    }
    return pd.DataFrame(rows), {"summary": summary, "interventions": intervention_rows}


def _plot_suite(metrics: pd.DataFrame, output_file: Path) -> None:
    grouped = metrics.groupby(["protocol", "step"], as_index=False).agg(
        val_acc=("val_acc", "mean"), train_acc=("train_acc", "mean")
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    for protocol, frame in grouped.groupby("protocol"):
        axes[0].plot(frame["step"], frame["val_acc"], label=protocol)
        axes[1].plot(frame["step"], frame["train_acc"], label=protocol)
    axes[0].set(
        title="Held-out modular-addition accuracy",
        xlabel="Full-batch optimizer step",
        ylabel="Validation accuracy",
        ylim=(-0.02, 1.02),
    )
    axes[1].set(
        title="Training accuracy",
        xlabel="Full-batch optimizer step",
        ylabel="Training accuracy",
        ylim=(-0.02, 1.02),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def run_hybrid_suite(
    velocity_model: TokenEmbeddingMeanFlow,
    normalizer_mean: float,
    normalizer_scale: float,
    output_dir: Path,
    device: str = "cuda",
    protocols_override: list[HybridProtocol] | None = None,
    seeds_override: list[int] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = GrokConfig(max_steps=20_000, stabilization_steps=1_000)
    default_protocols = [
        HybridProtocol("optimizer_only"),
        HybridProtocol(
            "raw_equal_fixed_9000",
            schedule="repeated",
            target_mode="fixed",
            norm_match=False,
        ),
        HybridProtocol(
            "norm_flow25_every100_fixed_9000",
            schedule="repeated",
            target_mode="fixed",
            norm_match=True,
            optimizer_weight=0.75,
            flow_weight=0.25,
        ),
        HybridProtocol(
            "norm_flow50_every100_fixed_9000",
            schedule="repeated",
            target_mode="fixed",
            norm_match=True,
        ),
        HybridProtocol(
            "norm_flow75_every100_fixed_9000",
            schedule="repeated",
            target_mode="fixed",
            norm_match=True,
            optimizer_weight=0.25,
            flow_weight=0.75,
        ),
        HybridProtocol(
            "norm_flow50_every250_fixed_9000",
            schedule="repeated",
            target_mode="fixed",
            interval=250,
            norm_match=True,
        ),
        HybridProtocol(
            "norm_flow50_every500_fixed_9000",
            schedule="repeated",
            target_mode="fixed",
            interval=500,
            norm_match=True,
        ),
        HybridProtocol(
            "norm_equal_rolling_2000",
            schedule="repeated",
            target_mode="rolling",
            norm_match=True,
        ),
        HybridProtocol(
            "one_shot_norm_fixed_9000",
            schedule="once",
            target_mode="fixed",
            norm_match=True,
        ),
    ]
    protocols = (
        protocols_override if protocols_override is not None else default_protocols
    )
    seeds = seeds_override if seeds_override is not None else [5_000, 5_001, 5_002]
    all_metrics, summaries = [], []
    velocity_model.eval()
    for protocol in protocols:
        for seed in seeds:
            metrics, payload = train_hybrid_run(
                velocity_model,
                normalizer_mean,
                normalizer_scale,
                protocol,
                seed,
                config,
                device,
            )
            all_metrics.append(metrics)
            summaries.append(payload["summary"])
            pd.DataFrame(payload["interventions"]).to_csv(
                output_dir / f"interventions_{protocol.name}_seed_{seed}.csv",
                index=False,
            )
            print(
                f"hybrid protocol={protocol.name} seed={seed} "
                f"grok={payload['summary']['grok_step']} val={payload['summary']['final_val_acc']:.4f}",
                flush=True,
            )
    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    aggregate = summary_frame.groupby("protocol", as_index=False).agg(
        trials=("seed", "count"),
        grokked=("grok_step", lambda values: int(values.notna().sum())),
        mean_grok_step=("grok_step", "mean"),
        std_grok_step=("grok_step", "std"),
        mean_final_val_acc=("final_val_acc", "mean"),
        mean_wall_seconds=("wall_seconds", "mean"),
        mean_flow_inference_seconds=("flow_inference_seconds", "mean"),
        mean_interventions=("interventions", "mean"),
        mean_optimizer_flow_cosine=("mean_optimizer_flow_cosine", "mean"),
        mean_raw_flow_to_optimizer_norm_ratio=(
            "mean_raw_flow_to_optimizer_norm_ratio",
            "mean",
        ),
    )
    metrics_frame.to_csv(output_dir / "training_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "trial_summary.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)
    _plot_suite(metrics_frame, output_dir / "hybrid_accuracy_curves.png")
    result = {
        "grok_config": config.to_dict(),
        "seeds": seeds,
        "protocols": [asdict(protocol) for protocol in protocols],
        "weights_saved": False,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
