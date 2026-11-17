from __future__ import annotations

import json
import math
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
from .meanflow_objective import sample_improved_meanflow
from .model import GrokTransformer
from .operation_data import make_binary_operation
from .train import evaluate, seed_everything
from .transformer_weight_dataset import TRANSFORMER_WEIGHT_NAMES
from .transformer_weight_meanflow import (
    ConditionalTransformerWeightGenerator,
    TransformerWeightGeneratorConfig,
    canonical_tensor_and_row_ids,
)
from .transformer_weight_meanflow_training import (
    denormalize_transformer_rows,
    unpack_canonical_rows,
)


@dataclass(frozen=True)
class TransformerIntervention:
    name: str
    generator_weight: float = 0.0
    intervention_step: int = 500
    target_step: int = 9_000
    reset_optimizer_moments: bool = False


def _current_rows(model: GrokTransformer) -> torch.Tensor:
    state = model.state_dict()
    rows = []
    for name in TRANSFORMER_WEIGHT_NAMES:
        value = state[name].detach().float()
        rows.append(value.T if name == "mlp_out.weight" else value)
    return torch.cat(rows, dim=0)


def _normalize_rows(
    rows: torch.Tensor, means: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    tensor_ids, _ = canonical_tensor_and_row_ids()
    ids = tensor_ids.to(rows.device)
    return (rows - means.to(rows.device)[ids, None]) / scales.to(rows.device)[ids, None]


def load_transformer_generator(
    path: Path, device: str
) -> tuple[ConditionalTransformerWeightGenerator, dict]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = ConditionalTransformerWeightGenerator(
        TransformerWeightGeneratorConfig(**artifact["model_config"])
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact


@torch.no_grad()
def generate_transformer_endpoint(
    generator_model: ConditionalTransformerWeightGenerator,
    artifact: dict,
    task_model: GrokTransformer,
    current_step: int,
    target_step: int,
    training_loss: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    current = _current_rows(task_model)
    means = artifact["normalizer_means"].float()
    scales = artifact["normalizer_scales"].float()
    early = _normalize_rows(current, means, scales).unsqueeze(0)
    start_step = torch.tensor([[current_step / 30_000.0]], device=current.device)
    end_step = torch.tensor([[target_step / 30_000.0]], device=current.device)
    start_metric = torch.tensor(
        [[math.log1p(training_loss) / math.log1p(5.0)]], device=current.device
    )
    target_metric = torch.zeros_like(start_metric)
    random = torch.Generator(device=current.device).manual_seed(seed)
    prediction, nfe = sample_improved_meanflow(
        generator_model,
        early,
        start_step,
        end_step,
        start_metric,
        target_metric,
        random,
    )
    endpoint_rows = denormalize_transformer_rows(prediction.squeeze(0), means, scales)
    endpoint = unpack_canonical_rows(endpoint_rows)
    delta = endpoint_rows - current
    return endpoint, {
        "generator_nfe": nfe,
        "raw_delta_norm": float(delta.norm()),
        "current_weight_norm": float(current.norm()),
        "relative_delta_norm": float(delta.norm() / current.norm().clamp_min(1e-12)),
    }


def train_target(
    protocol: TransformerIntervention,
    generator_model: ConditionalTransformerWeightGenerator | None,
    artifact: dict | None,
    seed: int,
    device: str,
    task: str = "addition",
    modulus: int = 113,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    config = GrokConfig(modulus=modulus, max_steps=20_000, stabilization_steps=1_000)
    seed_everything(seed)
    data = (
        make_modular_addition(config, split_seed=seed)
        if task == "addition"
        else make_binary_operation(config, split_seed=seed, operation=task)
    )
    train_x, train_y = data.train_x.to(device), data.train_y.to(device)
    val_x, val_y = data.val_x.to(device), data.val_y.to(device)
    model = GrokTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    parameter_lookup = dict(model.named_parameters())
    metrics = []
    interventions = []
    grok_candidates = []
    grok_step = memorization_step = None
    started = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if protocol.generator_weight > 0 and step == protocol.intervention_step:
            assert generator_model is not None and artifact is not None
            model.eval()
            current_train_loss, _ = evaluate(model, train_x, train_y)
            before = _current_rows(model)
            endpoint, diagnostics = generate_transformer_endpoint(
                generator_model,
                artifact,
                model,
                step,
                protocol.target_step,
                current_train_loss,
                seed * 100_000 + step,
            )
            with torch.no_grad():
                for name, generated in endpoint.items():
                    parameter = parameter_lookup[name]
                    blended = parameter + protocol.generator_weight * (
                        generated.to(parameter.device, parameter.dtype) - parameter
                    )
                    parameter.copy_(blended)
                    if protocol.reset_optimizer_moments:
                        optimizer.state[parameter].clear()
            after = _current_rows(model)
            interventions.append(
                {
                    "protocol": protocol.name,
                    "task": task,
                    "modulus": modulus,
                    "seed": seed,
                    "step": step,
                    "applied_delta_norm": float((after - before).norm()),
                    **diagnostics,
                }
            )
        if step == 1 or step % config.log_every == 0:
            model.eval()
            train_loss, train_acc = evaluate(model, train_x, train_y)
            val_loss, val_acc = evaluate(model, val_x, val_y)
            metrics.append(
                {
                    "protocol": protocol.name,
                    "task": task,
                    "modulus": modulus,
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
    summary = {
        "protocol": protocol.name,
        "task": task,
        "modulus": modulus,
        "seed": seed,
        "grok_step": grok_step,
        "censored_grok_step": grok_step if grok_step is not None else config.max_steps,
        "memorization_step": memorization_step,
        "final_val_acc": metrics[-1]["val_acc"],
        "wall_seconds": time.perf_counter() - started,
    }
    return pd.DataFrame(metrics), summary, pd.DataFrame(interventions)


def _plot(metrics: pd.DataFrame, output: Path, title: str = "Modular addition") -> None:
    grouped = metrics.groupby(["protocol", "step"], as_index=False).agg(
        val_acc=("val_acc", "mean"), train_acc=("train_acc", "mean")
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for name, frame in grouped.groupby("protocol"):
        axes[0].plot(frame.step, frame.val_acc, label=name)
        axes[1].plot(frame.step, frame.train_acc, label=name)
    axes[0].set_title(f"Held-out {title} accuracy")
    axes[1].set_title("Training accuracy")
    for axis in axes:
        axis.set(xlabel="Full-batch step", ylabel="Accuracy", ylim=(-0.02, 1.02))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def run_transformer_downstream(
    artifact_path: Path,
    output_dir: Path,
    seeds: list[int],
    device: str = "cuda",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "runs"
    run_dir.mkdir(exist_ok=True)
    generator_model, artifact = load_transformer_generator(artifact_path, device)
    protocols = [
        TransformerIntervention("optimizer_only"),
        TransformerIntervention("transformer_imf_one_shot_full_preserve", 1.0),
        TransformerIntervention(
            "transformer_imf_one_shot_full_reset", 1.0, reset_optimizer_moments=True
        ),
        TransformerIntervention("transformer_imf_one_shot_half_preserve", 0.5),
    ]
    all_metrics = []
    summaries = []
    for seed in seeds:
        for protocol in protocols:
            stem = f"{protocol.name}_seed_{seed}"
            metric_path = run_dir / f"metrics_{stem}.csv"
            summary_path = run_dir / f"summary_{stem}.json"
            intervention_path = run_dir / f"interventions_{stem}.csv"
            if metric_path.exists() and summary_path.exists():
                metrics = pd.read_csv(metric_path)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                metrics, summary, interventions = train_target(
                    protocol,
                    generator_model if protocol.generator_weight else None,
                    artifact if protocol.generator_weight else None,
                    seed,
                    device,
                )
                metrics.to_csv(metric_path, index=False)
                interventions.to_csv(intervention_path, index=False)
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            all_metrics.append(metrics)
            summaries.append(summary)
            print(
                f"transformer downstream {protocol.name} seed={seed} grok={summary['grok_step']}",
                flush=True,
            )
    metric_frame = pd.concat(all_metrics, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    aggregate = summary_frame.groupby("protocol", as_index=False).agg(
        trials=("seed", "count"),
        grokked=("grok_step", lambda values: int(values.notna().sum())),
        restricted_mean_grok_step=("censored_grok_step", "mean"),
        std_censored_grok_step=("censored_grok_step", "std"),
        mean_final_val_acc=("final_val_acc", "mean"),
    )
    metric_frame.to_csv(output_dir / "training_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "trial_summary.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)
    _plot(metric_frame, output_dir / "accuracy_curves.png")
    config = {
        "artifact": str(artifact_path),
        "seeds": seeds,
        "protocols": [asdict(protocol) for protocol in protocols],
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return config


def run_transformer_transfer_suite(
    artifact_path: Path,
    output_dir: Path,
    seeds: list[int],
    tasks: list[tuple[str, int]],
    device: str = "cuda",
) -> dict:
    """Evaluate an addition-trained transformer iMF on held-out task families.

    Each paired seed uses the same initialization and data split for the
    optimizer-only and full-reset protocols.  Only the attention and MLP
    weights are generated; task-specific embeddings and unembedding remain
    under ordinary optimization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "runs"
    run_dir.mkdir(exist_ok=True)
    generator_model, artifact = load_transformer_generator(artifact_path, device)
    protocols = [
        TransformerIntervention("optimizer_only"),
        TransformerIntervention(
            "addition_transformer_imf_full_reset",
            1.0,
            reset_optimizer_moments=True,
        ),
    ]
    status_path = output_dir / "status.json"
    target_runs = len(tasks) * len(seeds) * len(protocols)
    completed_runs = 0
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "completed": completed_runs,
                "target": target_runs,
                "current_task": None,
                "current_modulus": None,
                "current_protocol": None,
                "current_seed": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    all_metrics: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for task, modulus in tasks:
        for seed in seeds:
            for protocol in protocols:
                stem = f"p{modulus}_{task}_{protocol.name}_seed_{seed}"
                metric_path = run_dir / f"metrics_{stem}.csv"
                summary_path = run_dir / f"summary_{stem}.json"
                intervention_path = run_dir / f"interventions_{stem}.csv"
                if metric_path.exists() and summary_path.exists():
                    metrics = pd.read_csv(metric_path)
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    metrics, summary, interventions = train_target(
                        protocol,
                        generator_model if protocol.generator_weight else None,
                        artifact if protocol.generator_weight else None,
                        seed,
                        device,
                        task=task,
                        modulus=modulus,
                    )
                    metrics.to_csv(metric_path, index=False)
                    interventions.to_csv(intervention_path, index=False)
                    summary_path.write_text(
                        json.dumps(summary, indent=2), encoding="utf-8"
                    )
                all_metrics.append(metrics)
                summaries.append(summary)
                completed_runs += 1
                status_path.write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "completed": completed_runs,
                            "target": target_runs,
                            "current_task": task,
                            "current_modulus": modulus,
                            "current_protocol": protocol.name,
                            "current_seed": seed,
                            "latest_grok_step": summary["grok_step"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"transfer p={modulus} task={task} protocol={protocol.name} "
                    f"seed={seed} grok={summary['grok_step']}",
                    flush=True,
                )
        task_metrics = pd.concat(
            [
                frame
                for frame in all_metrics
                if frame["task"].iloc[0] == task
                and int(frame["modulus"].iloc[0]) == modulus
            ],
            ignore_index=True,
        )
        _plot(
            task_metrics,
            output_dir / f"p{modulus}_{task}_accuracy.png",
            title=f"{task} (mod {modulus})",
        )

    metric_frame = pd.concat(all_metrics, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    aggregate = summary_frame.groupby(
        ["modulus", "task", "protocol"], as_index=False
    ).agg(
        trials=("seed", "count"),
        grokked=("grok_step", lambda values: int(values.notna().sum())),
        restricted_mean_grok_step=("censored_grok_step", "mean"),
        std_censored_grok_step=("censored_grok_step", "std"),
        median_censored_grok_step=("censored_grok_step", "median"),
        mean_final_val_acc=("final_val_acc", "mean"),
    )
    metric_frame.to_csv(output_dir / "training_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "trial_summary.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)

    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    labels = [f"p={p}\n{task}" for task, p in tasks]
    x = range(len(tasks))
    width = 0.36
    for offset, protocol in zip((-width / 2, width / 2), protocols):
        values = []
        for task, modulus in tasks:
            row = aggregate[
                (aggregate.task == task)
                & (aggregate.modulus == modulus)
                & (aggregate.protocol == protocol.name)
            ]
            values.append(float(row.restricted_mean_grok_step.iloc[0]))
        axis.bar([value + offset for value in x], values, width, label=protocol.name)
    axis.set_xticks(list(x), labels)
    axis.set_ylabel("Restricted mean grok step (20k censoring)")
    axis.set_title("Addition-trained transformer iMF: cross-task transfer")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "cross_task_time_to_grok.png", dpi=170)
    plt.close(figure)

    config = {
        "artifact": str(artifact_path),
        "source_task": "addition_mod_113",
        "seeds": seeds,
        "tasks": [{"task": task, "modulus": modulus} for task, modulus in tasks],
        "protocols": [asdict(protocol) for protocol in protocols],
        "pairing": "same seed, initialization, and dataset split within each task",
        "generated_scope": list(TRANSFORMER_WEIGHT_NAMES),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed": completed_runs,
                "target": target_runs,
                "current_task": None,
                "current_modulus": None,
                "current_protocol": None,
                "current_seed": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config
