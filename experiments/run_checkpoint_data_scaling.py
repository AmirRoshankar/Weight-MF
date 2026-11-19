from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from weightmf.embedding_velocity_investigation import (
    VelocityInvestigationConfig,
    make_run_split,
    run_velocity_investigation,
)
from weightmf.hybrid_optimizer import HybridProtocol, run_hybrid_suite


SOURCE_SIZES = [50, 150, 500]
DOWNSTREAM_SEEDS = list(range(7_000, 7_020))


def _plot_scaling(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    endpoint = frame.drop_duplicates("source_runs").sort_values("source_runs")
    axes[0].plot(
        endpoint["source_runs"],
        endpoint["test_endpoint_mse"],
        marker="o",
        label="Test endpoint MSE",
    )
    axes[0].plot(
        endpoint["source_runs"],
        endpoint["test_do_nothing_mse"],
        linestyle="--",
        color="black",
        label="Do-nothing MSE",
    )
    axes[0].set(
        xlabel="Checkpoint source runs",
        ylabel="MSE",
        title="Fixed-test endpoint prediction",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for protocol, group in frame.groupby("protocol"):
        group = group.sort_values("source_runs")
        axes[1].errorbar(
            group["source_runs"],
            group["mean_threshold_step"],
            yerr=group["std_threshold_step"],
            marker="o",
            capsize=4,
            label=protocol,
        )
    axes[1].set(
        xlabel="Checkpoint source runs",
        ylabel="Mean threshold step +/- sample SD",
        title="Cross-task optimization acceleration",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/embedding"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/checkpoint_data_scaling_phase_1102"),
    )
    parser.add_argument("--sampler", default="phase_focused")
    parser.add_argument("--source-sizes", type=int, nargs="+", default=SOURCE_SIZES)
    parser.add_argument("--validation-runs", type=int, default=0)
    parser.add_argument("--test-runs", type=int, default=0)
    args = parser.parse_args()
    source_sizes = list(args.source_sizes)
    dataset_dir = args.dataset_dir
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    all_train, validation, test = make_run_split(
        manifest["runs"],
        20_261_105,
        validation_run_count=args.validation_runs,
        test_run_count=args.test_runs,
    )
    if max(source_sizes) > len(all_train):
        raise RuntimeError(
            f"Largest requested training subset is {max(source_sizes)}, but split has {len(all_train)} runs"
        )

    assisted = [
        HybridProtocol(
            "raw_equal_fixed_9000",
            schedule="repeated",
            target_mode="fixed",
            norm_match=False,
        ),
        HybridProtocol(
            "raw_optimizer10_flow90_fixed_9000",
            schedule="repeated",
            target_mode="fixed",
            norm_match=False,
            optimizer_weight=0.1,
            flow_weight=0.9,
        ),
    ]
    summaries: list[dict] = []
    baseline_done = False

    for source_runs in source_sizes:
        velocity_output = output_root / f"velocity_{source_runs:04d}_runs"
        hybrid_output = output_root / f"hybrid_{source_runs:04d}_runs"

        def callback(model, mean: float, scale: float, *, n: int = source_runs) -> None:
            nonlocal baseline_done
            protocols = list(assisted)
            if not baseline_done:
                protocols = [HybridProtocol("optimizer_only"), *protocols]
            run_hybrid_suite(
                model,
                mean,
                scale,
                hybrid_output,
                "cuda",
                protocols_override=protocols,
                seeds_override=DOWNSTREAM_SEEDS,
            )
            baseline_done = True

        config = VelocityInvestigationConfig(
            train_steps=30_600,
            batch_size=32,
            runs_per_batch=8,
            width=256,
            latent_tokens=64,
            latent_depth=4,
            heads=8,
            max_train_runs=source_runs,
            validation_run_count=args.validation_runs,
            test_run_count=args.test_runs,
        )
        velocity = run_velocity_investigation(
            dataset_dir,
            velocity_output,
            config,
            "cuda",
            trained_model_callback=callback,
        )
        aggregate = pd.read_csv(hybrid_output / "aggregate_summary.csv")
        selected_rows = all_train[:source_runs]
        source_gpu_seconds = sum(float(row["wall_seconds"]) for row in selected_rows)
        source_bytes = sum(int(row["bytes"]) for row in selected_rows)
        for row in aggregate.to_dict(orient="records"):
            summaries.append(
                {
                    "sampler": args.sampler,
                    "source_runs": source_runs,
                    "protocol": row["protocol"],
                    "downstream_trials": int(row["trials"]),
                    "mean_threshold_step": row["mean_threshold_step"],
                    "std_threshold_step": row["std_threshold_step"],
                    "mean_target_wall_seconds": row["mean_wall_seconds"],
                    "mean_flow_inference_seconds": row.get(
                        "mean_flow_inference_seconds"
                    ),
                    "best_validation_endpoint_mse": velocity[
                        "best_validation_endpoint_mse"
                    ],
                    "test_endpoint_mse": velocity["test_endpoint_mse"],
                    "test_do_nothing_mse": velocity["test_do_nothing_mse"],
                    "test_delta_cosine": velocity["test_delta_cosine"],
                    "predictor_training_seconds": velocity["training_seconds"],
                    "source_collection_gpu_seconds": source_gpu_seconds,
                    "source_storage_bytes": source_bytes,
                }
            )
        pd.DataFrame(summaries).to_csv(
            output_root / "scaling_summary_partial.csv", index=False
        )

    frame = pd.DataFrame(summaries)
    baseline = frame[frame["protocol"] == "optimizer_only"].iloc[0]
    assisted_frame = frame[frame["protocol"] != "optimizer_only"].copy()
    assisted_frame["step_reduction_vs_optimizer"] = (
        baseline["mean_threshold_step"] - assisted_frame["mean_threshold_step"]
    )
    assisted_frame["percent_reduction_vs_optimizer"] = (
        100 * assisted_frame["step_reduction_vs_optimizer"] / baseline["mean_threshold_step"]
    )
    assisted_frame["target_wall_seconds_saved"] = (
        baseline["mean_target_wall_seconds"]
        - assisted_frame["mean_target_wall_seconds"]
    )
    assisted_frame["meta_training_seconds"] = (
        assisted_frame["source_collection_gpu_seconds"]
        + assisted_frame["predictor_training_seconds"]
    )
    assisted_frame["estimated_runs_to_amortize"] = assisted_frame[
        "meta_training_seconds"
    ] / assisted_frame["target_wall_seconds_saved"].clip(lower=1e-9)
    baseline_frame = pd.DataFrame([baseline])
    final = pd.concat([baseline_frame, assisted_frame], ignore_index=True)
    final.to_csv(output_root / "scaling_summary.csv", index=False)
    _plot_scaling(
        final[final["protocol"] != "optimizer_only"],
        output_root / "checkpoint_data_scaling.png",
    )
    (output_root / "experiment_config.json").write_text(
        json.dumps(
            {
                "sampler": args.sampler,
                "source_sizes": source_sizes,
                "downstream_seeds": DOWNSTREAM_SEEDS,
                "fixed_validation_runs": len(validation),
                "fixed_test_runs": len(test),
                "nested_training_subsets": True,
                "weights_saved": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
