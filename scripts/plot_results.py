"""Regenerate the README figures from the recorded CSVs. No training is launched."""

from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)


def finish(fig, ax, path, log=False):
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.001, 1.05)
        fig.savefig(path.with_name(path.stem + "_log_y.png"), dpi=170)
    plt.close(fig)


def modular():
    root = R / "results/modular_transfer"
    a = pd.read_csv(root / "aggregate_summary.csv")
    m = pd.read_csv(root / "training_metrics.csv")
    tasks = [(str(r.task), int(r.modulus)) for r in a.itertuples()]
    tasks = list(dict.fromkeys(tasks))
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{t.replace('_', ' ')}\nmod {p}" for t, p in tasks]
    for offset, proto, label, color in [
        (-0.18, "optimizer_only", "AdamW", "#6b7280"),
        (0.18, "addition_transformer_imf_full_reset", "Addition iMF", "#2563eb"),
    ]:
        vals = [
            float(
                a[
                    (a.task == t) & (a.modulus == p) & (a.protocol == proto)
                ].restricted_mean_grok_step.iloc[0]
            )
            for t, p in tasks
        ]
        ax.bar(
            [i + offset for i in range(len(tasks))],
            vals,
            width=0.36,
            label=label,
            color=color,
        )
    ax.set_xticks(range(len(tasks)), labels, fontsize=8)
    ax.set_ylabel("Mean threshold step (20,000-step censoring)")
    finish(fig, ax, R / "figures/modular_transfer/cross_task_time_to_grok.png")
    for t, p in tasks:
        fig, ax = plt.subplots(figsize=(8, 4))
        for proto, label in [
            ("optimizer_only", "AdamW"),
            ("addition_transformer_imf_full_reset", "Addition iMF"),
        ]:
            runs = (
                m[(m.task == t) & (m.modulus == p) & (m.protocol == proto)]
                .pivot(index="step", columns="seed", values="val_acc")
                .sort_index()
                .ffill()
            )
            q = runs.mean(axis=1)
            ax.plot(q.index, q, label=label)
        ax.set(
            xlabel="Full-batch optimizer step",
            ylabel="Validation accuracy",
            title=f"{t.replace('_', ' ').title()} mod {p}",
            ylim=(0, 1.02),
        )
        finish(fig, ax, R / f"figures/modular_transfer/p{p}_{t}_accuracy.png")


def curves(paths, title, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, files in paths.items():
        x = pd.concat(
            [pd.read_csv(p).set_index("step").val_accuracy for p in files], axis=1
        )
        mu = x.mean(axis=1)
        sd = x.std(axis=1)
        ax.plot(
            x.index,
            mu,
            label={
                "reset_only": "Reset only",
                "imf_reset": "iMF + reset",
                "imf_preserve": "iMF (preserve state)",
                "random_matched": "Matched random",
            }.get(label, label),
        )
        ax.fill_between(x.index, (mu - sd).clip(0, 1), (mu + sd).clip(0, 1), alpha=0.1)
    ax.set(
        xlabel="Stage 2 optimizer updates",
        ylabel="Validation exact-answer accuracy",
        title=title,
        ylim=(0, 1.02),
    )
    finish(fig, ax, path, True)


def main():
    modular()
    seeds = range(101, 106)
    c = R / "results/curriculum"
    controls = R / "results/controls"
    s = R / "results/source_sampling"
    h = R / "results/horizons"
    baseline = [
        c / f"seed{x}_strength000_interval500/stage2/metrics.csv" for x in seeds
    ]
    reference = [
        c / f"seed{x}_strength010_interval500/stage2/metrics.csv" for x in seeds
    ]
    curves(
        {"AdamW": baseline, "iMF": reference},
        "Arithmetic curriculum: mean and one SD over five seeds",
        R / "figures/curriculum/paired_accuracy_linear.png",
    )
    curves(
        {
            "AdamW": baseline,
            **{
                a: [controls / f"seed{x}_{a}/stage2/metrics.csv" for x in seeds]
                for a in ["reset_only", "imf_reset", "imf_preserve", "random_matched"]
            },
        },
        "Causal controls: five paired seeds",
        R / "figures/source_sampling/causal_controls_accuracy.png",
    )
    curves(
        {
            "AdamW": baseline,
            "Transition source": reference,
            "Stratified original": [
                s / f"seed{x}_strength010_interval500/stage2/metrics.csv" for x in seeds
            ],
            **{
                f"Stratified +{gap}": [
                    h / f"seed{x}_horizon{gap}/stage2/metrics.csv" for x in seeds
                ]
                for gap in [100, 200]
            },
        },
        "Source and endpoint horizon: five paired seeds",
        R / "figures/horizons/validation_linear.png",
    )
    root = R / "results/source_scaling"
    frames = []
    for size in (50, 150, 500):
        x = pd.read_csv(root / f"hybrid_{size:04d}_runs/trial_summary.csv")
        x["source_runs"] = size
        x["capped_step"] = x.grok_step.fillna(20000)
        frames.append(x)
    d = pd.concat(frames)
    table = (
        d.groupby(["source_runs", "protocol"])
        .agg(
            mean_grok_step=("capped_step", "mean"),
            std_grok_step=("capped_step", "std"),
            successes=("grok_step", "count"),
            trials=("seed", "count"),
        )
        .reset_index()
    )
    table.to_csv(root / "censor_aware_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for proto, label in [
        ("raw_optimizer10_flow90_fixed_9000", "10% optimizer / 90% flow"),
        ("raw_equal_fixed_9000", "50% optimizer / 50% flow"),
    ]:
        q = table[table.protocol == proto].sort_values("source_runs")
        ax.plot(q.source_runs, q.mean_grok_step, "o-", label=label)
    baseline = float(table[table.protocol == "optimizer_only"].mean_grok_step.iloc[0])
    ax.axhline(baseline, color="gray", linestyle="--", label="AdamW reference")
    ax.set(
        xlabel="Source runs",
        ylabel="Mean threshold step (20,000-step censoring)",
        title="Embedding-only source-data study",
    )
    ax.set_xticks([50, 150, 500])
    finish(
        fig, ax, R / "figures/source_scaling/checkpoint_data_scaling_censor_aware.png"
    )


if __name__ == "__main__":
    main()
