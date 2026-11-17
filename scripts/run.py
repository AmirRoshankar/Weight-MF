"""Run the experiments used by the README. Outputs never overwrite recorded results."""

import argparse, importlib, json, runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/curriculum")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=[
            "collect-source",
            "collect-embedding",
            "recollect-source",
            "train-generator",
            "modular",
            "curriculum",
            "controls",
            "source-sampling",
            "horizons",
            "source-scaling",
            "plots",
        ],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--source", choices=["transition", "stratified"], default="transition"
    )
    parser.add_argument("--device", default="cuda")
    args, extra = parser.parse_known_args()
    out = args.output or ROOT / "runs" / args.experiment

    def invoke(file, argv):
        sys.argv = [file] + [str(x) for x in argv] + extra
        runpy.run_path(str(ROOT / file), run_name="__main__")

    if args.experiment == "plots":
        invoke("scripts/plot_results.py", [])
        return
    if args.experiment.startswith("collect-"):
        from grokflow.config import GrokConfig
        from grokflow.embedding_pair_dataset import (
            EmbeddingPairCollectionConfig,
            collect_embedding_pair_dataset,
        )

        if args.experiment == "collect-embedding":
            collect_embedding_pair_dataset(
                GrokConfig(),
                EmbeddingPairCollectionConfig(max_runs=1102),
                args.dataset or ROOT / "data/embedding",
                args.device,
            )
            return
        from grokflow.transformer_weight_dataset import (
            TransformerWeightCollectionConfig,
            collect_one_transformer_weight_shard,
            canonical_row_layout,
            TRANSFORMER_WEIGHT_NAMES,
        )

        dataset = args.dataset or ROOT / "data/source_transition"
        dataset.mkdir(parents=True, exist_ok=True)
        manifest_file = dataset / "manifest.json"
        manifest = (
            json.loads(manifest_file.read_text())
            if manifest_file.exists()
            else dict(
                format_version=1,
                scope="transformer_block_weights",
                canonical_shape=[1536, 128],
                canonical_row_layout=canonical_row_layout(),
                weight_names=list(TRANSFORMER_WEIGHT_NAMES),
                runs=[],
            )
        )
        completed = {x["seed"] for x in manifest["runs"]}
        for seed in range(20000, 20602):
            if seed in completed:
                continue
            shard = collect_one_transformer_weight_shard(
                GrokConfig(),
                TransformerWeightCollectionConfig(),
                seed,
                dataset / f"transformer_pairs_seed_{seed}.pt",
                args.device,
            )
            if shard is None:
                raise RuntimeError(
                    f"Seed {seed} did not grok; do not substitute another source seed silently."
                )
            manifest["runs"].append(shard)
            manifest_file.write_text(json.dumps(manifest, indent=2))
        return
    if args.experiment == "recollect-source":
        invoke("experiments/recollect_sources.py", [])
        return
    if args.experiment == "train-generator":
        invoke(
            "experiments/train_transformer_weight_meanflow.py",
            [
                "--dataset-dir",
                args.dataset or ROOT / f"data/source_{args.source}",
                "--output-dir",
                args.output or ROOT / f"artifacts/{args.source}",
                "--device",
                args.device,
            ],
        )
        return
    if args.experiment == "modular":
        invoke(
            "experiments/run_transformer_transfer_suite.py",
            [
                "--artifact",
                args.artifact or ROOT / "artifacts/transition/best_model.pt",
                "--output-dir",
                out,
                "--device",
                args.device,
            ],
        )
        return
    if args.experiment == "source-scaling":
        invoke(
            "experiments/run_checkpoint_data_scaling.py",
            [
                "--dataset-dir",
                args.dataset or ROOT / "data/embedding",
                "--output-root",
                out,
                "--source-sizes",
                "50",
                "150",
                "500",
                "--validation-runs",
                "51",
                "--test-runs",
                "51",
            ],
        )
        return
    if args.device != "cuda":
        parser.error("The historical curriculum protocol requires CUDA.")
    import engine as c

    c.BASE = ROOT
    c.DATA = ROOT / "data/precedence"
    c.ART = args.artifact or ROOT / "artifacts/transition/best_model.pt"
    c.OUT = out
    import paired as v

    v.ROOT = out
    if args.experiment == "curriculum":
        simple = c.stage1_data()
        composed = c.prepare(c.DATA)
        v.CASES = [(s, a, 500) for s in range(101, 106) for a in (0.0, 0.1)]
        for seed, strength, interval in v.CASES:
            v.run(
                f"seed{seed}_strength{int(strength * 100):03d}_interval500",
                seed,
                strength,
                interval,
                20000,
                simple,
                composed,
            )
        return
    name = {
        "controls": "controls",
        "source-sampling": "source_sampling",
        "horizons": "horizons",
    }[args.experiment]
    sys.argv = [name] + extra
    module = importlib.import_module(name)
    module.ROOT = out
    module.main()


if __name__ == "__main__":
    main()
