from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightmf.transformer_weight_meanflow_training import (
    TransformerWeightTrainingConfig,
    train_transformer_weight_meanflow,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = TransformerWeightTrainingConfig(
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        runs_per_batch=args.batch_size,
    )
    summary = train_transformer_weight_meanflow(
        args.dataset_dir, args.output_dir, config, args.device
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
