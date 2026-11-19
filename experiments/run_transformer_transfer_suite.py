from __future__ import annotations

import argparse
from pathlib import Path

from weightmf.transformer_weight_downstream import run_transformer_transfer_suite


DEFAULT_TASKS = (
    "addition:97",
    "subtraction:113",
    "subtraction:97",
    "multiplication:113",
    "multiplication:97",
    "division:97",
    "quadratic_symmetric:97",
    "bitwise_xor:97",
)


def parse_task(value: str) -> tuple[str, int]:
    try:
        task, modulus = value.rsplit(":", 1)
        return task, int(modulus)
    except ValueError as error:
        raise argparse.ArgumentTypeError("task must use OPERATION:MODULUS") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", type=parse_task)
    parser.add_argument("--first-seed", type=int, default=31_000)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tasks = args.task or [parse_task(value) for value in DEFAULT_TASKS]
    seeds = list(range(args.first_seed, args.first_seed + args.trials))
    run_transformer_transfer_suite(
        args.artifact.resolve(),
        args.output_dir.resolve(),
        seeds,
        tasks,
        args.device,
    )


if __name__ == "__main__":
    main()
