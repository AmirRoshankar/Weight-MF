"""Whole-integer arithmetic expressions with verified standard precedence."""

import json, random
from pathlib import Path
from fractions import Fraction
from collections import Counter
import torch

OPS = {"+": 100, "-": 101, "/": 102, "*": 103, "=": 104}
BOS, EOS, PAD = 105, 106, 107


def calculate(numbers, operators):
    # Collapse multiplication/division in order, then addition/subtraction.
    terms = [numbers[0]]
    low = []
    for op, n in zip(operators, numbers[1:]):
        if op == "*":
            terms[-1] *= n
        elif op == "/":
            if n == 0 or terms[-1] % n:
                return None
            terms[-1] //= n
        else:
            low.append(op)
            terms.append(n)
        if not 0 <= terms[-1] <= 99:
            return None
    result = terms[0]
    for op, n in zip(low, terms[1:]):
        result = result + n if op == "+" else result - n
        if not 0 <= result <= 99:
            return None
    return result


def fraction_reference(numbers, operators):
    # Independent sum-of-signed-products evaluator using exact fractions.
    total = Fraction(0)
    term = Fraction(numbers[0])
    sign = 1
    for op, n in zip(operators, numbers[1:]):
        if op in "+-":
            total += sign * term
            sign = 1 if op == "+" else -1
            term = Fraction(n)
        elif op == "*":
            term *= n
        else:
            term /= n
    return total + sign * term


def prepare(root: Path, seed=20261111):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "metadata.json").exists():
        return {
            s: [json.loads(l) for l in (root / f"{s}.jsonl").read_text().splitlines()]
            for s in ("train", "val", "holdout")
        }
    rng = random.Random(seed)
    seen = set()
    sets = {}
    for split, count in [("train", 32000), ("val", 6400), ("holdout", 10000)]:
        quota = count // 200
        buckets = Counter()
        examples = []
        attempts = 0
        while len(examples) < count:
            attempts += 1
            if attempts > 30000000:
                raise RuntimeError(f"generation exhausted {split}: {len(examples)}")
            n = rng.randint(3, 6)
            numbers = [rng.randrange(rng.choice([10, 20, 100])) for _ in range(n)]
            operators = rng.choices(["+", "-", "/", "*"], weights=[3, 3, 2, 2], k=n - 1)
            if not any(o in "+-" for o in operators) or not any(
                o in "*/" for o in operators
            ):
                continue
            if any(o == "/" and b == 0 for o, b in zip(operators, numbers[1:])):
                continue
            answer = calculate(numbers, operators)
            if answer is None:
                continue
            sequential = Fraction(numbers[0])
            for op, b in zip(operators, numbers[1:]):
                if op == "+":
                    sequential += b
                elif op == "-":
                    sequential -= b
                elif op == "*":
                    sequential *= b
                else:
                    sequential /= b
            critical = sequential != answer
            bucket = (answer, critical)
            if buckets[bucket] >= quota:
                continue
            ids = [BOS, numbers[0]]
            for op, b in zip(operators, numbers[1:]):
                ids.extend([OPS[op], b])
            ids.append(OPS["="])
            key = tuple(ids)
            if key in seen:
                continue
            assert fraction_reference(numbers, operators) == answer
            seen.add(key)
            buckets[bucket] += 1
            expression = " ".join(
                [
                    f"{numbers[0]:02d}",
                    *[
                        part
                        for o, b in zip(operators, numbers[1:])
                        for part in (o, f"{b:02d}")
                    ],
                ]
            )
            examples.append(
                dict(
                    expression=expression,
                    prompt_ids=ids,
                    answer=answer,
                    precedence_critical=critical,
                    operands=n,
                    has_division="/" in operators,
                )
            )
        rng.shuffle(examples)
        sets[split] = examples
        (root / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(e) for e in examples) + "\n"
        )
        print(f"dataset {split}: {len(examples)}, attempts={attempts}", flush=True)
    metadata = {
        "seed": seed,
        "vocab": {
            **{f"{i:02d}": i for i in range(100)},
            **OPS,
            "BOS": BOS,
            "EOS": EOS,
            "PAD": PAD,
        },
        "model_vocab_padded": 128,
        "operand_range": "00..99",
        "intermediate_and_answer_range": "00..99",
        "division": "nonzero divisor and exact integral quotient",
        "precedence": "*/ equal and left associative; then +- equal and left associative; no parentheses/exponents",
        "split_counts": {k: len(v) for k, v in sets.items()},
        "precedence_critical_fraction": 0.5,
        "balanced_answers": True,
        "duplicate_expressions_across_splits": 0,
        "division_example_counts": {
            k: sum(e["has_division"] for e in v) for k, v in sets.items()
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    torch.save(
        torch.tensor(
            [t for e in sets["train"] for t in e["prompt_ids"] + [e["answer"], EOS]],
            dtype=torch.int32,
        ),
        root / "train_tokens.pt",
    )
    torch.save(
        torch.tensor(
            [t for e in sets["val"] for t in e["prompt_ids"] + [e["answer"], EOS]],
            dtype=torch.int32,
        ),
        root / "val_tokens.pt",
    )
    return sets
