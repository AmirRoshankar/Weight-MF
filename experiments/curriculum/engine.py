"""Answer-only arithmetic curriculum and frozen addition-iMF experiments."""

from __future__ import annotations
import collections, json, math, random, subprocess, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from precedence_data import OPS, BOS
from modeling import (
    ReasoningModelConfig,
    compatible_block_rows,
    copy_rows_to_compatible_block,
    IMF_WEIGHT_NAMES,
)
from train import make_matched_models, load_imf, _normalize
from grokflow.meanflow_objective import sample_improved_meanflow
from grokflow.transformer_weight_meanflow_training import denormalize_transformer_rows

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "results/arithmetic_curriculum_imf_v1"
DATA = BASE / "data/precedence"
ART = BASE / "artifacts/transition/best_model.pt"
SEED = 42
OPS_NAMES = ["identity", "addition", "subtraction", "multiplication", "division"]
CONDITIONS = [
    ("curriculum_baseline", "none", "none"),
    ("one_shot_stage1", "once", "none"),
    ("repeated_stage1", "repeat", "none"),
    ("one_shot_both", "once", "once"),
    ("repeated_both", "repeat", "repeat"),
    ("direct_target", "skip", "none"),
    ("random_matched_both", "random", "random"),
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def status(state, condition, stage, step, maxstep):
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        gpu = "unavailable"
    n = len(list(OUT.glob("*/summary.json")))
    d = dict(
        state=state,
        condition=condition,
        stage=stage,
        step=step,
        stage_total=maxstep,
        completed_conditions=n,
        total_conditions=len(CONDITIONS),
        gpu=gpu,
        updated_unix=time.time(),
    )
    write(OUT / "status.json", d)
    (OUT / "PROGRESS.txt").write_text(
        f"{state} | {n}/{len(CONDITIONS)} done | {condition} | stage{stage} {step}/{maxstep} | GPU %/MiB {gpu}\n",
        encoding="utf-8",
    )


def stage1_data():
    by = collections.defaultdict(list)
    for a in range(100):
        by["identity"].append(
            dict(prompt_ids=[BOS, a, OPS["="]], answer=a, operation="identity")
        )
        for b in range(100):
            for op, name in [
                ("+", "addition"),
                ("-", "subtraction"),
                ("*", "multiplication"),
                ("/", "division"),
            ]:
                if op == "/":
                    if b == 0 or a % b:
                        continue
                    result = a // b
                elif op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                else:
                    result = a * b
                if 0 <= result <= 99:
                    by[name].append(
                        dict(
                            prompt_ids=[BOS, a, OPS[op], b, OPS["="]],
                            answer=result,
                            operation=name,
                        )
                    )
    rng = random.Random(20261112)
    splits = {x: [] for x in ("train", "val", "holdout")}
    for name in OPS_NAMES:
        items = by[name]
        rng.shuffle(items)
        n = len(items)
        i = int(n * 0.8)
        j = int(n * 0.9)
        for key, part in [
            ("train", items[:i]),
            ("val", items[i:j]),
            ("holdout", items[j:]),
        ]:
            splits[key].extend(part)
    for split in splits.values():
        rng.shuffle(split)
    keys = [
        set(tuple(e["prompt_ids"]) for e in splits[k])
        for k in ("train", "val", "holdout")
    ]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    return splits


def tensors(examples, device):
    # Left aligned; select output at '='. No EOS is included or supervised.
    length = max(len(e["prompt_ids"]) for e in examples)
    x = np.full((len(examples), length), 107, dtype=np.int64)
    for i, e in enumerate(examples):
        x[i, : len(e["prompt_ids"])] = e["prompt_ids"]
    return (
        torch.from_numpy(x).to(device),
        torch.tensor([len(e["prompt_ids"]) - 1 for e in examples], device=device),
        torch.tensor([e["answer"] for e in examples], device=device),
    )


@torch.no_grad()
def evaluate(model, data, batch=256, group=False):
    model.eval()
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    for start in range(0, len(data), batch):
        items = data[start : start + batch]
        x, pos, y = tensors(items, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)[torch.arange(len(items), device="cuda"), pos].float()
        losses = F.cross_entropy(logits, y, reduction="none").cpu().tolist()
        correct = (logits.argmax(-1) == y).cpu().tolist()
        for e, c, l in zip(items, correct, losses):
            groups = ["all"]
            if group:
                if "operation" in e:
                    groups.append(e["operation"])
                else:
                    groups.extend(
                        [
                            f"length_{e['operands']}",
                            "precedence_critical"
                            if e["precedence_critical"]
                            else "precedence_noncritical",
                        ]
                    )
            for key in groups:
                agg[key][0] += int(c)
                agg[key][1] += 1
                agg[key][2] += l
    return {
        key: dict(accuracy=v[0] / v[1], loss=v[2] / v[1], count=v[1])
        for key, v in agg.items()
    }


_BUCKETS = {}


def sample(data, rng, count, balanced=False):
    if not balanced:
        return [data[rng.randrange(len(data))] for _ in range(count)]
    if id(data) not in _BUCKETS:
        buckets = {k: [] for k in OPS_NAMES}
        for item in data:
            buckets[item["operation"]].append(item)
        _BUCKETS[id(data)] = buckets
    buckets = _BUCKETS[id(data)]
    return [
        buckets[k][rng.randrange(len(buckets[k]))]
        for k in (OPS_NAMES[rng.randrange(5)] for _ in range(count))
    ]


def lr(step, total):
    if step < 100:
        return 0.001 * (step + 1) / 100
    return 0.00006 + 0.5 * (
        1 + math.cos(math.pi * (step - 100) / max(1, total - 100))
    ) * (0.001 - 0.00006)


def update_imf(model, opt, generator, artifact, stage, step, kind, conditioning, val):
    started = time.perf_counter()
    before = evaluate(model, val)["all"]
    block = model.blocks[0]
    current = compatible_block_rows(block)
    with torch.no_grad():
        conditioning_loss = evaluate(model, conditioning)["all"]["loss"]
        target_step = min(30000, max(9000, step + 1000))
        predicted, nfe = sample_improved_meanflow(
            generator,
            _normalize(current, artifact).unsqueeze(0),
            torch.tensor([[step / 30000]], device="cuda"),
            torch.tensor([[target_step / 30000]], device="cuda"),
            torch.tensor(
                [[math.log1p(conditioning_loss) / math.log1p(5)]], device="cuda"
            ),
            torch.zeros((1, 1), device="cuda"),
            torch.Generator(device="cuda").manual_seed(
                SEED * 100000 + stage * 30000 + step
            ),
        )
        endpoint = denormalize_transformer_rows(
            predicted.squeeze(0),
            artifact["normalizer_means"].float(),
            artifact["normalizer_scales"].float(),
        ).cuda()
        delta = endpoint - current
        if kind == "random":
            noise = torch.randn(
                delta.shape,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(
                    SEED * 99999 + stage * 30000 + step
                ),
            )
            delta = noise * (delta.norm() / noise.norm().clamp_min(1e-12))
            endpoint = current + delta
        copy_rows_to_compatible_block(block, endpoint)
        for name, p in block.named_parameters():
            if name in IMF_WEIGHT_NAMES:
                opt.state[p].clear()
    after = evaluate(model, val)["all"]
    return dict(
        stage=stage,
        step=step,
        target_step=target_step,
        kind=kind,
        before=before,
        after=after,
        update_norm=float(delta.norm()),
        relative_update=float(delta.norm() / current.norm()),
        generator_evaluations=nfe,
        seconds=time.perf_counter() - started,
        conditioning_loss=conditioning_loss,
    )


def digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def should_intervene(kind, step):
    return (kind == "once" and step == 0) or (
        kind in ("repeat", "random") and step % 500 == 0
    )


def train_stage(
    name, stage, kind, init_state, train, val, holdout, simple, maxsteps, smoke=False
):
    directory = OUT / name / f"stage{stage}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.json").exists():
        return json.loads((directory / "summary.json").read_text())
    _, model = make_matched_models(ReasoningModelConfig(), SEED, "cuda")
    model.load_state_dict(init_state)
    initial_digest = digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    gen, artifact = (
        load_imf(ART, "cuda") if kind not in ("none", "skip") else (None, None)
    )
    rng = random.Random(SEED + stage * 1000)
    history = []
    events = []
    best = float("inf")
    beststep = None
    start = time.perf_counter()
    # Validation subsets fixed across conditions; full validation determines checkpoint selection.
    val_small = val[: min(512, len(val))]
    train_small = train[: min(512, len(train))]
    conditioning = train[: min(256, len(train))]

    def log(step, model=model):
        nonlocal best, beststep
        tr = evaluate(model, train_small)["all"]
        va = evaluate(model, val, group=True)
        loss = va["all"]["loss"]
        history.append(
            dict(
                step=step,
                train_accuracy=tr["accuracy"],
                train_loss=tr["loss"],
                val_accuracy=va["all"]["accuracy"],
                val_loss=loss,
            )
        )
        pd.DataFrame(history).to_csv(directory / "metrics.csv", index=False)
        if loss < best:
            best = loss
            beststep = step
            torch.save(model.state_dict(), directory / "selected.pt")
        status("training", name, stage, step, maxsteps)

    first = 0
    if (directory / "resume.pt").exists():
        ck = torch.load(
            directory / "resume.pt", map_location="cuda", weights_only=False
        )
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        rng.setstate(ck["rng"])
        history = ck["history"]
        events = ck["events"]
        best = ck["best"]
        beststep = ck["beststep"]
        first = ck["step"]
    else:
        log(0)
    failed = None
    completed_steps = first
    for step in range(first, maxsteps):
        trigger = should_intervene(kind, step)
        if trigger:
            events.append(
                update_imf(
                    model,
                    optimizer,
                    gen,
                    artifact,
                    stage,
                    step,
                    kind,
                    conditioning,
                    val_small,
                )
            )
            write(directory / "interventions.json", events)
        batch = sample(train, rng, 128, balanced=(stage == 1))
        # A fixed10% stage1 mixture during stage2, shared by all stage2 conditions including direct.
        if stage == 2:
            batch = batch[:115] + sample(simple["train"], rng, 13, balanced=True)
        x, pos, y = tensors(batch, "cuda")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:
            g["lr"] = lr(step, maxsteps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)[torch.arange(len(batch), device="cuda"), pos]
            loss = F.cross_entropy(logits.float(), y)
        if not torch.isfinite(loss):
            failed = f"nonfinite loss before optimizer update at step {step}"
            write(directory / "FAILED.json", dict(reason=failed, completed_steps=step))
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        done = step + 1
        completed_steps = done
        if done % 500 == 0 or done == maxsteps:
            log(done)
        if done % 1000 == 0:
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=done,
                    rng=rng.getstate(),
                    history=history,
                    events=events,
                    best=best,
                    beststep=beststep,
                ),
                directory / "resume.pt",
            )
    if failed:
        endpoint = {
            k: {"all": {"accuracy": None, "loss": None, "count": 0}}
            for k in ("validation", "holdout", "simple_holdout")
        }
    else:
        endpoint = {
            "validation": evaluate(model, val, group=True),
            "holdout": evaluate(model, holdout, group=True),
            "simple_holdout": evaluate(model, simple["holdout"], group=True),
        }
        torch.save(model.state_dict(), directory / "endpoint.pt")
    model.load_state_dict(
        torch.load(directory / "selected.pt", map_location="cuda", weights_only=True)
    )
    selected = {
        "validation": evaluate(model, val, group=True),
        "holdout": evaluate(model, holdout, group=True),
        "simple_holdout": evaluate(model, simple["holdout"], group=True),
    }
    result = dict(
        condition=name,
        stage=stage,
        schedule=kind,
        failed=failed,
        initial_digest=initial_digest,
        selected_digest=digest(model.state_dict()),
        steps=completed_steps,
        batch_size=128,
        examples_drawn=completed_steps * 128,
        stage2_simple_mixture_fraction=13 / 128 if stage == 2 else None,
        selected_step=beststep,
        selected_validation_loss=best,
        endpoint=endpoint,
        selected=selected,
        events=events,
        wall_seconds=time.perf_counter() - start,
        intervention_seconds=sum(e["seconds"] for e in events),
    )
    write(directory / "summary.json", result)
    del model, gen, optimizer
    torch.cuda.empty_cache()
    return result
