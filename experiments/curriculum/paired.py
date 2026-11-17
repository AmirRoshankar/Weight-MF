"""Reproducibility gate and paired curriculum follow-up; independent output directory."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json, time, random
import subprocess
import numpy as np
import torch
import engine as c
from modeling import (
    ReasoningModelConfig,
    compatible_block_rows,
    copy_rows_to_compatible_block,
)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
ROOT = c.BASE / "results/arithmetic_paired"
ORIGINAL = c.update_imf
CASES = (
    [(s, 0, 500) for s in (101, 102, 103, 104, 105)]
    + [(s, 0.1, 500) for s in (101, 102, 103, 104, 105)]
    + [(42, 0.05, 500), (42, 0.2, 500), (42, 0.1, 5), (42, 0.1, 30)]
)


def reliable_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def accurate_status(state, condition, stage, step, maxstep):
    n = sum(
        (
            ROOT
            / f"seed{s}_strength{int(strength * 100):03d}_interval{interval}"
            / "summary.json"
        ).exists()
        for s, strength, interval in CASES
    )
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
    d = dict(
        state=state,
        condition=condition,
        stage=stage,
        step=step,
        stage_total=maxstep,
        completed_conditions=n,
        total_conditions=len(CASES),
        gpu=gpu,
        updated_unix=time.time(),
    )
    reliable_write(ROOT / "status.json", d)
    (ROOT / "PROGRESS.txt").write_text(
        f"{state} | {n}/{len(CASES)} done | {condition} | stage{stage} {step}/{maxstep} | GPU %/MiB {gpu}\n",
        encoding="utf-8",
    )


c.write = reliable_write
c.status = accurate_status


def run(name, seed, strength, interval, steps, simple, composed):
    c.SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    c.should_intervene = lambda kind, step: kind == "repeat" and step % interval == 0

    def update(model, opt, generator, artifact, stage, step, kind, conditioning, val):
        before = compatible_block_rows(model.blocks[0]).clone()
        event = ORIGINAL(
            model, opt, generator, artifact, stage, step, kind, conditioning, val
        )
        scaled = before + strength * (compatible_block_rows(model.blocks[0]) - before)
        copy_rows_to_compatible_block(model.blocks[0], scaled)
        event["raw_after"] = event["after"]
        event["after"] = c.evaluate(model, val)["all"]
        event["strength"] = strength
        event["interval"] = interval
        return event

    c.update_imf = update
    _, model = c.make_matched_models(ReasoningModelConfig(), seed, "cuda")
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    kind = "none" if strength == 0 else "repeat"
    a = c.train_stage(
        name,
        1,
        kind,
        initial,
        simple["train"],
        simple["val"],
        simple["holdout"],
        simple,
        steps,
    )
    if a["failed"]:
        raise RuntimeError(f"{name} stage1: {a['failed']}")
    state = torch.load(
        c.OUT / name / "stage1/selected.pt", map_location="cpu", weights_only=True
    )
    b = c.train_stage(
        name,
        2,
        kind,
        state,
        composed["train"],
        composed["val"],
        composed["holdout"],
        simple,
        steps,
    )
    if b["failed"]:
        raise RuntimeError(f"{name} stage2: {b['failed']}")
    c.write(
        c.OUT / name / "summary.json",
        dict(seed=seed, strength=strength, interval=interval, stage1=a, stage2=b),
    )
    return a, b
