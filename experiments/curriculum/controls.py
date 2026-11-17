"""Matched controls for the 0.1/500 curriculum intervention."""

import time, random, sys, traceback
import numpy as np
import torch
import paired as v
import engine as c
from modeling import (
    ReasoningModelConfig,
    compatible_block_rows,
    copy_rows_to_compatible_block,
    IMF_WEIGHT_NAMES,
)

SMOKE = "--smoke" in sys.argv
ROOT = c.BASE / (
    "results/arithmetic_causal_controls_smoke_v1"
    if SMOKE
    else "results/arithmetic_causal_controls_v1"
)
CASES = [
    (seed, arm)
    for seed in ((101,) if SMOKE else (101, 102, 103, 104, 105))
    for arm in ("reset_only", "imf_reset", "imf_preserve", "random_matched")
]
STEPS = 4 if SMOKE else 20000
ORIGINAL = c.update_imf


def update(model, opt, generator, artifact, stage, step, kind, conditioning, val):
    before = compatible_block_rows(model.blocks[0]).clone()
    started = time.perf_counter()
    if kind == "reset_only":
        pre = c.evaluate(model, val)["all"]
        for name, p in model.blocks[0].named_parameters():
            if name in IMF_WEIGHT_NAMES:
                opt.state[p].clear()
        return dict(
            stage=stage,
            step=step,
            kind=kind,
            before=pre,
            after=pre,
            seconds=time.perf_counter() - started,
            update_norm=0.0,
            generator_evaluations=0,
        )
    saved = {
        name: {
            k: (x.clone() if torch.is_tensor(x) else x) for k, x in opt.state[p].items()
        }
        for name, p in model.blocks[0].named_parameters()
        if name in IMF_WEIGHT_NAMES
    }
    event = ORIGINAL(
        model, opt, generator, artifact, stage, step, "repeat", conditioning, val
    )
    proposed = compatible_block_rows(model.blocks[0])
    delta = 0.1 * (proposed - before)
    if kind == "random_matched":
        g = torch.Generator(device="cuda").manual_seed(
            c.SEED * 99999 + stage * 30000 + step
        )
        noise = torch.randn(delta.shape, device="cuda", generator=g)
        pieces = []
        for low, high in (
            (0, 128),
            (128, 256),
            (256, 384),
            (384, 512),
            (512, 1024),
            (1024, 1536),
        ):
            q = noise[low:high]
            d = delta[low:high]
            pieces.append(q * (d.norm() / q.norm().clamp_min(1e-12)))
        delta = torch.cat(pieces)
    copy_rows_to_compatible_block(model.blocks[0], before + delta)
    if kind == "imf_preserve":
        for name, p in model.blocks[0].named_parameters():
            if name in IMF_WEIGHT_NAMES:
                opt.state[p].update(saved[name])
    event.update(
        kind=kind,
        strength=0.1,
        applied_update_norm=float(delta.norm()),
        after=c.evaluate(model, val)["all"],
        seconds=time.perf_counter() - started,
    )
    return event


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    c.OUT = ROOT
    c.should_intervene = lambda kind, step: kind != "none" and step % 500 == 0
    c.update_imf = update
    c.write = v.reliable_write

    def status(state, condition, stage, step, maxstep):
        count = sum((ROOT / f"seed{s}_{a}" / "summary.json").exists() for s, a in CASES)
        v.reliable_write(
            ROOT / "status.json",
            dict(
                state=state,
                condition=condition,
                stage=stage,
                step=step,
                stage_total=maxstep,
                completed=count,
                total=len(CASES),
            ),
        )
        (ROOT / "PROGRESS.txt").write_text(
            f"{state} | {count}/{len(CASES)} | {condition} | stage {stage} {step}/{maxstep}\n"
        )

    c.status = status
    simple = c.stage1_data()
    composed = c.prepare(c.DATA)
    c.write(
        ROOT / "plan.json",
        dict(
            seeds=[101, 102, 103, 104, 105],
            arms=["reset_only", "imf_reset", "imf_preserve", "random_matched"],
            strength=0.1,
            interval=500,
            steps_per_stage=20000,
            control="random direction matches 0.1 iMF displacement per matrix on its own current state",
            reference="existing v2 optimizer and 0.1 iMF results; imf_reset rerun for fidelity",
        ),
    )
    for seed, arm in CASES:
        name = f"seed{seed}_{arm}"
        c.SEED = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        _, model = c.make_matched_models(ReasoningModelConfig(), seed, "cuda")
        initial = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        del model
        a = c.train_stage(
            name,
            1,
            arm,
            initial,
            simple["train"],
            simple["val"],
            simple["holdout"],
            simple,
            STEPS,
        )
        if a["failed"]:
            c.write(
                ROOT / name / "summary.json",
                dict(seed=seed, arm=arm, stage1=a, stage2=None),
            )
            continue
        state = torch.load(
            ROOT / name / "stage1/selected.pt", map_location="cpu", weights_only=True
        )
        b = c.train_stage(
            name,
            2,
            arm,
            state,
            composed["train"],
            composed["val"],
            composed["holdout"],
            simple,
            STEPS,
        )
        c.write(
            ROOT / name / "summary.json", dict(seed=seed, arm=arm, stage1=a, stage2=b)
        )
    status("complete", "all", 2, STEPS, STEPS)
    c.write(ROOT / "terminal.json", dict(state="complete"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ROOT.mkdir(parents=True, exist_ok=True)
        v.reliable_write(
            ROOT / "failure.json", dict(error=repr(e), traceback=traceback.format_exc())
        )
        raise
