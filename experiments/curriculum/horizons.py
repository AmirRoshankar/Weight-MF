"""Paired horizon ablation for the stratified-source generator."""

import traceback
import paired as v
import engine as c

ROOT = c.BASE / "results/arithmetic_stratified_horizon_v1"
base_update = v.ORIGINAL
sample = c.sample_improved_meanflow
CASES = [(s, h) for h in (100, 200) for s in (101, 102, 103, 104, 105)]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    c.OUT = ROOT
    c.ART = c.BASE / "artifacts/stratified/best_model.pt"
    c.write = v.reliable_write

    def status(state, condition, stage, step, maxstep):
        n = sum(
            (ROOT / f"seed{s}_horizon{h}" / "summary.json").exists() for s, h in CASES
        )
        v.reliable_write(
            ROOT / "status.json",
            dict(
                state=state,
                condition=condition,
                stage=stage,
                step=step,
                stage_total=maxstep,
                completed=n,
                total=len(CASES),
            ),
        )
        (ROOT / "PROGRESS.txt").write_text(
            f"{state} | {n}/{len(CASES)} | {condition} | stage {stage} {step}/{maxstep}\n"
        )

    c.status = status
    c.write(
        ROOT / "plan.json",
        dict(
            seeds=[101, 102, 103, 104, 105],
            horizons=[100, 200],
            strength=0.1,
            interval=500,
            stages=[1, 2],
            steps_per_stage=20000,
            source="stratified generator",
            references="same seeds: transition generator and stratified generator under min(30000,max(9000,step+1000)); optimizer baseline",
            comparisons="fixed data, paired initializations and minibatch seeds; endpoint prompts changed",
        ),
    )
    simple = c.stage1_data()
    composed = c.prepare(c.DATA)
    for seed, h in CASES:

        def horizon_update(
            model, opt, generator, artifact, stage, step, kind, conditioning, val, h=h
        ):
            def override(model_arg, early, start, end, *args, **kwargs):
                return sample(
                    model_arg, early, start, start + h / 30000.0, *args, **kwargs
                )

            c.sample_improved_meanflow = override
            try:
                event = base_update(
                    model,
                    opt,
                    generator,
                    artifact,
                    stage,
                    step,
                    kind,
                    conditioning,
                    val,
                )
                event["target_step"] = step + h
                event["horizon"] = h
                return event
            finally:
                c.sample_improved_meanflow = sample

        v.ORIGINAL = horizon_update
        v.run(f"seed{seed}_horizon{h}", seed, 0.1, 500, 20000, simple, composed)
    status("complete", "all", 2, 20000, 20000)
    c.write(ROOT / "terminal.json", dict(state="complete", conditions=len(CASES)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ROOT.mkdir(parents=True, exist_ok=True)
        v.reliable_write(
            ROOT / "terminal.json",
            dict(state="failed", error=repr(e), traceback=traceback.format_exc()),
        )
        raise
