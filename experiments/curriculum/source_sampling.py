"""Five paired target seeds using the newly collected source generator."""

import traceback
import paired as v
import engine as c

ROOT = c.BASE / "results/arithmetic_uniform_source_transfer_v1"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    v.ROOT = ROOT
    c.OUT = ROOT
    c.ART = c.BASE / "artifacts/stratified/best_model.pt"
    v.CASES = [(s, 0.1, 500) for s in (101, 102, 103, 104, 105)]
    c.write = v.reliable_write
    c.status = v.accurate_status
    simple = c.stage1_data()
    composed = c.prepare(c.DATA)
    c.write(
        ROOT / "plan.json",
        dict(
            seeds=[101, 102, 103, 104, 105],
            strength=0.1,
            interval=500,
            stages=[1, 2],
            steps_per_stage=20000,
            source="uniform recollection",
            comparison="original transition focused iMF and optimizer on same target seeds from arithmetic_paired",
            note="validation and holdout source runs remain disjoint, both generators trained for 10000 updates",
        ),
    )
    for seed, strength, interval in v.CASES:
        v.run(
            f"seed{seed}_strength010_interval500",
            seed,
            strength,
            interval,
            20000,
            simple,
            composed,
        )
    v.accurate_status("complete", "all conditions", 2, 20000, 20000)
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
