import json, unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_source_split_is_disjoint(self):
        d = json.loads((ROOT / "configs/transition/split_manifest.json").read_text())
        parts = [set(d[k]) for k in ["train_seeds", "validation_seeds", "test_seeds"]]
        self.assertEqual([len(x) for x in parts], [500, 51, 51])
        self.assertEqual(len(set.union(*parts)), 602)

    def test_curriculum_headline(self):
        means = {}
        for strength in ("000", "010"):
            values = []
            for seed in range(101, 106):
                d = json.loads(
                    (
                        ROOT
                        / f"results/curriculum/seed{seed}_strength{strength}_interval500/summary.json"
                    ).read_text()
                )
                values.append(d["stage2"]["endpoint"]["holdout"]["all"]["accuracy"])
            means[strength] = sum(values) / 5
        self.assertEqual(round(100 * means["000"], 2), 57.18)
        self.assertEqual(round(100 * means["010"], 2), 75.17)

    def test_scaling_censors_failures(self):
        base = pd.read_csv(
            ROOT / "results/source_scaling/hybrid_0050_runs/trial_summary.csv"
        )
        base = base[base.protocol == "optimizer_only"]
        self.assertEqual(len(base), 20)
        self.assertEqual(int(base.grok_step.isna().sum()), 1)
        self.assertEqual(float(base.grok_step.fillna(20000).mean()), 7255.0)
        flow = pd.read_csv(
            ROOT / "results/source_scaling/hybrid_0500_runs/trial_summary.csv"
        )
        flow = flow[flow.protocol == "raw_optimizer10_flow90_fixed_9000"]
        self.assertEqual(float(flow.grok_step.fillna(20000).mean()), 3690.0)


if __name__ == "__main__":
    unittest.main()
