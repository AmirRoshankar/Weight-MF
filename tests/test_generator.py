import sys, unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from weightmf.transformer_weight_meanflow import (
    ConditionalTransformerWeightGenerator,
    TransformerWeightGeneratorConfig,
)
from weightmf.meanflow_objective import improved_meanflow_loss, sample_improved_meanflow


class GeneratorTests(unittest.TestCase):
    def test_objective_backward_and_sample(self):
        torch.manual_seed(7)
        m = ConditionalTransformerWeightGenerator(
            TransformerWeightGeneratorConfig(
                width=32, latent_tokens=8, latent_depth=1, heads=4
            )
        )
        early = torch.randn(2, 1536, 128)
        late = early + 0.01 * torch.randn_like(early)
        start = torch.zeros(2, 1)
        end = torch.ones(2, 1) * 0.3
        loss, _ = improved_meanflow_loss(m, early, late, start, end, start, start)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(
            any(
                p.grad is not None and bool((p.grad != 0).any()) for p in m.parameters()
            )
        )
        out, nfe = sample_improved_meanflow(
            m, early, start, end, start, start, torch.Generator().manual_seed(9)
        )
        self.assertEqual(out.shape, early.shape)
        self.assertEqual(nfe, 1)
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    unittest.main()
