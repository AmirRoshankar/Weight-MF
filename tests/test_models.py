import sys, unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/curriculum")]
from train import make_matched_models
from modeling import (
    ReasoningModelConfig,
    compatible_block_rows,
    copy_rows_to_compatible_block,
)
from grokflow.config import GrokConfig
from grokflow.operation_data import make_binary_operation


class ModelTests(unittest.TestCase):
    def test_bridge_matches_original(self):
        paper, bridge = make_matched_models(
            ReasoningModelConfig(block_size=32), 101, "cpu"
        )
        x = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        with torch.no_grad():
            torch.testing.assert_close(paper(x), bridge(x), atol=1e-5, rtol=1e-5)

    def test_checkpoint_roundtrip(self):
        _, m = make_matched_models(ReasoningModelConfig(block_size=32), 101, "cpu")
        rows = compatible_block_rows(m.blocks[0]).clone()
        self.assertEqual(tuple(rows.shape), (1536, 128))
        copy_rows_to_compatible_block(m.blocks[0], rows)
        torch.testing.assert_close(
            compatible_block_rows(m.blocks[0]), rows, atol=0, rtol=0
        )

    def test_modular_division(self):
        d = make_binary_operation(GrokConfig(modulus=97), 101, "division")
        x = torch.cat([d.train_x, d.val_x])
        y = torch.cat([d.train_y, d.val_y])
        self.assertTrue(bool((x[:, 1] != 0).all()))
        self.assertTrue(torch.equal((y * x[:, 1]) % 97, x[:, 0]))
        self.assertEqual(len(x), 97 * 96)


if __name__ == "__main__":
    unittest.main()
