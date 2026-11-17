import random
import numpy as np
import torch
from torch.nn import functional as F
from .model import GrokTransformer


@torch.inference_mode()
def evaluate(
    model: GrokTransformer, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, float]:
    logits = model(x)
    return float(F.cross_entropy(logits, y)), float(
        (logits.argmax(dim=-1) == y).float().mean()
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
