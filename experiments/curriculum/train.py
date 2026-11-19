from pathlib import Path
import torch
from modeling import (
    ReasoningModelConfig,
    PaperReasoningTransformer,
    IMFCompatiblePaperTransformer,
    copy_paper_to_bridge,
)
from weightmf.transformer_weight_meanflow import (
    ConditionalTransformerWeightGenerator,
    TransformerWeightGeneratorConfig,
    canonical_tensor_and_row_ids,
)


def load_imf(
    path: Path, device: str
) -> tuple[ConditionalTransformerWeightGenerator, dict]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = ConditionalTransformerWeightGenerator(
        TransformerWeightGeneratorConfig(**artifact["model_config"])
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact


def _normalize(rows: torch.Tensor, artifact: dict) -> torch.Tensor:
    tensor_ids, _ = canonical_tensor_and_row_ids()
    ids = tensor_ids.to(rows.device)
    means = artifact["normalizer_means"].to(rows.device).float()
    scales = artifact["normalizer_scales"].to(rows.device).float()
    return (rows - means[ids, None]) / scales[ids, None]


def make_matched_models(
    config: ReasoningModelConfig,
    seed: int,
    device: str,
) -> tuple[PaperReasoningTransformer, IMFCompatiblePaperTransformer]:
    torch.manual_seed(seed)
    paper = PaperReasoningTransformer(config)
    bridge = IMFCompatiblePaperTransformer(config)
    copy_paper_to_bridge(paper, bridge)
    return paper.to(device), bridge.to(device)
