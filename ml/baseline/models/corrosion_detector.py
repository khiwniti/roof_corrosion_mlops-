"""Stage 2 model: SegFormer-B2 for corrosion segmentation.

Fine-tuned on Caribbean irregular_metal class as corrosion proxy.
Uses Focal + Dice loss to handle severe class imbalance (corrosion is rare).

Production target: Corrosion IoU ≥ 0.55 on frozen real test set.
Go/no-go gate: IoU ≥ 0.45 on open-data baseline before investing in paid data.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class CorrosionDetector(nn.Module):
    """SegFormer-B2 for corrosion segmentation on roof crops.

    3 classes: 0=background, 1=roof (healthy), 2=corrosion
    Trained with Focal + Dice loss for class imbalance.
    """

    PRETRAINED_IDS = {
        "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }

    def __init__(
        self,
        backbone: str = "b2",
        num_classes: int = 3,  # background + roof + corrosion
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        model_id = self.PRETRAINED_IDS.get(backbone, self.PRETRAINED_IDS["b2"])

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerForSemanticSegmentation.from_pretrained(model_id).config
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze encoder for initial fine-tuning of decode head only."""
        for name, param in self.model.named_parameters():
            if "decode_head" not in name:
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Args:
            pixel_values: (B, 3, H, W) float32 normalized images
            labels: (B, H, W) int64 ground truth masks (optional)

        Returns:
            dict with 'logits' (B, num_classes, H/4, W/4) and optional 'loss'
        """
        return self.model(pixel_values=pixel_values, labels=labels)

    def predict_mask(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Predict 3-class mask for inference.

        Returns:
            (B, H, W) int64 — 0=background, 1=roof, 2=corrosion
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits  # (B, num_classes, H/4, W/4)
            logits = F.interpolate(
                logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
            )
            return logits.argmax(dim=1)

    def predict_corrosion_probability(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Predict per-pixel corrosion probability for uncertainty estimation.

        Returns:
            (B, H, W) float32 — probability of corrosion class
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            logits = F.interpolate(
                logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
            )
            probs = F.softmax(logits, dim=1)
            return probs[:, 2]  # corrosion class probability


class CorrosionDetectorUNet(nn.Module):
    """UNet++ with EfficientNet-B4 encoder for corrosion segmentation.

    Alternative to SegFormer when transformer attention isn't needed
    and you want a more proven architecture for fine texture.

    Requires: segmentation_models_pytorch
    pip install segmentation-models-pytorch
    """

    def __init__(self, encoder_name: str = "efficientnet-b4", num_classes: int = 3):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp

            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                classes=num_classes,
            )
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch required for UNet++ variant. "
                "Install with: pip install segmentation-models-pytorch"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning per-pixel logits.

        Args:
            x: (B, 3, H, W) float32

        Returns:
            (B, num_classes, H, W) logits
        """
        return self.model(x)

    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Predict 3-class mask."""
        self.eval()
        with torch.no_grad():
            logits = self.model(x)
            return logits.argmax(dim=1)


def mc_dropout_uncertainty(
    model: nn.Module,
    pixel_values: torch.Tensor,
    num_samples: int = 10,
    dropout_rate: float = 0.1,
) -> dict:
    """Monte Carlo dropout uncertainty estimation.

    Enables dropout at inference time and runs multiple forward passes
    to estimate prediction uncertainty. Used for:
    1. Confidence gating on quotes
    2. Active learning sample selection

    Args:
        model: CorrosionDetector (or any SegFormer-based model)
        pixel_values: (B, 3, H, W) input
        num_samples: number of MC forward passes
        dropout_rate: dropout probability

    Returns:
        dict with 'mean' (B, H, W), 'std' (B, H, W), 'entropy' (B, H, W)
    """
    # Enable dropout in eval mode
    model.train()  # enables dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            logits = F.interpolate(
                logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
            )
            probs = F.softmax(logits, dim=1)[:, 2]  # corrosion probability
            predictions.append(probs)

    predictions = torch.stack(predictions)  # (num_samples, B, H, W)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    # Entropy-based uncertainty
    entropy = -(mean * torch.log(mean + 1e-8) + (1 - mean) * torch.log(1 - mean + 1e-8))

    model.eval()
    return {"mean": mean, "std": std, "entropy": entropy}
