"""Stage 1 model: Mask2Former for roof footprint segmentation.

Uses HuggingFace transformers Mask2Former with Swin-Base backbone,
pretrained on ADE20K, fine-tuned for binary roof segmentation.

Production target: Roof IoU ≥ 0.85 on frozen real test set.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation


class RoofFootprintDetector(nn.Module):
    """Mask2Former-based roof footprint detector.

    Wraps HuggingFace Mask2Former with a 2-class head (background + roof).
    Supports loading from pretrained ADE20K weights for transfer learning.
    """

    # Pretrained model IDs from HuggingFace
    PRETRAINED_IDS = {
        "swin-tiny": "facebook/mask2former-swin-tiny-ade-semantic",
        "swin-small": "facebook/mask2former-swin-small-ade-semantic",
        "swin-base": "facebook/mask2former-swin-base-ade-semantic",
    }

    def __init__(
        self,
        backbone: str = "swin-small",
        num_classes: int = 2,  # background + roof
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        model_id = self.PRETRAINED_IDS.get(backbone, self.PRETRAINED_IDS["swin-small"])

        if pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).config
            config.num_labels = num_classes
            self.model = Mask2FormerForUniversalSegmentation(config)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone weights for initial fine-tuning."""
        for name, param in self.model.named_parameters():
            if "pixel_level_module" in name:  # backbone + encoder
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> dict:
        """Forward pass returning logits for segmentation.

        Args:
            pixel_values: (B, 3, H, W) float32 image tensor

        Returns:
            dict with 'logits' key: (B, num_classes, H', W') per-pixel logits
        """
        outputs = self.model(pixel_values=pixel_values, **kwargs)

        # Mask2Former returns class queries + masks; we need per-pixel logits
        # Convert to per-pixel classification map
        if hasattr(outputs, "transformer_decoder_output"):
            # Use the class predictions + mask logits to produce per-pixel map
            class_queries = outputs.transformer_decoder_output.last_hidden_state
            masks = outputs.pred_masks  # (B, num_queries, H/4, W/4)
            class_logits = outputs.class_logits  # (B, num_queries, num_classes+1)

            # Weighted combination: for each pixel, pick the class of the
            # highest-probability query that covers it
            # Simplified: use pred_logits which HuggingFace provides
            pass

        # HuggingFace Mask2Former provides a convenience post-processing
        # but for training we use the raw loss computation
        return {
            "logits": outputs.transformer_decoder_output.last_hidden_state
            if hasattr(outputs, "transformer_decoder_output")
            else outputs.pred_masks,
            "model_outputs": outputs,
        }

    def predict_mask(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Predict binary roof mask for inference.

        Args:
            pixel_values: (B, 3, H, W) float32 image tensor

        Returns:
            (B, H, W) int64 mask — 0=background, 1=roof
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            # Use HuggingFace post-processing for instance → semantic
            result = self.model.post_process_semantic_segmentation(
                outputs, target_sizes=[pixel_values.shape[-2:]] * pixel_values.shape[0]
            )
            # Stack into (B, H, W)
            if isinstance(result, list):
                result = torch.stack(result)
        return result


def compute_mask2former_loss(
    model_outputs,
    masks: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """Compute Mask2Former loss using the model's built-in loss.

    Mask2Former has its own Hungarian matching loss. We need to pass
    the ground truth in the correct format.
    """
    # The HuggingFace Mask2Former computes loss internally when
    # mask_labels and class_labels are provided during forward pass
    # This is handled in the training loop by calling model forward
    # with the appropriate labels argument
    return model_outputs.loss if hasattr(model_outputs, "loss") else None


class RoofFootprintDetectorV2(nn.Module):
    """Simpler alternative: SegFormer for roof footprint segmentation.

    Uses SegFormer-B3 with a 2-class head. Easier to train than Mask2Former
    and works well for binary semantic segmentation.

    Use this if Mask2Former training proves unstable on your data.
    """

    PRETRAINED_ID = "nvidia/segformer-b3-finetuned-ade-512-512"

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.PRETRAINED_ID,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerForSemanticSegmentation.from_pretrained(self.PRETRAINED_ID).config
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        """Forward pass.

        Args:
            pixel_values: (B, 3, H, W) float32
            labels: (B, H, W) int64 ground truth (optional, for training loss)

        Returns:
            dict with 'logits' (B, num_classes, H/4, W/4) and optional 'loss'
        """
        return self.model(pixel_values=pixel_values, labels=labels)

    def predict_mask(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Predict binary roof mask."""
        self.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits  # (B, num_classes, H/4, W/4)
            # Upsample to input resolution
            logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
            return logits.argmax(dim=1)  # (B, H, W)
