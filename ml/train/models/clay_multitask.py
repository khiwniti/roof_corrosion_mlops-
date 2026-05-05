"""Clay v1.5 + ViT-Adapter + Mask2Former multi-task model stub.

Architecture Decision: ADR-002
- Backbone: Clay v1.5 (sensor-agnostic, 10-band S2 + 4-band VHR streams)
- Adapter: ViT-Adapter (ICLR 2023) for dense prediction
- Decoder: Mask2Former with 3 query groups:
    * material head (5 classes: metal, tile, concrete, vegetation, other)
    * corrosion head (binary: rust / no-rust)
    * severity head (ordinal: 0=none, 1=light, 2=moderate, 3=severe)

This is a Phase 2 stub. Real implementation will use TerraTorch ≥1.2.4
for Clay integration and MMSegmentation / Detectron2 for Mask2Former.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("train.models.clay_multitask")


class ClayMultiTaskModel(nn.Module):
    """Stub multi-task model for roof material + corrosion + severity.

    In production this wraps:
      1. Clay v1.5 encoder (from HuggingFace ` clay-m-aws/...`)
      2. ViT-Adapter dense feature injection
      3. Mask2Former pixel decoder + transformer decoder
      4. Three segmentation heads
    """

    NUM_MATERIAL_CLASSES = 5
    NUM_CORROSION_CLASSES = 2
    NUM_SEVERITY_LEVELS = 4

    def __init__(
        self,
        backbone: str = "clay_v1.5",
        hidden_dim: int = 256,
        num_queries: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # ── Stub encoder (replace with Clay.from_pretrained(...)) ──
        # Clay v1.5 outputs patch embeddings at multiple scales.
        # For the stub we use a simple ConvNet placeholder.
        self.stub_encoder = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),   # S2 10-band input
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # ── Stub pixel decoder (replaces Mask2Former pixel decoder) ──
        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
        )

        # ── Query embeddings (replaces Mask2Former transformer decoder) ──
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_decoder = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=1024, batch_first=True
        )

        # ── Segmentation heads ──
        self.material_head = nn.Conv2d(hidden_dim, self.NUM_MATERIAL_CLASSES, kernel_size=1)
        self.corrosion_head = nn.Conv2d(hidden_dim, self.NUM_CORROSION_CLASSES, kernel_size=1)
        self.severity_head = nn.Conv2d(hidden_dim, self.NUM_SEVERITY_LEVELS, kernel_size=1)

        logger.info(
            "Initialized ClayMultiTaskModel (stub) — backbone=%s, hidden=%s, queries=%s",
            backbone, hidden_dim, num_queries,
        )

    def forward(
        self,
        x: torch.Tensor,
        vhr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, 10, H, W) Sentinel-2 10-band stack
        vhr : optional (B, 4, H, W) VHR 4-band (Pléiades/RGB+NIR) for Tier-1

        Returns
        -------
        dict with logits for each head:
            material_logits: (B, 5, H, W)
            corrosion_logits: (B, 2, H, W)
            severity_logits: (B, 4, H, W)
        """
        b, c, h, w = x.shape

        # Encoder stub
        features = self.stub_encoder(x)  # (B, hidden_dim, H/4, W/4)

        # Pixel decoder (upsample back to input resolution)
        pix_feat = self.pixel_decoder(features)  # (B, hidden_dim, H, W)

        # Query-based decoding (simplified — real Mask2Former uses cross-attn)
        # Flatten spatial dims for transformer interaction
        flat = pix_feat.view(b, self.hidden_dim, -1).permute(0, 2, 1)  # (B, HW, C)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B, Q, C)
        decoded = self.query_decoder(queries, flat)  # (B, Q, C)

        # Map queries back to spatial for per-pixel logits
        # Simplified: weighted combination of queries per pixel
        attn = torch.bmm(flat, decoded.permute(0, 2, 1))  # (B, HW, Q)
        attn = F.softmax(attn, dim=-1)
        weighted = torch.bmm(attn, decoded)  # (B, HW, C)
        weighted = weighted.permute(0, 2, 1).view(b, self.hidden_dim, h, w)

        # Combine with pixel features
        combined = pix_feat + weighted

        material_logits = self.material_head(combined)
        corrosion_logits = self.corrosion_head(combined)
        severity_logits = self.severity_head(combined)

        return {
            "material_logits": material_logits,
            "corrosion_logits": corrosion_logits,
            "severity_logits": severity_logits,
        }

    def load_clay_weights(self, checkpoint_path: str) -> None:
        """Load pretrained Clay v1.5 weights from HuggingFace or local file.

        Stub: logs the call. Real implementation uses:
            from terratorch.models import ClayModel
            clay = ClayModel.from_pretrained("clay-m-aws/clay-v1.5")
        """
        logger.info("Loading Clay weights from %s (stub — no-op)", checkpoint_path)


def build_model(config: dict[str, Any]) -> ClayMultiTaskModel:
    """Factory function from config dict."""
    model_cfg = config.get("model", {})
    return ClayMultiTaskModel(
        backbone=model_cfg.get("backbone", "clay_v1.5"),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_queries=model_cfg.get("num_queries", 100),
    )
