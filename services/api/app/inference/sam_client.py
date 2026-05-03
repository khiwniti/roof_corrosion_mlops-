"""SAM 2 (Segment Anything Model v2) client for zero-shot roof segmentation.

Multi-backend client — supports:
1. RunPod Serverless SAM endpoint (self-hosted, set SAM_RUNPOD_ENDPOINT_ID)
2. Replicate API (hosted, set REPLICATE_API_TOKEN)
3. Local segment-geospatial (requires torch + SAM weights, no GPU needed for ViT-B)

SAM 2 is used here as a fallback when Microsoft/OSM building footprints
aren't available or are misaligned with the actual roof.

Prompting strategy:
- Given a lat/lng → project to pixel (cx, cy) in the tile using tile metadata
- Pass a single positive point prompt to SAM → get segmentation mask
- Return the mask as a numpy bool array aligned with the input image

Usage:
    client = SAMClient()
    mask = await client.segment(tile_image, point_xy=(512, 512))
    # → (H, W) bool array of the roof
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SAMClient:
    """Multi-backend SAM 2 client.

    Auto-detects which backend to use based on environment variables.
    Priority: RunPod > Replicate > Local (if segment-geospatial installed).
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        runpod_endpoint_id: Optional[str] = None,
        runpod_api_key: Optional[str] = None,
        replicate_token: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.runpod_endpoint_id = runpod_endpoint_id or os.environ.get("SAM_RUNPOD_ENDPOINT_ID", "")
        self.runpod_api_key = runpod_api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.replicate_token = replicate_token or os.environ.get("REPLICATE_API_TOKEN", "")
        self.timeout = timeout

        # Select backend
        if backend:
            self.backend = backend
        elif self.runpod_endpoint_id and self.runpod_api_key:
            self.backend = "runpod"
        elif self.replicate_token:
            self.backend = "replicate"
        else:
            self.backend = "local"

        logger.info(f"SAM client using backend: {self.backend}")

    async def segment(
        self,
        tile_image: np.ndarray,
        point_xy: Optional[tuple[int, int]] = None,
        box_xyxy: Optional[tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Segment the object indicated by a point or box prompt.

        Args:
            tile_image: (H, W, 3) uint8 RGB numpy array
            point_xy: (x, y) pixel coordinate of a positive prompt point
            box_xyxy: (x1, y1, x2, y2) bounding box prompt

        Returns:
            (H, W) bool numpy array — True = inside segmented object
        """
        if point_xy is None and box_xyxy is None:
            # Default: center point
            h, w = tile_image.shape[:2]
            point_xy = (w // 2, h // 2)

        if self.backend == "runpod":
            return await self._segment_runpod(tile_image, point_xy, box_xyxy)
        elif self.backend == "replicate":
            return await self._segment_replicate(tile_image, point_xy, box_xyxy)
        else:
            return self._segment_local(tile_image, point_xy, box_xyxy)

    # ── RunPod backend ───────────────────────────────────────

    async def _segment_runpod(
        self,
        tile_image: np.ndarray,
        point_xy: Optional[tuple[int, int]],
        box_xyxy: Optional[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Call a self-hosted SAM 2 worker on RunPod Serverless."""
        url = f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}/runsync"
        image_b64 = self._image_to_b64(tile_image)

        payload: dict = {"input": {"image_base64": image_b64}}
        if point_xy:
            payload["input"]["point"] = [point_xy[0], point_xy[1]]
        if box_xyxy:
            payload["input"]["box"] = list(box_xyxy)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.runpod_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        output = data.get("output", {})
        if "error" in output:
            raise SAMError(f"RunPod SAM worker failed: {output['error']}")

        mask_b64 = output.get("mask_b64")
        if not mask_b64:
            raise SAMError(f"No mask in SAM response: {output}")

        return self._b64_to_mask(mask_b64, tile_image.shape[:2])

    # ── Replicate backend ────────────────────────────────────

    async def _segment_replicate(
        self,
        tile_image: np.ndarray,
        point_xy: Optional[tuple[int, int]],
        box_xyxy: Optional[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Call Meta's SAM 2 via Replicate API.

        Replicate model: meta/sam-2
        See: https://replicate.com/meta/sam-2
        """
        image_b64 = self._image_to_b64(tile_image)
        url = "https://api.replicate.com/v1/predictions"

        input_data: dict = {"image": f"data:image/png;base64,{image_b64}"}
        if point_xy:
            input_data["point_coords"] = f"[[{point_xy[0]}, {point_xy[1]}]]"
            input_data["point_labels"] = "[1]"
        if box_xyxy:
            input_data["box"] = list(box_xyxy)

        # Model version for meta/sam-2 (pin for reproducibility)
        model_version = os.environ.get(
            "REPLICATE_SAM_VERSION",
            "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Token {self.replicate_token}",
                    "Content-Type": "application/json",
                },
                json={"version": model_version, "input": input_data},
            )
            resp.raise_for_status()
            prediction = resp.json()

            # Poll for completion
            get_url = prediction["urls"]["get"]
            import asyncio
            for _ in range(60):
                await asyncio.sleep(1)
                poll = await client.get(
                    get_url,
                    headers={"Authorization": f"Token {self.replicate_token}"},
                )
                poll.raise_for_status()
                pred = poll.json()
                if pred["status"] == "succeeded":
                    # Output is typically a URL to the mask PNG
                    mask_url = pred["output"]
                    if isinstance(mask_url, list):
                        mask_url = mask_url[0]
                    mask_resp = await client.get(mask_url)
                    mask_img = Image.open(io.BytesIO(mask_resp.content)).convert("L")
                    mask_arr = np.array(mask_img)
                    return mask_arr > 128
                elif pred["status"] == "failed":
                    raise SAMError(f"Replicate prediction failed: {pred.get('error')}")

            raise SAMError("Replicate prediction timed out")

    # ── Local backend ────────────────────────────────────────

    def _segment_local(
        self,
        tile_image: np.ndarray,
        point_xy: Optional[tuple[int, int]],
        box_xyxy: Optional[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Run SAM locally via segment-geospatial / sam2 package.

        Requires: pip install segment-geospatial
        """
        try:
            # Prefer SAM 2 if available
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch

            checkpoint = os.environ.get("SAM2_CHECKPOINT", "sam2_hiera_tiny.pt")
            model_cfg = os.environ.get("SAM2_CONFIG", "sam2_hiera_t.yaml")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            sam2_model = build_sam2(model_cfg, checkpoint, device=device)
            predictor = SAM2ImagePredictor(sam2_model)

            predictor.set_image(tile_image)
            kwargs: dict = {}
            if point_xy:
                kwargs["point_coords"] = np.array([[point_xy[0], point_xy[1]]])
                kwargs["point_labels"] = np.array([1])
            if box_xyxy:
                kwargs["box"] = np.array(box_xyxy)

            masks, scores, _ = predictor.predict(**kwargs, multimask_output=False)
            return masks[0].astype(bool)
        except ImportError:
            logger.warning("SAM 2 not installed locally; returning centered bbox as roof")
            return self._fallback_bbox_mask(tile_image.shape[:2], point_xy, box_xyxy)

    @staticmethod
    def _fallback_bbox_mask(
        shape: tuple[int, int],
        point_xy: Optional[tuple[int, int]],
        box_xyxy: Optional[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Cheap fallback: return a centered or box-based rectangle mask."""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        if box_xyxy:
            x1, y1, x2, y2 = box_xyxy
            mask[y1:y2, x1:x2] = True
        else:
            # Default: middle 50% of the image
            mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
        return mask

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _image_to_b64(tile_image: np.ndarray) -> str:
        pil = Image.fromarray(tile_image)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _b64_to_mask(mask_b64: str, shape: tuple[int, int]) -> np.ndarray:
        mask_bytes = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_arr = np.array(mask_img)
        if mask_arr.shape != shape:
            mask_img = mask_img.resize((shape[1], shape[0]), Image.NEAREST)
            mask_arr = np.array(mask_img)
        return mask_arr > 128


class SAMError(Exception):
    """Base error for SAM client operations."""
    pass
