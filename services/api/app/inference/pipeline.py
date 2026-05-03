"""Two-stage inference pipeline: roof footprint → corrosion segmentation.

Production implementation that loads models from MLflow registry
or HuggingFace Hub, with TorchScript/TensorRT optimization.
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from app.inference.types import CorrosionResult, classify_severity
from app.routes.metrics import MODEL_INFERENCE_TIME, MODEL_CONFIDENCE


class CorrosionPipeline:
    """Two-stage roof corrosion analysis pipeline.

    Loads models from MLflow registry (production stage) or HuggingFace
    Hub (dev fallback). Supports TorchScript-optimized inference.

    Usage:
        pipeline = CorrosionPipeline.from_production()
        result = pipeline.analyze(tile_image, gsd=0.3)
    """

    def __init__(
        self,
        roof_model: Optional[torch.nn.Module] = None,
        corrosion_model: Optional[torch.nn.Module] = None,
        roof_model_uri: str = "stub",
        corrosion_model_uri: str = "stub",
        device: str = "auto",
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.roof_model = roof_model
        self.corrosion_model = corrosion_model
        self.roof_model_uri = roof_model_uri
        self.corrosion_model_uri = corrosion_model_uri

    @classmethod
    def from_production(cls, device: str = "auto") -> "CorrosionPipeline":
        """Load production models from MLflow registry."""
        roof_model, roof_uri = cls._load_model("roof_detector", "production", device)
        corrosion_model, corrosion_uri = cls._load_model("corrosion_detector", "production", device)
        return cls(
            roof_model=roof_model,
            corrosion_model=corrosion_model,
            roof_model_uri=roof_uri,
            corrosion_model_uri=corrosion_uri,
            device=device,
        )

    @classmethod
    def from_huggingface(cls, device: str = "auto") -> "CorrosionPipeline":
        """Load dev models directly from HuggingFace Hub (no MLflow needed)."""
        from ml.baseline.models.roof_detector import RoofFootprintDetectorV2
        from ml.baseline.models.corrosion_detector import CorrosionDetector

        roof = RoofFootprintDetectorV2(num_classes=2, pretrained=True)
        corrosion = CorrosionDetector(backbone="b2", num_classes=3, pretrained=True)

        return cls(
            roof_model=roof,
            corrosion_model=corrosion,
            roof_model_uri="huggingface/segformer-b3-ade",
            corrosion_model_uri="huggingface/segformer-b2-ade",
            device=device,
        )

    @classmethod
    def from_torchscript(cls, roof_path: str, corrosion_path: str, device: str = "auto") -> "CorrosionPipeline":
        """Load TorchScript-optimized models from disk."""
        dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        roof_model = torch.jit.load(roof_path, map_location=dev)
        corrosion_model = torch.jit.load(corrosion_path, map_location=dev)
        return cls(
            roof_model=roof_model,
            corrosion_model=corrosion_model,
            roof_model_uri=f"torchscript://{roof_path}",
            corrosion_model_uri=f"torchscript://{corrosion_path}",
            device=dev,
        )

    @classmethod
    def from_runpod_serverless(cls) -> "RunPodCorrosionPipeline":
        """Create a pipeline that delegates inference to RunPod Serverless GPU.

        No local GPU needed — all model inference runs on RunPod.
        Returns a RunPodCorrosionPipeline (subclass) with async analyze().
        """
        return RunPodCorrosionPipeline()

    @staticmethod
    def _load_model(model_name: str, stage: str, device: str) -> tuple:
        """Load a model from MLflow registry with fallback to HuggingFace."""
        import mlflow

        dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        uri = f"models:/{model_name}/{stage}"

        try:
            model = mlflow.pytorch.load_model(uri, map_location=dev)
            model = model.to(dev)
            model.eval()
            print(f"Loaded {model_name} from MLflow: {uri}")
            return model, uri
        except Exception as e:
            print(f"MLflow load failed for {uri}: {e}")
            print(f"Falling back to HuggingFace pretrained model")

            # Fallback to HuggingFace
            if model_name == "roof_detector":
                from ml.baseline.models.roof_detector import RoofFootprintDetectorV2
                model = RoofFootprintDetectorV2(num_classes=2, pretrained=True)
            elif model_name == "corrosion_detector":
                from ml.baseline.models.corrosion_detector import CorrosionDetector
                model = CorrosionDetector(backbone="b2", num_classes=3, pretrained=True)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            model = model.to(dev)
            model.eval()
            return model, f"huggingface/fallback"

    def analyze(self, tile_image: np.ndarray, gsd: float = 0.3) -> CorrosionResult:
        """Run two-stage analysis on a satellite tile.

        Args:
            tile_image: (H, W, 3) uint8 RGB image
            gsd: ground sample distance in meters per pixel
        """
        # Preprocess
        img_tensor = self._preprocess(tile_image)

        # Stage 1: Roof footprint
        t0 = time.time()
        roof_mask = self._predict_roof(img_tensor, tile_image.shape[:2])
        MODEL_INFERENCE_TIME.labels(stage="roof").observe(time.time() - t0)

        # Stage 2: Corrosion on roof crop (with 10m buffer)
        buffer_pixels = int(10.0 / gsd)
        t0 = time.time()
        corrosion_mask = self._predict_corrosion(img_tensor, roof_mask, buffer_pixels, tile_image.shape[:2])
        MODEL_INFERENCE_TIME.labels(stage="corrosion").observe(time.time() - t0)

        # Compute areas
        roof_pixels = roof_mask.sum()
        corrosion_pixels = (corrosion_mask & roof_mask).sum()
        roof_area_m2 = roof_pixels * gsd * gsd
        corroded_area_m2 = corrosion_pixels * gsd * gsd
        corrosion_percent = (corroded_area_m2 / roof_area_m2 * 100) if roof_area_m2 > 0 else 0.0

        severity = classify_severity(corrosion_percent)

        # Confidence from model (or MC-dropout if available)
        confidence = self._compute_confidence(img_tensor, corrosion_mask, roof_mask, tile_image.shape[:2])
        MODEL_CONFIDENCE.observe(confidence)

        return CorrosionResult(
            roof_area_m2=roof_area_m2,
            corroded_area_m2=corroded_area_m2,
            corrosion_percent=corrosion_percent,
            severity=severity,
            confidence=confidence,
            roof_mask=roof_mask,
            corrosion_mask=corrosion_mask,
            gsd=gsd,
            roof_model_version=self.roof_model_uri,
            corrosion_model_version=self.corrosion_model_uri,
        )

    def _preprocess(self, tile_image: np.ndarray) -> torch.Tensor:
        """Convert (H, W, 3) uint8 to (1, 3, H, W) normalized tensor."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        from PIL import Image
        pil = Image.fromarray(tile_image)
        tensor = transform(pil).unsqueeze(0).to(self.device)
        return tensor

    def _predict_roof(self, img_tensor: torch.Tensor, original_shape: tuple) -> np.ndarray:
        """Stage 1: Predict roof footprint mask."""
        if self.roof_model is None:
            return np.zeros(original_shape, dtype=bool)

        with torch.no_grad():
            # Handle different model types
            if hasattr(self.roof_model, 'predict_mask'):
                # SegFormer wrapper
                mask = self.roof_model.predict_mask(img_tensor)  # (B, H, W)
            elif isinstance(self.roof_model, torch.jit.ScriptModule):
                # TorchScript model
                output = self.roof_model(img_tensor)
                if output.shape[-2:] != torch.Size(original_shape):
                    output = F.interpolate(output.unsqueeze(0) if output.dim() == 3 else output,
                                           size=original_shape, mode="bilinear", align_corners=False)
                mask = output.argmax(dim=1) if output.dim() == 4 else output
            else:
                # Raw model output
                output = self.roof_model(pixel_values=img_tensor)
                if hasattr(output, 'logits'):
                    logits = output.logits
                    logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
                    mask = logits.argmax(dim=1)
                else:
                    mask = torch.zeros(1, *original_shape, dtype=torch.long, device=self.device)

        return (mask.cpu().numpy()[0] == 1)  # binary roof mask

    def _predict_corrosion(
        self,
        img_tensor: torch.Tensor,
        roof_mask: np.ndarray,
        buffer_px: int,
        original_shape: tuple,
    ) -> np.ndarray:
        """Stage 2: Predict corrosion mask within roof crop + buffer."""
        if self.corrosion_model is None:
            return np.zeros(original_shape, dtype=bool)

        with torch.no_grad():
            if hasattr(self.corrosion_model, 'predict_mask'):
                mask = self.corrosion_model.predict_mask(img_tensor)  # (B, H, W)
            elif isinstance(self.corrosion_model, torch.jit.ScriptModule):
                output = self.corrosion_model(img_tensor)
                if output.shape[-2:] != torch.Size(original_shape):
                    output = F.interpolate(output.unsqueeze(0) if output.dim() == 3 else output,
                                           size=original_shape, mode="bilinear", align_corners=False)
                mask = output.argmax(dim=1) if output.dim() == 4 else output
            else:
                output = self.corrosion_model(pixel_values=img_tensor)
                if hasattr(output, 'logits'):
                    logits = output.logits
                    logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
                    mask = logits.argmax(dim=1)
                else:
                    mask = torch.zeros(1, *original_shape, dtype=torch.long, device=self.device)

        corrosion_pred = (mask.cpu().numpy()[0] == 2)  # class 2 = corrosion

        # Apply roof mask + buffer (only count corrosion within roof area)
        from scipy.ndimage import binary_dilation
        roof_with_buffer = binary_dilation(roof_mask, iterations=buffer_px)
        return corrosion_pred & roof_with_buffer

    def _compute_confidence(
        self,
        img_tensor: torch.Tensor,
        corrosion_mask: np.ndarray,
        roof_mask: np.ndarray,
        original_shape: tuple,
    ) -> float:
        """Compute model confidence for the prediction.

        Uses max class probability averaged over the roof area.
        Falls back to 0.5 if model doesn't support probability output.
        """
        if self.corrosion_model is None:
            return 0.5

        try:
            with torch.no_grad():
                if hasattr(self.corrosion_model, 'model'):
                    # SegFormer wrapper — get raw logits
                    outputs = self.corrosion_model(pixel_values=img_tensor)
                    logits = outputs.logits
                    logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (3, H, W)

                    # Average max probability over roof pixels
                    if roof_mask.sum() > 0:
                        roof_probs = probs[:, roof_mask]  # (3, N_roof)
                        max_probs = roof_probs.max(axis=0)  # (N_roof,)
                        return float(max_probs.mean())
                elif hasattr(self.corrosion_model, 'predict_corrosion_probability'):
                    prob_map = self.corrosion_model.predict_corrosion_probability(img_tensor)
                    prob_np = prob_map.cpu().numpy()[0]
                    if roof_mask.sum() > 0:
                        return float(prob_np[roof_mask].mean())
        except Exception:
            pass

        return 0.5


class RunPodCorrosionPipeline:
    """Pipeline that delegates all GPU inference to RunPod Serverless.

    No local GPU or model files needed. The FastAPI orchestrator just
    calls the RunPod endpoint and returns the result.

    Usage:
        pipeline = CorrosionPipeline.from_runpod_serverless()
        result = await pipeline.analyze_async(tile_image, gsd=0.3)
    """

    def __init__(self):
        from app.inference.runpod_client import RunPodClient
        self._client = RunPodClient()
        self.roof_model_uri = "runpod/serverless"
        self.corrosion_model_uri = "runpod/serverless"

    async def analyze_async(
        self,
        tile_image: np.ndarray,
        gsd: float = 0.3,
        address: str = "",
    ) -> CorrosionResult:
        """Run two-stage analysis via RunPod Serverless GPU.

        Args:
            tile_image: (H, W, 3) uint8 RGB image
            gsd: ground sample distance in meters/pixel
            address: optional address for quote context

        Returns:
            CorrosionResult with all metrics
        """
        result = await self._client.analyze(
            image_array=tile_image,
            gsd=gsd,
            return_masks=True,
        )

        if "error" in result:
            raise RuntimeError(f"RunPod inference failed: {result['error']}")

        # Decode masks from base64
        import base64
        import io as _io
        from PIL import Image as _Image

        roof_mask = np.zeros(tile_image.shape[:2], dtype=bool)
        corrosion_mask = np.zeros(tile_image.shape[:2], dtype=bool)

        if result.get("roof_mask_b64"):
            roof_png = base64.b64decode(result["roof_mask_b64"])
            roof_arr = np.array(_Image.open(_io.BytesIO(roof_png)))
            roof_mask = roof_arr > 128

        if result.get("corrosion_mask_b64"):
            corr_png = base64.b64decode(result["corrosion_mask_b64"])
            corr_arr = np.array(_Image.open(_io.BytesIO(corr_png)))
            corrosion_mask = corr_arr > 128

        return CorrosionResult(
            roof_area_m2=result.get("roof_area_m2", 0.0),
            corroded_area_m2=result.get("corroded_area_m2", 0.0),
            corrosion_percent=result.get("corrosion_percent", 0.0),
            severity=result.get("severity", "none"),
            confidence=result.get("confidence", 0.5),
            roof_mask=roof_mask,
            corrosion_mask=corrosion_mask,
            gsd=gsd,
            roof_model_version=result.get("model_versions", {}).get("roof", "unknown"),
            corrosion_model_version=result.get("model_versions", {}).get("corrosion", "unknown"),
        )

    def analyze(self, tile_image: np.ndarray, gsd: float = 0.3) -> CorrosionResult:
        """Synchronous wrapper for analyze_async."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.analyze_async(tile_image, gsd)).result()
        else:
            return asyncio.run(self.analyze_async(tile_image, gsd))
