"""Model export: PyTorch → TorchScript for production serving.

Exports trained models from MLflow registry to TorchScript (.pt) files
that can be loaded without Python/transformers dependency, enabling:
1. Faster inference (no dynamic dispatch)
2. Smaller deployment footprint
3. Triton Inference Server compatibility
4. RunPod serverless cold-start reduction

Usage:
    python ml/training/export_model.py \
        --model-name corrosion_detector \
        --stage production \
        --output-dir models/exported/
"""

import argparse
import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F


def export_segformer_to_torchscript(
    model: torch.nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 512, 512),
    device: str = "cpu",
) -> dict:
    """Export a SegFormer model to TorchScript.

    SegFormer models from HuggingFace need a wrapper to make them
    TorchScript-compatible (the HuggingFace forward() has complex typing).
    """
    model = model.to(device).eval()

    # Create wrapper that TorchScript can trace
    class SegFormerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits  # (B, C, H/4, W/4)
            # Upsample to input resolution
            logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return logits

    wrapper = SegFormerWrapper(model)
    wrapper = wrapper.to(device).eval()

    # Trace with example input
    example_input = torch.randn(*input_size, device=device)

    try:
        traced = torch.jit.trace(wrapper, example_input)
        traced.save(output_path)
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)

        # Verify the exported model
        loaded = torch.jit.load(output_path, map_location=device)
        with torch.no_grad():
            output = loaded(example_input)
            assert output.shape == (input_size[0], model.config.num_labels if hasattr(model, 'config') else 3, input_size[2], input_size[3])

        return {
            "status": "ok",
            "path": output_path,
            "size_mb": size_mb,
            "input_size": input_size,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


def export_mask2former_to_torchscript(
    model: torch.nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 512, 512),
    device: str = "cpu",
) -> dict:
    """Export Mask2Former to TorchScript (simplified semantic output)."""
    model = model.to(device).eval()

    class Mask2FormerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Simplified: return semantic segmentation logits
            outputs = self.model(pixel_values=pixel_values)
            # Use class logits + mask logits to produce per-pixel classification
            class_logits = outputs.class_logits  # (B, Q, C+1)
            mask_logits = outputs.pred_masks  # (B, Q, H/4, W/4)

            # Weighted combination: softmax over queries, then weighted sum
            query_weights = F.softmax(class_logits[:, :, :-1], dim=1)  # (B, Q, C)
            mask_up = F.interpolate(mask_logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
            # (B, Q, H, W) @ (B, Q, C) -> (B, C, H, W)
            semantic = torch.einsum("bqhw,bqc->bchw", mask_up, query_weights)
            return semantic

    wrapper = Mask2FormerWrapper(model)
    example_input = torch.randn(*input_size, device=device)

    try:
        traced = torch.jit.trace(wrapper, example_input)
        traced.save(output_path)
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        return {"status": "ok", "path": output_path, "size_mb": size_mb}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def export_model(
    model_name: str,
    stage: str = "production",
    output_dir: str = "models/exported",
    device: str = "cpu",
) -> dict:
    """Export a model from MLflow registry to TorchScript."""
    output_path = Path(output_dir) / f"{model_name}_{stage}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name}/{stage} from MLflow...")
    try:
        model = mlflow.pytorch.load_model(f"models:/{model_name}/{stage}", map_location=device)
    except Exception as e:
        print(f"Failed to load from MLflow: {e}")
        print("Falling back to HuggingFace pretrained...")
        if model_name == "roof_detector":
            from ml.baseline.models.roof_detector import RoofFootprintDetectorV2
            model = RoofFootprintDetectorV2(num_classes=2, pretrained=True).model
        elif model_name == "corrosion_detector":
            from ml.baseline.models.corrosion_detector import CorrosionDetector
            model = CorrosionDetector(backbone="b2", num_classes=3, pretrained=True).model
        else:
            raise ValueError(f"Unknown model: {model_name}")

    print(f"Exporting to TorchScript: {output_path}")
    result = export_segformer_to_torchscript(model, str(output_path), device=device)

    if result["status"] == "ok":
        print(f"✅ Exported: {result['size_mb']:.1f} MB → {output_path}")
    else:
        print(f"❌ Export failed: {result['error']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Export ML models to TorchScript")
    parser.add_argument("--model-name", type=str, required=True, choices=["roof_detector", "corrosion_detector"])
    parser.add_argument("--stage", type=str, default="production")
    parser.add_argument("--output-dir", type=str, default="models/exported")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    result = export_model(args.model_name, args.stage, args.output_dir, args.device)
    print(json.dumps(result, indent=2))


import json  # noqa: E402

if __name__ == "__main__":
    main()
