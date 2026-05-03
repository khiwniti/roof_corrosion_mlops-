"""NVIDIA NIM client for vision-language model corrosion assessment.

Uses NIM's OpenAI-compatible API to call a large VLM (Llama 3.2 90B Vision)
for zero-shot corrosion severity assessment on satellite roof tiles.

No training, no labeled data, no GPU required on our side.
Hosted endpoint: https://integrate.api.nvidia.com/v1
Self-hosted NIM container: http://<your-runpod-ip>:8000/v1

Usage:
    client = NIMVisionClient()
    result = await client.assess_corrosion(tile_image, roof_bbox)
    # → {"corrosion_percent": 18.5, "severity": "moderate", "confidence": 0.82,
    #    "description": "Visible rust patches concentrated on eastern panels",
    #    "rationale": "..."}
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Any, Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════

CORROSION_SYSTEM_PROMPT = """You are an expert roof-inspection AI. You analyze high-resolution overhead satellite or aerial imagery of rooftops and identify visible metal corrosion (rust, oxidation, pitting, coating failure).

You are precise, conservative, and always return valid JSON. You never invent corrosion that isn't clearly visible. If the image is not a roof, or is too low quality to judge, report low confidence and `"assessable": false`.

Rules:
- Corrosion indicators: orange/brown rust patches, reddish-brown discoloration, uneven coating, streaking rust stains, pitting texture
- NOT corrosion: shadows, moss/lichen (green), dirt (uniform gray/brown), tile roofs (terracotta is normal), asphalt roofs (uniformly dark)
- Only report corrosion on **metal roofs** (corrugated steel, standing seam, tin)
- If roof material is non-metal (tile, asphalt, concrete), set `corrosion_percent: 0` and note material in description
- Severity bands:
    none:     < 5% of roof area
    light:    5-25%
    moderate: 25-50%
    severe:   > 50%
- Confidence reflects how sure you are — low confidence → requires human review

Always respond with ONLY a JSON object, no prose. Use this exact schema:

{
  "assessable": bool,
  "roof_material": "metal_corrugated" | "metal_standing_seam" | "tile" | "asphalt" | "concrete" | "unknown",
  "corrosion_percent": float,
  "severity": "none" | "light" | "moderate" | "severe",
  "confidence": float,
  "description": string,
  "rationale": string,
  "visible_issues": [string, ...]
}
"""


def build_user_prompt(gsd: float, address: str = "", roof_context: str = "") -> str:
    """Build the user turn for the VLM."""
    # Inject active-region hint (Thailand by default) into roof_context if none given
    if not roof_context:
        try:
            from app.region import get_active_region
            region = get_active_region()
            if region.roof_context_hint:
                roof_context = region.roof_context_hint
        except Exception:
            pass

    parts = [
        "Analyze this overhead image of a rooftop for metal corrosion.",
        f"Ground sample distance: approximately {gsd*100:.0f}cm per pixel.",
    ]
    if address:
        parts.append(f"Property address (for regional context): {address}")
    if roof_context:
        parts.append(f"Regional / roof context: {roof_context}")
    parts.append("")
    parts.append("Return your assessment as a single JSON object per the schema.")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════

class NIMVisionClient:
    """Client for NVIDIA NIM vision-language model API.

    Defaults to the hosted API at integrate.api.nvidia.com (OpenAI-compatible).
    Can point to a self-hosted NIM container by setting NIM_BASE_URL.
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = (base_url or os.environ.get("NIM_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("NIM_MODEL") or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key and "integrate.api.nvidia.com" in self.base_url:
            raise ValueError(
                "NVIDIA_API_KEY not set. Get one free at https://build.nvidia.com → "
                "and add to .env.local, or set NIM_BASE_URL to a self-hosted endpoint."
            )

    @property
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # ── Public API ───────────────────────────────────────────

    async def assess_corrosion(
        self,
        tile_image: np.ndarray | Image.Image,
        gsd: float = 0.3,
        address: str = "",
        roof_context: str = "",
        max_image_size: int = 1024,
    ) -> dict:
        """Assess roof corrosion from an overhead image.

        Args:
            tile_image: (H, W, 3) uint8 numpy array or PIL Image
            gsd: ground sample distance in meters/pixel
            address: optional address for regional context
            roof_context: optional context like "corrugated metal shed"
            max_image_size: resize longest side to at most this many pixels
                            (VLM APIs charge per image tile)

        Returns:
            dict with keys: assessable, roof_material, corrosion_percent,
            severity, confidence, description, rationale, visible_issues
        """
        # Normalize image input
        if isinstance(tile_image, np.ndarray):
            pil = Image.fromarray(tile_image)
        else:
            pil = tile_image.convert("RGB") if tile_image.mode != "RGB" else tile_image

        # Resize to bound cost — VLMs charge per image tile
        w, h = pil.size
        if max(w, h) > max_image_size:
            scale = max_image_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil = pil.resize((new_w, new_h), Image.LANCZOS)

        image_data_uri = self._to_data_uri(pil)

        # Build messages
        user_prompt = build_user_prompt(gsd=gsd, address=address, roof_context=roof_context)
        messages = [
            {"role": "system", "content": CORROSION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # low for deterministic assessment
            "max_tokens": 1024,
            "top_p": 0.95,
        }

        # Call with retries
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response_text = await self._chat_completion(payload)
                result = self._parse_json(response_text)
                return self._normalize_result(result)
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(f"NIM request failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                # JSON parse error or schema error — don't retry, they'll fail again
                last_error = e
                logger.error(f"NIM response parsing failed: {e}")
                break

        raise NIMError(f"NIM assessment failed after {self.max_retries+1} attempts: {last_error}")

    # ── Internal ─────────────────────────────────────────────

    async def _chat_completion(self, payload: dict) -> str:
        """Call the OpenAI-compatible /chat/completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Extract content from OpenAI-compatible response
        choices = data.get("choices", [])
        if not choices:
            raise NIMError(f"No choices in NIM response: {data}")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise NIMError(f"Empty content in NIM response: {data}")
        return content

    @staticmethod
    def _to_data_uri(pil: Image.Image) -> str:
        """Convert PIL image to base64 data URI."""
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from LLM response, tolerating markdown fences."""
        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Find content between ```[json]? and ```
            m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
            else:
                text = text.strip("`").strip()

        # Try to extract JSON object even if there's leading/trailing prose
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise NIMError(f"Failed to parse JSON from NIM response: {e}\nRaw: {text[:500]}")

    @staticmethod
    def _normalize_result(raw: dict) -> dict:
        """Normalize and validate the VLM response against our schema."""
        # Defaults for missing fields
        result = {
            "assessable": bool(raw.get("assessable", True)),
            "roof_material": str(raw.get("roof_material", "unknown")),
            "corrosion_percent": float(raw.get("corrosion_percent", 0.0)),
            "severity": str(raw.get("severity", "none")).lower(),
            "confidence": float(raw.get("confidence", 0.5)),
            "description": str(raw.get("description", "")),
            "rationale": str(raw.get("rationale", "")),
            "visible_issues": list(raw.get("visible_issues", [])),
        }

        # Clamp ranges
        result["corrosion_percent"] = max(0.0, min(100.0, result["corrosion_percent"]))
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        # Validate severity
        valid_severity = {"none", "light", "moderate", "severe"}
        if result["severity"] not in valid_severity:
            # Derive from percent
            pct = result["corrosion_percent"]
            if pct < 5:
                result["severity"] = "none"
            elif pct < 25:
                result["severity"] = "light"
            elif pct < 50:
                result["severity"] = "moderate"
            else:
                result["severity"] = "severe"

        return result


class NIMError(Exception):
    """Base error for NIM client operations."""
    pass
