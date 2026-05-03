"""RunPod Serverless client for offloading GPU inference to the cloud.

The local FastAPI orchestrator calls this client to run the two-stage
roof corrosion pipeline on a RunPod Serverless GPU endpoint, avoiding
the need for a local GPU.

Usage:
    client = RunPodClient()
    result = await client.analyze(image_url="https://...", gsd=0.3)

Or with a base64-encoded image:
    result = await client.analyze(image_base64="data:image/png;base64,...", gsd=0.3)
"""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Optional

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


class RunPodClient:
    """Async client for RunPod Serverless inference endpoint.

    Handles:
    - Synchronous requests (/runsync) for low-latency inference
    - Asynchronous requests (/run) for batch processing
    - Automatic retry on cold starts
    - Timeout management
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not set. Add to .env.local")
        if not self.endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID not set. Deploy inference endpoint first.")

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def _endpoint_url(self) -> str:
        return f"{RUNPOD_API_BASE}/{self.endpoint_id}"

    async def analyze(
        self,
        image_url: str = "",
        image_base64: str = "",
        image_array: Optional[np.ndarray] = None,
        gsd: float = 0.3,
        return_masks: bool = False,
    ) -> dict:
        """Run two-stage corrosion analysis on RunPod GPU.

        Args:
            image_url: publicly accessible URL to satellite tile
            image_base64: base64-encoded image (data URI or raw)
            image_array: (H, W, 3) uint8 numpy array — auto-converted to base64
            gsd: ground sample distance in meters/pixel
            return_masks: include base64-encoded masks in response

        Returns:
            dict with keys: roof_area_m2, corroded_area_m2, corrosion_percent,
            severity, confidence, model_versions, inference_time_ms
        """
        # Convert numpy array to base64 if needed
        if image_array is not None and not image_base64:
            pil = Image.fromarray(image_array)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            image_base64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        if not image_url and not image_base64:
            raise ValueError("Must provide image_url, image_base64, or image_array")

        payload = {
            "input": {
                "gsd": gsd,
                "return_masks": return_masks,
            }
        }
        if image_url:
            payload["input"]["image_url"] = image_url
        if image_base64:
            payload["input"]["image_base64"] = image_base64

        # Try runsync first (waits for result, up to timeout)
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._runsync(payload)
                return result
            except RunPodTimeoutError:
                if attempt < self.max_retries:
                    logger.warning(f"RunPod timeout (attempt {attempt+1}), retrying...")
                    await asyncio.sleep(2)
                else:
                    raise
            except RunPodColdStartError as e:
                # Cold start: submit async and poll
                logger.info(f"Cold start detected, submitting async job: {e}")
                return await self._run_async(payload)

    async def _runsync(self, payload: dict) -> dict:
        """Send synchronous request to RunPod endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self._endpoint_url}/runsync",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status", "UNKNOWN")
        if status == "COMPLETED":
            return data.get("output", {})
        elif status == "FAILED":
            error = data.get("error", data.get("output", {}).get("error", "Unknown error"))
            raise RunPodInferenceError(f"Inference failed: {error}")
        elif status == "IN_QUEUE" or status == "IN_PROGRESS":
            # Job still running — poll for result
            job_id = data.get("id")
            if job_id:
                return await self._poll_result(job_id)
            raise RunPodTimeoutError("Job submitted but no ID returned")
        else:
            raise RunPodInferenceError(f"Unexpected status: {status}")

    async def _run_async(self, payload: dict) -> dict:
        """Submit async job and poll for result."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self._endpoint_url}/run",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        job_id = data.get("id")
        if not job_id:
            raise RunPodInferenceError(f"No job ID in response: {data}")

        return await self._poll_result(job_id)

    async def _poll_result(self, job_id: str, poll_interval: float = 5.0) -> dict:
        """Poll for async job result."""
        start = time.time()
        while time.time() - start < self.timeout:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{self._endpoint_url}/status/{job_id}",
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()

            status = data.get("status", "UNKNOWN")
            if status == "COMPLETED":
                return data.get("output", {})
            elif status == "FAILED":
                error = data.get("error", data.get("output", {}).get("error", "Unknown"))
                raise RunPodInferenceError(f"Job {job_id} failed: {error}")
            elif status in ("IN_QUEUE", "IN_PROGRESS"):
                await asyncio.sleep(poll_interval)
            else:
                raise RunPodInferenceError(f"Unexpected status for job {job_id}: {status}")

        raise RunPodTimeoutError(f"Job {job_id} timed out after {self.timeout}s")

    async def health(self) -> dict:
        """Check endpoint health."""
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(
                    f"{self._endpoint_url}/health",
                    headers=self._headers,
                )
                return resp.json()
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}


class RunPodError(Exception):
    """Base error for RunPod operations."""
    pass


class RunPodTimeoutError(RunPodError):
    """Request timed out."""
    pass


class RunPodColdStartError(RunPodError):
    """Cold start detected — workers need to spin up."""
    pass


class RunPodInferenceError(RunPodError):
    """Inference failed on the remote worker."""
    pass
