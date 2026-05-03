"""Relabeling worker: processes feedback queue → Label Studio.

When customers flag incorrect predictions, this worker:
1. Dequeues feedback items from the relabel Redis queue
2. Fetches the original tile + prediction from S3
3. Creates a Label Studio pre-annotation task
4. Exports to Label Studio for human correction
5. On completion, adds corrected labels to the training dataset

Usage:
    python -m app.inference.relabel_worker [--poll-interval 30]
"""

import argparse
import json
import time
from datetime import datetime
from typing import Optional

from app.db import get_supabase
from app.queue import RELABEL_QUEUE, dequeue_job, get_redis


LABEL_STUDIO_PROJECT_ID = 1  # Roof corrosion segmentation project


def create_label_studio_task(
    tile_id: str,
    image_s3_key: str,
    prediction_s3_key: Optional[str] = None,
    feedback_notes: Optional[str] = None,
    issues: Optional[dict] = None,
) -> dict:
    """Create a Label Studio pre-annotation task for relabeling.

    The task includes the original model prediction as a pre-annotation
    so the human annotator only needs to correct the errors, not label from scratch.
    """
    import os

    label_studio_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    label_studio_key = os.getenv("LABEL_STUDIO_API_KEY", "")

    task = {
        "data": {
            "tile_id": tile_id,
            "image": f"/data/local-files/?d=tiles/{tile_id}.tif",
            "feedback_notes": feedback_notes or "",
            "issues": issues or {},
            "relabeling_reason": "customer_feedback",
            "created_at": datetime.utcnow().isoformat(),
        },
        "predictions": [],  # TODO: add model prediction as pre-annotation
    }

    # If we have the prediction mask, add as pre-annotation
    if prediction_s3_key:
        task["predictions"] = [{
            "result": [{
                "type": "brushlabels",
                "value": {
                    "format": "rle",
                    "rle": [],  # TODO: convert mask to RLE
                },
                "from_name": "corrosion_mask",
                "to_name": "image",
                "origin": "manual",
                "score": 0.5,
            }],
            "score": 0.5,
            "model_version": "production",
        }]

    return task


def push_to_label_studio(tasks: list[dict]) -> dict:
    """Push relabeling tasks to Label Studio via API."""
    import os
    import httpx

    label_studio_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    label_studio_key = os.getenv("LABEL_STUDIO_API_KEY", "")

    if not label_studio_key:
        print("⚠️  LABEL_STUDIO_API_KEY not set. Skipping Label Studio push.")
        return {"status": "skipped", "reason": "no_api_key"}

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{label_studio_url}/api/projects/{LABEL_STUDIO_PROJECT_ID}/import",
                headers={"Authorization": f"Token {label_studio_key}"},
                json=tasks,
            )
            response.raise_for_status()
            return {"status": "ok", "tasks_created": len(tasks)}
    except Exception as e:
        print(f"⚠️  Label Studio push failed: {e}")
        return {"status": "failed", "error": str(e)}


def process_relabel_item(item: dict) -> dict:
    """Process a single relabeling item from the queue."""
    job_id = item.get("job_id", "")
    reason = item.get("reason", "customer_feedback")
    issues = item.get("issues", {})
    notes = item.get("notes", "")

    print(f"  Processing relabel for job {job_id}: {reason}")

    # Look up the job in Supabase to get tile info
    tile_id = job_id  # fallback
    image_s3_key = None
    prediction_s3_key = None

    try:
        supabase = get_supabase()
        job = supabase.table("jobs").select("id").eq("id", job_id).single().execute()
        if job.data:
            assessment = supabase.table("assessments").select("*").eq("job_id", job_id).single().execute()
            if assessment.data:
                image_s3_key = assessment.data.get("roof_mask_s3_key")
                prediction_s3_key = assessment.data.get("corrosion_mask_s3_key")
                tile_id = job_id
    except Exception as e:
        print(f"  ⚠️  Supabase lookup failed: {e}")

    # Create Label Studio task
    task = create_label_studio_task(
        tile_id=tile_id,
        image_s3_key=image_s3_key or "",
        prediction_s3_key=prediction_s3_key,
        feedback_notes=notes,
        issues=issues,
    )

    return task


def run_relabel_worker(poll_interval: int = 30):
    """Main relabeling worker loop."""
    print("=" * 60)
    print("Roof Corrosion AI — Relabeling Worker")
    print("=" * 60)
    print(f"Polling {RELABEL_QUEUE} every {poll_interval}s...")
    print("Press Ctrl+C to stop.\n")

    items_processed = 0
    pending_tasks = []

    while True:
        try:
            item = dequeue_job(RELABEL_QUEUE, timeout=poll_interval)
            if item:
                task = process_relabel_item(item)
                pending_tasks.append(task)
                items_processed += 1

                # Batch push to Label Studio every 10 items
                if len(pending_tasks) >= 10:
                    result = push_to_label_studio(pending_tasks)
                    print(f"  Pushed {len(pending_tasks)} tasks to Label Studio: {result['status']}")
                    pending_tasks = []

            else:
                # No item — flush pending tasks if any
                if pending_tasks:
                    result = push_to_label_studio(pending_tasks)
                    print(f"  Flushed {len(pending_tasks)} tasks to Label Studio: {result['status']}")
                    pending_tasks = []

        except KeyboardInterrupt:
            # Final flush
            if pending_tasks:
                push_to_label_studio(pending_tasks)
            print(f"\nRelabel worker stopped. {items_processed} items processed.")
            break
        except Exception as e:
            print(f"⚠️  Relabel worker error: {e}")
            time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roof Corrosion AI Relabeling Worker")
    parser.add_argument("--poll-interval", type=int, default=30)
    args = parser.parse_args()
    run_relabel_worker(poll_interval=args.poll_interval)
