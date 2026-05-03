"""Customer feedback endpoints — wired to Supabase + relabeling queue."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.db import get_supabase
from app.queue import enqueue_job, RELABEL_QUEUE

router = APIRouter()


class FeedbackRequest(BaseModel):
    job_id: str = Field(..., description="Quote job ID this feedback refers to")
    customer_id: str | None = Field(None, description="Authenticated customer ID")
    correct: bool = Field(..., description="Was the corrosion assessment correct?")
    notes: str | None = Field(None, description="Free-text feedback")
    roof_boundary_wrong: bool = Field(False, description="Roof boundary was incorrect")
    corrosion_area_wrong: bool = Field(False, description="Corrosion area was incorrect")
    severity_wrong: bool = Field(False, description="Severity grade was incorrect")


@router.post("/")
async def submit_feedback(req: FeedbackRequest) -> dict:
    """Submit feedback on a quote result.

    If the assessment was incorrect, the prediction is flagged for relabeling
    and enqueued into the active learning loop.
    """
    flagged = not req.correct

    # Store in Supabase
    try:
        supabase = get_supabase()
        supabase.table("feedback").insert({
            "job_id": req.job_id,
            "customer_id": req.customer_id or "00000000-0000-0000-0000-000000000000",
            "correct": req.correct,
            "notes": req.notes,
            "roof_boundary_wrong": req.roof_boundary_wrong,
            "corrosion_area_wrong": req.corrosion_area_wrong,
            "severity_wrong": req.severity_wrong,
            "flagged_for_relabeling": flagged,
        }).execute()
    except Exception:
        pass

    # Enqueue for relabeling if incorrect
    if flagged:
        enqueue_job(RELABEL_QUEUE, {
            "job_id": req.job_id,
            "reason": "customer_feedback",
            "issues": {
                "roof_boundary_wrong": req.roof_boundary_wrong,
                "corrosion_area_wrong": req.corrosion_area_wrong,
                "severity_wrong": req.severity_wrong,
            },
            "notes": req.notes,
        })

    return {
        "status": "received",
        "job_id": req.job_id,
        "flagged_for_relabeling": flagged,
    }
