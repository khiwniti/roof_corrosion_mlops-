"""Customer feedback endpoints for model improvement loop."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class FeedbackRequest(BaseModel):
    job_id: str = Field(..., description="Quote job ID this feedback refers to")
    correct: bool = Field(..., description="Was the corrosion assessment correct?")
    notes: str | None = Field(None, description="Free-text feedback")


@router.post("/")
async def submit_feedback(req: FeedbackRequest) -> dict[str, str]:
    """Submit feedback on a quote result. Flagged predictions go to relabeling queue."""
    # TODO: store feedback in Postgres, if !correct → enqueue for relabeling
    return {"status": "received", "job_id": req.job_id}
