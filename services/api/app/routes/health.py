"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    # TODO: check Redis, model loading, tile API connectivity
    return {"status": "ready"}
