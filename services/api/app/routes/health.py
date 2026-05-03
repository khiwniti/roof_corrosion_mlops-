"""Health and readiness check endpoints."""

import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.queue import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check():
    """Readiness probe: verifies Redis connectivity and env configuration."""
    checks: dict[str, str] = {}
    healthy = True

    # Redis check
    try:
        r = get_redis()
        if r is not None:
            r.ping()
            checks["redis"] = "ok"
        else:
            checks["redis"] = "unavailable"
    except Exception as exc:
        logger.warning("Redis readiness check failed: %s", exc)
        checks["redis"] = f"error: {exc}"
        healthy = False

    # Required env vars
    missing = [
        var for var in ("NEXT_PUBLIC_SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
        if not os.getenv(var)
    ]
    if missing:
        checks["config"] = f"missing env vars: {', '.join(missing)}"
    else:
        checks["config"] = "ok"

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if healthy else "degraded", "checks": checks},
    )
