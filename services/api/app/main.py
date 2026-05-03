"""FastAPI inference service for roof corrosion detection."""

import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware import RateLimitMiddleware
from app.routes import health, quote, feedback, metrics

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("main")

# ── Startup validation ────────────────────────────────────────────────────
_REQUIRED_ENV = []  # No strictly required env vars — all have defaults
_RECOMMENDED_ENV = ["REGION", "PIPELINE"]
_OPTIONAL_SECRET_ENV = [
    "NVIDIA_API_KEY",
    "RUNPOD_API_KEY",
    "NEXT_PUBLIC_SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
]


def _validate_env() -> None:
    """Validate environment configuration at startup. Warn on missing optional secrets."""
    for var in _REQUIRED_ENV:
        if not os.getenv(var):
            logger.error("FATAL: Required env var %s is not set. Exiting.", var)
            sys.exit(1)
    for var in _RECOMMENDED_ENV:
        if not os.getenv(var):
            logger.warning("Recommended env var %s is not set — using default.", var)
    for var in _OPTIONAL_SECRET_ENV:
        if not os.getenv(var):
            logger.info(
                "Optional secret %s not set — some features may be degraded.", var
            )
    logger.info(
        "Environment OK: REGION=%s PIPELINE=%s",
        os.getenv("REGION", "TH"),
        os.getenv("PIPELINE", "fm"),
    )


# ── Graceful shutdown ─────────────────────────────────────────────────────
def _shutdown_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown in containers."""
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — shutting down gracefully...", sig_name)
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)


# ── Lifespan (startup/shutdown hooks) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _validate_env()
    logger.info("Roof Corrosion API starting — REGION=%s PIPELINE=%s",
                os.getenv("REGION", "TH"), os.getenv("PIPELINE", "fm"))
    yield
    # Shutdown
    logger.info("Roof Corrosion API shutting down")


_allowed_origins_raw = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000",
)
_allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]

app = FastAPI(
    title="Roof Corrosion AI API",
    version="0.1.0",
    description="Satellite-based roof corrosion detection and quoting service",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(quote.router, prefix="/quote", tags=["quote"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(metrics.router, tags=["monitoring"])
