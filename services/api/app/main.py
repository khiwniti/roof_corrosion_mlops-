"""FastAPI inference service for roof corrosion detection."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware import RateLimitMiddleware
from app.routes import health, quote, feedback, metrics

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_allowed_origins_raw = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000",
)
_allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]

app = FastAPI(
    title="Roof Corrosion AI API",
    version="0.1.0",
    description="Satellite-based roof corrosion detection and quoting service",
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
