"""FastAPI inference service for roof corrosion detection."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware import RateLimitMiddleware
from app.routes import health, quote, feedback, metrics

app = FastAPI(
    title="Roof Corrosion AI API",
    version="0.1.0",
    description="Satellite-based roof corrosion detection and quoting service",
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Vercel dev proxy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(quote.router, prefix="/quote", tags=["quote"])
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(metrics.router, tags=["monitoring"])
