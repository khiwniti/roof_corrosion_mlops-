"""Prometheus metrics endpoint for the FastAPI service."""

from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()

# ── Metrics definitions ─────────────────────────────────────

# Job metrics
JOBS_SUBMITTED = Counter(
    "roof_jobs_submitted_total",
    "Total number of quote jobs submitted",
    ["source"],
)
JOBS_COMPLETED = Counter(
    "roof_jobs_completed_total",
    "Total number of jobs completed",
    ["status"],  # completed, failed, requires_review
)
JOBS_PROCESSING_TIME = Histogram(
    "roof_job_processing_seconds",
    "Time to process a single quote job",
    buckets=[10, 30, 60, 90, 120, 180, 300, 600],
)

# Model metrics
MODEL_INFERENCE_TIME = Histogram(
    "roof_model_inference_seconds",
    "Model inference time per stage",
    ["stage"],  # roof, corrosion
)
MODEL_CONFIDENCE = Histogram(
    "roof_model_confidence",
    "Model confidence score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Queue metrics
QUEUE_DEPTH = Gauge(
    "roof_queue_depth",
    "Current depth of job queues",
    ["queue_name"],  # quote_jobs, feedback_jobs, relabel_jobs
)

# Business metrics
QUOTES_GENERATED = Counter(
    "roof_quotes_generated_total",
    "Total quotes generated",
    ["severity"],
)
QUOTES_REQUIRING_REVIEW = Counter(
    "roof_quotes_requires_review_total",
    "Quotes that required human review",
)
FEEDBACK_RECEIVED = Counter(
    "roof_feedback_received_total",
    "Customer feedback received",
    ["correct"],
)

# Tile API metrics
TILE_FETCH_TIME = Histogram(
    "roof_tile_fetch_seconds",
    "Time to fetch satellite tiles",
    ["source"],  # maxar, nearmap
    buckets=[1, 2, 5, 10, 30],
)
TILE_CACHE_HITS = Counter(
    "roof_tile_cache_hits_total",
    "S3 tile cache hits",
    ["source"],
)
TILE_CACHE_MISSES = Counter(
    "roof_tile_cache_misses_total",
    "S3 tile cache misses",
    ["source"],
)


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Update queue depth gauge
    try:
        from app.queue import get_queue_length, QUOTE_QUEUE, FEEDBACK_QUEUE, RELABEL_QUEUE
        QUEUE_DEPTH.labels(queue_name="quote").set(get_queue_length(QUOTE_QUEUE))
        QUEUE_DEPTH.labels(queue_name="feedback").set(get_queue_length(FEEDBACK_QUEUE))
        QUEUE_DEPTH.labels(queue_name="relabel").set(get_queue_length(RELABEL_QUEUE))
    except Exception:
        pass

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
