"""Redis job queue for async inference processing.

Falls back to in-memory queue when Redis is unavailable (for local dev / tests).
"""

import json
import os
from typing import Optional

import redis


_redis_client: Optional[redis.Redis] = None
_in_memory_queues: dict[str, list[str]] = {}
_in_memory_status: dict[str, str] = {}
_redis_available: Optional[bool] = None


def get_redis() -> Optional[redis.Redis]:
    """Get or create Redis client. Returns None if Redis is unavailable."""
    global _redis_client, _redis_available

    if _redis_available is False:
        return None

    if _redis_client is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            _redis_client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
            _redis_client.ping()
            _redis_available = True
        except (redis.ConnectionError, redis.TimeoutError):
            _redis_client = None
            _redis_available = False

    return _redis_client


# Queue names
QUOTE_QUEUE = "quote_jobs"
FEEDBACK_QUEUE = "feedback_jobs"
RELABEL_QUEUE = "relabel_jobs"


def enqueue_job(queue_name: str, job_data: dict) -> str:
    """Push a job to a queue. Falls back to in-memory if Redis unavailable."""
    job_id = job_data.get("job_id", job_data.get("id", str(hash(json.dumps(job_data)))))
    payload = json.dumps(job_data)

    r = get_redis()
    if r:
        r.rpush(queue_name, payload)
    else:
        _in_memory_queues.setdefault(queue_name, []).append(payload)

    return job_id


def dequeue_job(queue_name: str, timeout: int = 5) -> Optional[dict]:
    """Pop a job from a queue (blocking). Falls back to in-memory."""
    r = get_redis()
    if r:
        result = r.blpop(queue_name, timeout=timeout)
        if result:
            _, data = result
            return json.loads(data)
        return None
    else:
        queue = _in_memory_queues.get(queue_name, [])
        if queue:
            return json.loads(queue.pop(0))
        return None


def get_queue_length(queue_name: str) -> int:
    """Get number of jobs in a queue."""
    r = get_redis()
    if r:
        return r.llen(queue_name)
    return len(_in_memory_queues.get(queue_name, []))


def get_job_status(job_id: str) -> Optional[dict]:
    """Get job status from Redis hash or in-memory."""
    r = get_redis()
    if r:
        data = r.hget("job_status", job_id)
        if data:
            return json.loads(data)
    else:
        if job_id in _in_memory_status:
            return json.loads(_in_memory_status[job_id])
    return None


def set_job_status(job_id: str, status: dict) -> None:
    """Update job status in Redis hash or in-memory."""
    r = get_redis()
    if r:
        r.hset("job_status", job_id, json.dumps(status))
    else:
        _in_memory_status[job_id] = json.dumps(status)
