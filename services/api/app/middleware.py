"""Rate limiting middleware for FastAPI.

Uses Redis-backed sliding window rate limiting.
Falls back to in-memory when Redis is unavailable.
"""

import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter.

    Limits:
    - Anonymous: 10 requests/minute
    - Authenticated: 60 requests/minute
    - Ops role: 200 requests/minute
    """

    # Rate limits (requests per minute)
    LIMITS = {
        "anonymous": 10,
        "authenticated": 60,
        "ops": 200,
    }

    def __init__(self, app, redis_url: Optional[str] = None):
        super().__init__(app)
        self._redis = None
        self._in_memory: dict[str, list[float]] = defaultdict(list)
        self._redis_url = redis_url or os.getenv("REDIS_URL")

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health/metrics endpoints
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)

        # Determine client identity and tier
        client_id = self._get_client_id(request)
        tier = self._get_tier(request)
        limit = self.LIMITS[tier]

        # Check rate limit
        key = f"ratelimit:{client_id}"
        now = time.time()
        window = 60.0  # 1 minute sliding window

        allowed = self._check_rate(key, now, window, limit)

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {limit} requests/minute for {tier} tier",
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Tier"] = tier
        return response

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try JWT subject first
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            # Use token hash as identifier (don't decode JWT in middleware)
            import hashlib
            return hashlib.sha256(auth.encode()).hexdigest()[:16]

        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()

        return request.client.host if request.client else "unknown"

    def _get_tier(self, request: Request) -> str:
        """Determine rate limit tier from request."""
        # Check for ops role in JWT claims (simplified)
        # In production, decode JWT and check role claim
        auth = request.headers.get("Authorization", "")
        if auth:
            return "authenticated"
        return "anonymous"

    def _check_rate(self, key: str, now: float, window: float, limit: int) -> bool:
        """Check if request is within rate limit."""
        # Try Redis first
        if self._redis is None and self._redis_url:
            try:
                import redis
                self._redis = redis.from_url(self._redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

        if self._redis:
            return self._check_rate_redis(key, now, window, limit)
        else:
            return self._check_rate_memory(key, now, window, limit)

    def _check_rate_redis(self, key: str, now: float, window: float, limit: int) -> bool:
        """Redis sliding window rate check."""
        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, int(window) + 1)
        results = pipe.execute()
        count = results[2]
        return count <= limit

    def _check_rate_memory(self, key: str, now: float, window: float, limit: int) -> bool:
        """In-memory sliding window rate check (fallback)."""
        timestamps = self._in_memory[key]
        # Remove expired entries
        self._in_memory[key] = [t for t in timestamps if t > now - window]
        self._in_memory[key].append(now)
        return len(self._in_memory[key]) <= limit
