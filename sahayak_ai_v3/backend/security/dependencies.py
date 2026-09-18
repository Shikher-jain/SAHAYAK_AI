import logging
from fastapi import Request, HTTPException, status
from sahayak_ai_v3.backend.security.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)

async def RateLimitDependency(request: Request):
    """
    FastAPI Dependency that extracts the client identifier and enforces
    the Upstash Redis token bucket rate limit.
    """
    # Prefer explicit user headers if passed from auth middleware, otherwise fallback to IP
    identifier = request.headers.get("X-User-ID")
    if not identifier:
        client_ip = request.headers.get("X-Forwarded-For")
        if client_ip:
            identifier = client_ip.split(",")[0].strip()
        elif request.client:
            identifier = request.client.host
        else:
            identifier = "unknown_client"
            
    # Try to consume 1 token. Defaults: 10 max tokens, 0.5 per second (2 secs per token)
    allowed = await rate_limiter.acquire(identifier=identifier, capacity=10, refill_rate=0.5)
    
    if not allowed:
        logger.warning(f"Throttled request from {identifier} to {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before sending more requests.",
            headers={"Retry-After": "2"}
        )
