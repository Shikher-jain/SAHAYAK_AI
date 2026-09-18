import pytest
from unittest.mock import AsyncMock, patch
from fastapi import Request, HTTPException
import time

from sahayak_ai_v3.backend.security.rate_limiter import TokenBucketRateLimiter
from sahayak_ai_v3.backend.security.dependencies import RateLimitDependency

@pytest.fixture
def mock_redis():
    with patch("sahayak_ai_v3.backend.security.rate_limiter.redis") as mock:
        mock.eval = AsyncMock()
        yield mock

@pytest.fixture
def rate_limiter(mock_redis):
    # Initialize the rate limiter (it will use the mocked redis client)
    return TokenBucketRateLimiter()


@pytest.mark.asyncio
async def test_acquire_success_token_consumed(rate_limiter, mock_redis):
    # Mock redis.eval to simulate success: [1, 9] (1 = allowed, 9 = remaining tokens)
    mock_redis.eval.return_value = [1, 9]
    
    identifier = "192.168.1.1"
    allowed = await rate_limiter.acquire(identifier, capacity=10, refill_rate=0.5)
    
    assert allowed is True
    mock_redis.eval.assert_called_once()
    
    # Verify Lua script args
    args, kwargs = mock_redis.eval.call_args
    assert kwargs["keys"] == [f"ratelimit:{identifier}"]
    assert kwargs["args"][0] == 10  # capacity
    assert kwargs["args"][1] == 0.5  # refill rate
    assert isinstance(kwargs["args"][2], float)  # current time


@pytest.mark.asyncio
async def test_acquire_failure_exhausted_bucket(rate_limiter, mock_redis):
    # Mock redis.eval to simulate failure: [0, 0] (0 = rejected)
    mock_redis.eval.return_value = [0, 0]
    
    allowed = await rate_limiter.acquire("spam_user", capacity=10, refill_rate=0.5)
    
    assert allowed is False
    mock_redis.eval.assert_called_once()


@pytest.mark.asyncio
async def test_dependency_allows_request(mock_redis):
    # Setup mock request
    request = Request(scope={
        "type": "http", 
        "headers": [(b"x-user-id", b"valid_user")], 
        "client": ("127.0.0.1", 8000),
        "path": "/api/v2/chat",
        "method": "POST"
    })
    
    # Mock rate_limiter.acquire globally for the dependency
    with patch("sahayak_ai_v3.backend.security.dependencies.rate_limiter.acquire", new_callable=AsyncMock) as mock_acquire:
        mock_acquire.return_value = True
        
        # Should not raise exception
        result = await RateLimitDependency(request)
        assert result is None
        mock_acquire.assert_called_once_with(identifier="valid_user", capacity=10, refill_rate=0.5)


@pytest.mark.asyncio
async def test_dependency_raises_429(mock_redis):
    # Setup mock request
    request = Request(scope={
        "type": "http", 
        "headers": [], 
        "client": ("192.168.1.100", 8000),
        "path": "/api/v2/chat",
        "method": "POST"
    })
    
    with patch("sahayak_ai_v3.backend.security.dependencies.rate_limiter.acquire", new_callable=AsyncMock) as mock_acquire:
        mock_acquire.return_value = False
        
        with pytest.raises(HTTPException) as exc_info:
            await RateLimitDependency(request)
            
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == "Rate limit exceeded. Please wait before sending more requests."
        assert exc_info.value.headers == {"Retry-After": "2"}
        mock_acquire.assert_called_once_with(identifier="192.168.1.100", capacity=10, refill_rate=0.5)


@pytest.mark.asyncio
async def test_redis_outage_fails_open(rate_limiter, mock_redis):
    # Simulate an Upstash Redis outage (Exception during eval)
    mock_redis.eval.side_effect = Exception("Upstash Redis connection timeout")
    
    # Even though Redis failed, the rate limiter MUST fail-open (return True) 
    # so we don't break the entire application if the monitoring system goes down.
    allowed = await rate_limiter.acquire("user_123")
    
    assert allowed is True
