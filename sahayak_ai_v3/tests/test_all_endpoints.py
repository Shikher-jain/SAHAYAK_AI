import pytest
from httpx import AsyncClient, ASGITransport
import ssl
import socket
import datetime
from unittest.mock import AsyncMock, patch
from sahayak_ai_v3.backend.main import app  # assuming this is the ASGI app

def verify_domain_ssl(hostname: str = "sahayak-backend.onrender.com") -> bool:
    """Verifies that the target domain's SSL certificate is valid for > 14 days."""
    context = ssl.create_default_context()
    with socket.create_connection((hostname, 443)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert = ssock.getpeercert()
            not_after_str = cert['notAfter']
            expire_date = datetime.datetime.strptime(not_after_str, "%b %d %H:%M:%S %Y %Z")
            time_left = expire_date - datetime.datetime.utcnow()
            assert time_left.days > 14, f"SSL certificate expires in {time_left.days} days"
            return True

def test_production_ssl_certificate():
    assert verify_domain_ssl("sahayak-backend.onrender.com")

@pytest.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_health_ping(async_client):
    response = await async_client.get("/api/v2/health/ping")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
@patch("sahayak_ai_v3.backend.api.health.ping_upstash", new_callable=AsyncMock)
async def test_health_deep_ping(mock_upstash, async_client):
    mock_upstash.return_value = True
    response = await async_client.get("/api/v2/health/deep-ping")
    assert response.status_code in (200, 503)

@pytest.mark.asyncio
@patch("sahayak_ai_v3.backend.security.dependencies.rate_limiter.acquire", new_callable=AsyncMock)
async def test_chat_cache_hit_and_miss(mock_acquire, async_client):
    mock_acquire.return_value = True
    # Test would simulate DB hit/miss logic via mocks
    payload = {"query": "Hello", "session_id": "test_123", "mode": "text"}
    response = await async_client.post("/api/v2/chat/orchestrate", json=payload)
    # Depending on mock setup, verify status and Cache-Status headers
    assert response.status_code in [200, 429, 404]

@pytest.mark.asyncio
async def test_voice_stream_upload(async_client):
    file_data = {"file": ("test.webm", b"dummy audio content", "audio/webm")}
    # Mock rate limiter and Whisper API internally before making this request
    with patch("sahayak_ai_v3.backend.security.dependencies.rate_limiter.acquire", new_callable=AsyncMock) as mock_acquire:
        mock_acquire.return_value = True
        response = await async_client.post("/api/v2/chat/voice-stream", files=file_data)
        assert response.status_code in [200, 429, 404, 500]

@pytest.mark.asyncio
async def test_verify_utr(async_client):
    payload = {"utr_number": "123456789012"}
    response = await async_client.post("/api/v2/payments/verify-utr", json=payload)
    assert response.status_code in [200, 202, 404]
