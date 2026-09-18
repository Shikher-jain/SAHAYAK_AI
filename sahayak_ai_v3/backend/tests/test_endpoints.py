import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from backend.api.main import app

client = TestClient(app)

@pytest.fixture
def mock_retrieval():
    with patch("backend.api.routers.chat.logger") as mock_logger:
        yield mock_logger

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint(mock_retrieval):
    response = client.post(
        "/chat/ask",
        json={"query": "What is AI?", "session_id": "test"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "evidence_used" in data

def test_ingest_pdf_endpoint():
    response = client.post(
        "/ingest/pdf",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
