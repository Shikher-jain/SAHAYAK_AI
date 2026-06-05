from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/admin/health")
    assert response.status_code == 200
    assert response.json()["available"] is not None

def test_auth_register():
    response = client.post("/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword",
        "role": "student"
    })
    assert response.status_code == 201
    assert response.json()["username"] == "testuser"
    assert "id" in response.json()

def test_auth_login():
    # First, ensure the user exists (registration might be needed if running tests in isolation)
    # For simplicity, assuming test_auth_register runs first or user already exists.
    response = client.post("/auth/login", json={
        "username": "testuser",
        "password": "testpassword"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"