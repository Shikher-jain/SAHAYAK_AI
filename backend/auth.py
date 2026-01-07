from fastapi import Header, HTTPException

API_KEY = "your_api_key"  # TODO: Replace with secure config

def api_key_auth(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
# backend/auth.py

from fastapi import Header, HTTPException

# Simple API key store (in production, use secure DB)
API_KEY = "your_api_key"  # TODO: Replace with secure config

def api_key_auth(api_key: str = Header(...)):
    """
    Simple API key authentication for FastAPI endpoints
    """
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
