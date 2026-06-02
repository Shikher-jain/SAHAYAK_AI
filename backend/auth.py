from __future__ import annotations

import os

from fastapi import Header, HTTPException


def _configured_api_key() -> str:
    return os.getenv("SAHAYAK_API_KEY", "").strip()


def api_key_auth(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = _configured_api_key()
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
