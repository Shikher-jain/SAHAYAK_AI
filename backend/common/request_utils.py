"""Helper utilities for parsing HTTP request parameters seamlessly across JSON and Form data."""
from typing import Any, Dict
from fastapi import Request


async def get_request_data(request: Request) -> Dict[str, Any]:
    """Extract payload dictionary from an incoming request, auto-detecting JSON vs Form data."""
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                return body
        except Exception:
            pass

    if "form" in content_type:
        try:
            form_data = await request.form()
            return dict(form_data)
        except Exception:
            pass

    # Fallback parsing attempts
    try:
        body = await request.json()
        if isinstance(body, dict):
            return body
    except Exception:
        pass

    try:
        form_data = await request.form()
        return dict(form_data)
    except Exception:
        pass

    return {}
