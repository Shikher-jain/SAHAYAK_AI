import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from sahayak_ai_v3.backend.core.logging import correlation_id_var

logger = logging.getLogger(__name__)

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every request has a unique correlation ID.
    The ID is attached to the response headers and structured JSON logs.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract existing ID from headers, or generate a new UUID4
        correlation_id = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
            
        # Set the context variable for the duration of this request
        token = correlation_id_var.set(correlation_id)
        
        start_time = time.time()
        
        # Get Client IP securely (handle proxy headers if present)
        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Record structured access log
            process_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"{request.method} {request.url.path} HTTP/{request.scope.get('http_version', '1.1')}",
                extra={
                    "extra_data": {
                        "path": request.url.path,
                        "method": request.method,
                        "client_ip": client_ip,
                        "status_code": response.status_code,
                        "latency_ms": round(process_time_ms, 2)
                    }
                }
            )
            
            # Attach ID to outbound response
            response.headers["X-Correlation-ID"] = correlation_id
            return response
            
        finally:
            # Reset context variable to prevent leakage across async tasks
            correlation_id_var.reset(token)
