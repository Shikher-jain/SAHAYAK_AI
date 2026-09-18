import logging
from typing import Optional, Dict, Any
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from sahayak_ai_v3.backend.core.logging import correlation_id_var

logger = logging.getLogger(__name__)

class SahayakBaseException(Exception):
    """Base exception for all domain-specific errors in Sahayak AI."""
    def __init__(self, message: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class AgentExecutionError(SahayakBaseException):
    """Raised when LangGraph sub-agents fail or hit limits (Groq, Qdrant)."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=502, error_code="AGENT_EXECUTION_FAILED", details=details)


class UpstreamServiceError(SahayakBaseException):
    """Raised when external APIs (Upstash, HF, Edge-TTS) timeout or fail."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=503, error_code="UPSTREAM_API_ERROR", details=details)


class PaymentVerificationError(SahayakBaseException):
    """Raised for invalid UTRs, hash mismatches, or bank gateway rejections."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, error_code="PAYMENT_VERIFICATION_FAILED", details=details)


class PIIViolationError(SahayakBaseException):
    """Raised if unscrubbed sensitive data escapes safety barriers."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=403, error_code="PII_VIOLATION_BLOCKED", details=details)


def register_exception_handlers(app: FastAPI):
    """Wires global exception handlers to the FastAPI application instance."""

    @app.exception_handler(SahayakBaseException)
    async def sahayak_exception_handler(request: Request, exc: SahayakBaseException):
        logger.error(f"Domain Error: {exc.error_code} - {exc.message}", extra={"extra_data": exc.details})
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "correlation_id": correlation_id_var.get()
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation Error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "UNPROCESSABLE_ENTITY",
                "message": "The request payload failed validation.",
                "details": exc.errors(),
                "correlation_id": correlation_id_var.get()
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(f"HTTP Error {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": str(exc.detail),
                "correlation_id": correlation_id_var.get()
            }
        )

    @app.exception_handler(Exception)
    async def catch_all_exception_handler(request: Request, exc: Exception):
        # By logging exc_info=True, the JSON formatter natively captures & scrubs the traceback
        logger.error("Unhandled Exception caught in global handler", exc_info=True)
        
        # Fire-and-forget alert dispatching to prevent blocking the response
        asyncio.create_task(alert_dispatcher.capture_exception(exc, request))
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal error occurred. Please contact support with the correlation ID.",
                "correlation_id": correlation_id_var.get()
            }
        )
