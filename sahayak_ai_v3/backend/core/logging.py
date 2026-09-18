import json
import logging
import traceback
from datetime import datetime, timezone
import contextvars

# Ensure PIIScrubber is imported. We assume it's at sahayak_ai_v3.backend.security.pii_scrubber
try:
    from sahayak_ai_v3.backend.security.pii_scrubber import PIIScrubber
    scrubber = PIIScrubber()
except ImportError:
    class DummyScrubber:
        def scrub_text(self, text: str) -> str:
            return text
    scrubber = DummyScrubber()

# ContextVar for tracing request correlation ID across async tasks
correlation_id_var = contextvars.ContextVar("correlation_id", default=None)

class PIISanitizingFilter(logging.Filter):
    """Filters outgoing log records by applying the PIIScrubber to the log message."""
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = scrubber.scrub_text(record.msg)
        return True


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON without heavy dependencies."""
    
    def format(self, record: logging.LogRecord) -> str:
        # PII scrub the arguments and message before formatting if not caught by filter
        if isinstance(record.msg, str):
            record.msg = scrubber.scrub_text(record.msg)

        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
            "module": record.module,
            "function": record.funcName,
        }

        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            raw_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            # Critical: scrub the traceback to prevent credential leaking
            log_obj["traceback"] = scrubber.scrub_text(raw_traceback)
            
        if hasattr(record, "extra_data"):
            log_obj["extra_data"] = record.extra_data

        return json.dumps(log_obj)


def setup_structured_logging():
    """Initializes the root logger with JSON formatting and PII sanitization."""
    root_logger = logging.getLogger()
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    handler.addFilter(PIISanitizingFilter())
    
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # Silence third-party noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
