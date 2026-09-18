import os
import hashlib
import time
import logging
import asyncio
import traceback
import httpx
from fastapi import Request
from typing import Optional

try:
    from sahayak_ai_v3.backend.security.pii_scrubber import PIIScrubber
    scrubber = PIIScrubber()
except ImportError:
    class DummyScrubber:
        def scrub_text(self, text: str) -> str:
            return text
    scrubber = DummyScrubber()

from sahayak_ai_v3.backend.core.logging import correlation_id_var

logger = logging.getLogger(__name__)

# Config
TELEGRAM_BOT_TOKEN = os.getenv("ALERT_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("ALERT_TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK_URL = os.getenv("ALERT_DISCORD_WEBHOOK_URL")
COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))


class AlertDispatcher:
    """
    Singleton for dispatching non-blocking, PII-scrubbed, deduplicated alerts
    to external channels (Telegram / Discord) on catastrophic errors.
    """
    
    def __init__(self):
        self._cache = {}
        
    def _is_rate_limited(self, exc: Exception) -> bool:
        """Computes exception signature hash and applies sliding window cooldown."""
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            last_frame = tb[-1]
            signature = f"{type(exc).__name__}:{last_frame.filename}:{last_frame.lineno}"
        else:
            signature = f"{type(exc).__name__}:unknown_location"
            
        error_hash = hashlib.md5(signature.encode()).hexdigest()
        now = time.time()
        
        last_sent = self._cache.get(error_hash, 0)
        if now - last_sent < COOLDOWN_SECONDS:
            return True
            
        self._cache[error_hash] = now
        
        # Periodic cleanup of cache to prevent memory leak on 512MB limit
        if len(self._cache) > 1000:
            self._cache = {k: v for k, v in self._cache.items() if now - v < COOLDOWN_SECONDS}
            
        return False

    async def _send_telegram(self, title: str, fields: dict, traceback_snippet: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
            
        text = f"<b>{title}</b>\n\n"
        for k, v in fields.items():
            text += f"<b>{k}:</b> <code>{v}</code>\n"
            
        text += f"\n<b>Traceback:</b>\n<pre>{traceback_snippet}</pre>"
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }
        
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                await client.post(url, json=payload)
            except Exception as e:
                logger.error(f"Failed to dispatch Telegram alert: {e}")

    async def _send_discord(self, title: str, fields: dict, traceback_snippet: str):
        if not DISCORD_WEBHOOK_URL:
            return
            
        embed_fields = [{"name": k, "value": f"`{v}`", "inline": False} for k, v in fields.items()]
        
        payload = {
            "embeds": [{
                "title": title,
                "color": 0xE74C3C,
                "fields": embed_fields,
                "description": f"```python\n{traceback_snippet}\n```"
            }]
        }
        
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                await client.post(DISCORD_WEBHOOK_URL, json=payload)
            except Exception as e:
                logger.error(f"Failed to dispatch Discord alert: {e}")

    async def capture_exception(self, exc: Exception, request: Request):
        """Unified async entrypoint for capturing and dispatching alerts."""
        if self._is_rate_limited(exc):
            return
            
        # 1. Build Fields
        correlation_id = correlation_id_var.get() or "unknown"
        exc_type = type(exc).__name__
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path = f"{request.method} {request.url.path}"
        
        # 2. Extract & Scrub Traceback (truncated to 500 chars for safety & payload size)
        raw_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        scrubbed_traceback = scrubber.scrub_text(raw_traceback)[-500:]
        
        # 3. PII Scrub fields
        fields = {
            "Correlation ID": correlation_id,
            "Endpoint": scrubber.scrub_text(path),
            "Exception": exc_type,
            "Timestamp": timestamp
        }
        
        title = "🚨 [Sahayak AI] HTTP 500 Internal Error"
        
        # 4. Fire and forget concurrent requests
        await asyncio.gather(
            self._send_telegram(title, fields, scrubbed_traceback),
            self._send_discord(title, fields, scrubbed_traceback),
            return_exceptions=True
        )

alert_dispatcher = AlertDispatcher()
