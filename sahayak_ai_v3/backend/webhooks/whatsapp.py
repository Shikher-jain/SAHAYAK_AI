"""WhatsApp Cloud API webhook (v2_advanced).

  GET /api/v2/webhooks/whatsapp      — Meta webhook verification (hub.challenge)
  POST /api/v2/webhooks/whatsapp     — inbound messages: text / audio / document

Voice notes are transcribed with Groq's hosted Whisper (whisper-large-v3) via
async httpx — zero local RAM (AGENTS.md Rule 1: no torch/whisper on the
512MB Render box). Media is streamed into memory (io.BytesIO); nothing touches
disk. Replies are sent back over the WhatsApp Cloud API.

The supervisor import is guarded: it prefers the canonical supervisor but
degrades gracefully to the legacy one, so a supervisor hiccup never crashes
the webhook router (Rule 4: external failures must not take down the server).
"""
import io
import logging
import os

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status

from sahayak_ai_v3.backend.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/webhooks", tags=["WhatsApp Webhook"])

GRAPH_BASE = "https://graph.facebook.com/v19.0"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3"


def _try_supervisor():
    """Return the supervisor app from the first importable module.

    Prefers the canonical `backend.agents` package; falls back to the legacy
    `backend.services.agents` one. Returns None if neither resolves so the
    webhook degrades to a polite fallback instead of crashing the router.
    """
    for module in (
        "backend.agents.supervisor",
        "backend.services.agents.supervisor",
    ):
        try:
            mod = __import__(module, fromlist=["supervisor_app"])
            app = getattr(mod, "supervisor_app", None) or getattr(
                mod, "sahayak_agent_app", None
            )
            if app is not None:
                return app
        except Exception as e:  # noqa: BLE001 - any import failure is a fallback
            logger.warning("Supervisor import %s failed: %s", module, e)
    return None


# =====================================================================
#  WhatsApp Cloud API: outbound (async httpx, zero local RAM)
# =====================================================================
async def send_whatsapp_message(to_phone: str, text: str) -> bool:
    """Send a text message back through the WhatsApp Cloud API."""
    if not (settings.WHATSAPP_TOKEN and settings.WHATSAPP_PHONE_NUMBER_ID):
        return False
    url = f"{GRAPH_BASE}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text[:4000]},
    }
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                logger.warning("WhatsApp send failed (%s): %s", resp.status_code, resp.text)
                return False
            return True
    except Exception as e:  # noqa: BLE001
        logger.warning("WhatsApp send raised: %s", e)
        return False


async def _resolve_media_url(media_id: str) -> str:
    """Step A: media-id -> hosted URL (requires Bearer token)."""
    url = f"{GRAPH_BASE}/{media_id}"
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    media_url = (data or {}).get("url")
    if not media_url:
        raise ValueError(f"No url in media metadata for {media_id}")
    return media_url


async def download_whatsapp_media(media_id: str) -> bytes:
    """Step B: download the resolved media as in-memory bytes.

    The Bearer token must be passed on THIS request too, or Meta rejects it.
    """
    media_url = await _resolve_media_url(media_id)
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(media_url, headers=headers)
        resp.raise_for_status()
        return resp.content


# =====================================================================
#  Groq hosted Whisper transcription (zero local model load)
# =====================================================================
async def transcribe_audio_groq(audio_bytes: bytes, filename: str) -> str:
    """Transcribe an audio blob (e.g. WhatsApp .ogg voice note) via Groq Whisper."""
    if not settings.GROQ_API_KEY:
        return ""
    files = {
        "file": (filename, io.BytesIO(audio_bytes), "audio/ogg"),
        "model": (None, WHISPER_MODEL),
        "response_format": (None, "text"),
    }
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(GROQ_WHISPER_URL, files=files, headers=headers)
        if resp.status_code >= 400:
            logger.warning("Groq transcription failed (%s): %s", resp.status_code, resp.text)
            return ""
        return resp.text.strip()


# =====================================================================
#  Inbound payload routing
# =====================================================================
async def _handle_user_message(payload: dict) -> None:
    """Extract the user message and push it through the supervisor + reply."""
    if not isinstance(payload, dict):
        return
    entry = (payload.get("entry") or [{}])[0]
    change = (entry.get("changes") or [{}])[0]
    value = change.get("value") or {}
    contact = (value.get("contacts") or [{}])[0]
    user_id = (contact or {}).get("wa_id") or ""
    messages = value.get("messages") or []
    if not messages or not user_id:
        return

    text = ""
    first = messages[0]
    msg_type = first.get("type")

    # ── Voice Note -> Whisper → Supervisor ──────────────────────────
    if msg_type == "audio":
        media_id = (first.get("audio") or {}).get("id")
        if media_id:
            await send_whatsapp_message(user_id, "I'm listening to your voice note...")
            try:
                audio_bytes = await download_whatsapp_media(media_id)
                text = await transcribe_audio_groq(audio_bytes, "voice.ogg")
            except Exception as e:  # noqa: BLE001
                logger.warning("Voice processing failed: %s", e)
                text = ""

    # ── Document -> acknowledge (v3 ingest upstream handles indexing) ─
    elif msg_type == "document":
        doc = first.get("document") or {}
        name = doc.get("filename", "document")
        await send_whatsapp_message(
            user_id, f"Received `{name}`. Full document indexing is handled in the v3 ingestion API."
        )
        return

    # ── Text ────────────────────────────────────────────────────────
    elif msg_type == "text":
        text = (first.get("text") or {}).get("body", "").strip()

    if not text:
        await send_whatsapp_message(
            user_id,
            "I couldn't understand that message. Try typing your question or sending a voice note.",
        )
        return

    supervisor = _try_supervisor()
    if supervisor is None:
        await send_whatsapp_message(user_id, "Sorry, the AI brain is not ready right now.")
        return

    try:
        initial = {
            "messages": [{"role": "user", "content": text}],
            "next_agent": "",
            "user_id": user_id,
            "user_tier": "free",
            "emergency_flag": False,
            "final_output": "",
        }
        result = await supervisor.ainvoke(initial)
        answer = (result or {}).get("final_output") or (result or {}).get("final_response") or ""
        
        if msg_type == "audio":
            # Generate audio reply via edge-tts
            from sahayak_ai_v3.backend.audio.tts_service import generate_audio_bytes
            import aiofiles
            import uuid
            
            audio_bytes = await generate_audio_bytes(answer)
            # Since WhatsApp requires a media upload, we'd need a separate endpoint for media upload,
            # but we can also use a public URL. For now we will just send text with an audio indicator
            # Alternatively, we could upload it to WA API and send by media ID.
            # To adhere to ZERO-RAM, uploading stream directly via httpx:
            url = f"{GRAPH_BASE}/{settings.WHATSAPP_PHONE_NUMBER_ID}/media"
            headers = {"Authorization": f"Bearer {settings.WHATSAPP_TOKEN}"}
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                upload_resp = await client.post(
                    url, 
                    headers=headers, 
                    data={"messaging_product": "whatsapp"},
                    files={"file": ("reply.ogg", audio_bytes, "audio/ogg")}
                )
                if upload_resp.status_code < 400:
                    media_id = upload_resp.json().get("id")
                    if media_id:
                        # Send audio message
                        msg_url = f"{GRAPH_BASE}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
                        payload = {
                            "messaging_product": "whatsapp",
                            "to": user_id,
                            "type": "audio",
                            "audio": {"id": media_id}
                        }
                        await client.post(msg_url, json=payload, headers={"Authorization": f"Bearer {settings.WHATSAPP_TOKEN}", "Content-Type": "application/json"})
                        return
                        
        await send_whatsapp_message(user_id, answer[:4000])
    except Exception as e:  # noqa: BLE001 - never let one turn kill the webhook
        logger.warning("Supervisor turn failed: %s", e)
        await send_whatsapp_message(user_id, "Something went wrong processing your request.")


# =====================================================================
#  Routes
# =====================================================================
@router.get("/whatsapp")
async def whatsapp_verify(
    hub_mode: str = Query(default=""),
    hub_verify_token: str = Query(default=""),
    hub_challenge: str = Query(default=""),
):
    """Meta webhook verification handshake. Returns the raw challenge."""
    if (
        hub_mode == "subscribe"
        and settings.WHATSAPP_VERIFY_TOKEN
        and hmac.compare_digest(hub_verify_token, settings.WHATSAPP_VERIFY_TOKEN)
    ):
        return Response(content=hub_challenge, media_type="text/plain")
    return Response(
        content="Verification failed", status_code=status.HTTP_403_FORBIDDEN
    )


@router.post("/whatsapp")
async def whatsapp_webhook(request: Request, background: BackgroundTasks):
    """Acknowledge immediately (always 200), process the turn in the background."""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = {}
    background.add_task(_handle_user_message, payload)
    return {"status": "ok"}
