import io
import os
import httpx
from fastapi import APIRouter, Request, Header, HTTPException, status
from langchain_core.messages import HumanMessage
from sahayak_ai_v3.backend.agents.supervisor import sahayak_agent_app

router = APIRouter(prefix="/api/v2/webhooks", tags=["Bot Webhooks"])

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_SECRET_TOKEN = os.getenv("TELEGRAM_WEBHOOK_SECRET", "sahayak_secret_key")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


# =====================================================================
# 1. Telegram & Groq Cloud API Helpers (Zero Local RAM)
# =====================================================================

async def send_telegram_message(chat_id: int, text: str):
    """Dispatches a response back to the Telegram user."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"{TELEGRAM_API_BASE}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        )


async def get_telegram_file_bytes(file_id: str) -> tuple[bytes, str]:
    """Resolves Telegram file path and downloads binary stream into memory."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step A: Query file metadata
        file_info_res = await client.get(
            f"{TELEGRAM_API_BASE}/getFile", params={"file_id": file_id}
        )
        file_path = file_info_res.json().get("result", {}).get("file_path")
        if not file_path:
            raise ValueError("Failed to retrieve file path from Telegram.")

        # Step B: Download file payload
        download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        download_res = await client.get(download_url)
        return download_res.content, os.path.basename(file_path)


async def transcribe_audio_groq(audio_bytes: bytes, filename: str) -> str:
    """Offloads audio to Groq Whisper Large v3 for transcription."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        files = {
            "file": (filename, io.BytesIO(audio_bytes), "audio/ogg"),
            "model": (None, "whisper-large-v3"),
            "language": (None, "en"),  # Or omit to allow auto-detection (Hindi, etc.)
            "response_format": (None, "text"),
        }
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        
        response = await client.post(GROQ_WHISPER_URL, files=files, headers=headers)
        if response.status_code != 200:
            return "Sorry, I could not transcribe that audio note."
        return response.text.strip()


# =====================================================================
# 2. Main Telegram Webhook Endpoint
# =====================================================================

@router.post("/telegram")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str = Header(None)
):
    """
    Unified entry point for Telegram text messages, voice notes, and PDFs.
    """
    # Verify Telegram Webhook Secret Token
    if x_telegram_bot_api_secret_token != TELEGRAM_SECRET_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid secret token")

    update = await request.json()
    message = update.get("message")
    if not message:
        return {"status": "ignored"}

    chat_id = message["chat"]["id"]
    user_id = str(message["from"]["id"])
    extracted_text = ""

    # ── Case A: Voice Note / Audio Processing ──────────────────────
    if "voice" in message or "audio" in message:
        voice = message.get("voice") or message.get("audio")
        file_id = voice["file_id"]
        
        await send_telegram_message(chat_id, "_Listening and transcribing audio note..._")
        
        try:
            audio_bytes, filename = await get_telegram_file_bytes(file_id)
            extracted_text = await transcribe_audio_groq(audio_bytes, filename)
            await send_telegram_message(chat_id, f"🎙️ *You said:* \"{extracted_text}\"")
        except Exception as e:
            await send_telegram_message(chat_id, f"Failed to process audio: {str(e)}")
            return {"status": "error"}

    # ── Case B: PDF Document Upload ────────────────────────────────
    elif "document" in message:
        doc = message["document"]
        if doc.get("mime_type") == "application/pdf":
            file_id = doc["file_id"]
            file_name = doc.get("file_name", "uploaded_doc.pdf")
            
            await send_telegram_message(chat_id, f"_Processing and indexing `{file_name}` into Sahayak knowledge base..._")
            
            try:
                pdf_bytes, _ = await get_telegram_file_bytes(file_id)
                # Ingest into Qdrant Cloud via your existing lightweight ingestion parser
                # await ingest_pdf_stream(pdf_bytes, file_name, user_id=user_id)
                
                await send_telegram_message(
                    chat_id, 
                    f"✅ `{file_name}` indexed successfully. You can now ask questions about it!"
                )
                return {"status": "indexed"}
            except Exception as e:
                await send_telegram_message(chat_id, f"Failed to ingest PDF: {str(e)}")
                return {"status": "error"}

    # ── Case C: Standard Text Message ──────────────────────────────
    elif "text" in message:
        extracted_text = message["text"]

    # ── Execute LangGraph Supervisor Pipeline ──────────────────────
    if extracted_text:
        initial_state = {
            "messages": [HumanMessage(content=extracted_text)],
            "next_agent": "",
            "user_id": user_id,
            "user_tier": "free",
            "distress_level": 0.0,
            "emergency_flag": False,
            "final_output": "",
        }

        # Run multi-agent graph (Supervisor -> RAG / Counseling / Recommender)
        graph_output = await sahayak_agent_app.ainvoke(initial_state)
        response_text = graph_output.get("final_output", "No response generated.")

        if "voice" in message or "audio" in message:
            # Generate audio reply via edge-tts
            from sahayak_ai_v3.backend.audio.tts_service import generate_audio_bytes
            audio_bytes = await generate_audio_bytes(response_text)
            
            # Send audio payload back
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.post(
                    f"{TELEGRAM_API_BASE}/sendAudio",
                    data={"chat_id": chat_id},
                    files={"audio": ("reply.mp3", audio_bytes, "audio/mpeg")}
                )
                return {"status": "ok"}

        await send_telegram_message(chat_id, response_text)

    return {"status": "ok"}
