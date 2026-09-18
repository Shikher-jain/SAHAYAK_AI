from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    SAHAYAK_PIPELINE_VERSION: Literal["legacy", "v2_advanced"] = "legacy"
    
    # API Keys
    HF_TOKEN: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    
    # Cloud DBs
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    NEO4J_URI: str = ""
    NEO4J_USER: str = ""
    NEO4J_PASSWORD: str = ""
    # Sahayak AI v2 Advanced Toggles
    CRAG_ENABLED: bool = True
    GRAPH_RAG_ENABLED: bool = True
    RERANKER_ENABLED: bool = True
    HYBRID_SEARCH_ENABLED: bool = True
    MAX_RETRIEVAL_ATTEMPTS: int = 2
    DENSE_TOP_K: int = 50
    SPARSE_TOP_K: int = 50
    GRAPH_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    FINAL_CONTEXT_K: int = 5

    # Groq model used by the v2 agents (Whisper stays on "whisper-large-v3")
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # v2 Monetization (Stripe)
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_ID: str = ""
    STRIPE_SUCCESS_URL: str = "http://localhost:3000/upgrade/success"
    STRIPE_CANCEL_URL: str = "http://localhost:3000/upgrade"

    # Payment tier / session store path (sqlite default; point to Neon URL later)
    SAHAYAK_DB_PATH: str = "./data/sahayak_payments.db"

    # --- WhatsApp Cloud API (Task 4 of the v2 webhook migration) ---
    # Meta WhatsApp Cloud API credentials. Verify token is YOUR OWN random
    # string (set in the Meta App Dashboard); Meta echoes it back during the
    # GET verification handshake. Phone number ID is the business phone's id.
    WHATSAPP_TOKEN: str = ""
    WHATSAPP_VERIFY_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""
    WHATSAPP_GRAPH_BASE: str = "https://graph.facebook.com/v19.0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
