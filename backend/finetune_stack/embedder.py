# DEPRECATED: import from backend.common.embedder instead (unified singleton embedder).
from backend.common.embedder import embed_text, embed_texts, get_model

__all__ = ["embed_text", "embed_texts", "get_model"]
