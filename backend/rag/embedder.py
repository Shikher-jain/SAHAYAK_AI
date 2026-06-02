from __future__ import annotations

import uuid

import numpy as np

# DEPRECATED: import from backend.common.embedder instead (unified singleton embedder).
from backend.common.embedder import embed_text as _embed_text


def embed_query(text: str) -> np.ndarray:
    return _embed_text(text)


class Embedder:
    def embed_text(self, text: str) -> list[float]:
        return embed_query(text).tolist()

    def generate_id(self) -> str:
        return f"doc_{uuid.uuid4().hex}"