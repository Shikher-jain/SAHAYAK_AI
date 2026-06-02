from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.ingestion.audio import transcribe_audio
from backend.ingestion.image import ocr_image
# DEPRECATED: import from backend.common.embedder instead (unified singleton embedder).
from backend.common.embedder import embed_text


def get_embedding(text: str) -> np.ndarray:
    return embed_text(text)


class EmbeddingProcessor:
    def embed_text(self, text: str) -> list[float]:
        return embed_text(text).tolist()

    def embed_image(self, image_path: str | Path) -> list[float]:
        extracted = ocr_image(image_path)
        return self.embed_text(extracted)

    def transcribe_audio(self, audio_path: str | Path) -> str:
        return transcribe_audio(audio_path)

    def embed_audio(self, audio_path: str | Path) -> tuple[list[float], str]:
        text = self.transcribe_audio(audio_path)
        return self.embed_text(text), text
