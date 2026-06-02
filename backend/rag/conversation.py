"""In-memory conversational history keyed by session_id."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple


class ConversationManager:
    """Per-session sliding window of user/assistant turns."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._sessions: Dict[str, Deque[Tuple[str, str]]] = {}

    def _get_memory(self, session_id: str) -> Deque[Tuple[str, str]]:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.window_size)
        return self._sessions[session_id]

    def add_exchange(self, session_id: str, query: str, answer: str) -> None:
        memory = self._get_memory(session_id)
        memory.append((query, answer))

    def get_history(self, session_id: str) -> str:
        memory = self._sessions.get(session_id)
        if memory is None:
            return ""
        formatted = []
        for query, answer in memory:
            formatted.append(f"User: {query}")
            formatted.append(f"Assistant: {answer}")
        return "\n".join(formatted).strip()

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
