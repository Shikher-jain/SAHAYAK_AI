"""
LangChain conversational memory — in-memory sessions keyed by session_id.

Uses ConversationBufferWindowMemory (last 5 exchanges per session).
"""
from __future__ import annotations

from typing import Dict

try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:  # pragma: no cover
    from langchain_community.chat_message_histories import ChatMessageHistory  # noqa: F401
    from langchain.memory import ConversationBufferWindowMemory  # type: ignore


class ConversationManager:
    """Per-session sliding window of user/assistant turns."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._sessions: Dict[str, ConversationBufferWindowMemory] = {}

    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationBufferWindowMemory(
                k=self.window_size,
                return_messages=True,
            )
        return self._sessions[session_id]

    def add_exchange(self, session_id: str, query: str, answer: str) -> None:
        memory = self._get_memory(session_id)
        memory.save_context({"input": query}, {"output": answer})

    def get_history(self, session_id: str) -> str:
        memory = self._sessions.get(session_id)
        if memory is None:
            return ""
        variables = memory.load_memory_variables({})
        history = variables.get("history")
        if isinstance(history, str):
            return history.strip()
        formatted = []
        for message in history or []:
            role = getattr(message, "type", "")
            content = getattr(message, "content", "")
            if role == "human":
                formatted.append(f"User: {content}")
            elif role == "ai":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(str(content))
        return "\n".join(formatted).strip()

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
