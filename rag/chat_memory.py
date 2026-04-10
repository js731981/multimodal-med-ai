from __future__ import annotations

from typing import Dict, List

from rag.storage_backend import get_storage_backend


class ChatMemoryUnavailableError(RuntimeError):
    """Raised when chat memory cannot be accessed due to Redis unavailability."""


def save_message(session_id: str, role: str, content: str) -> None:
    """
    Append one chat message to the configured backend and keep only the last 20 messages.
    """
    get_storage_backend().save_message(session_id, role, content)


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Return the chat history as a list of ``{role, content}`` dicts.

    If no history exists, returns an empty list.
    """
    return get_storage_backend().get_chat_history(session_id)

