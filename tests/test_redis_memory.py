"""
Integration checks for Redis-backed chat history and session embeddings.

Requires Redis (see REDIS_HOST / REDIS_PORT). Run with prints:

  python -m pytest tests/test_redis_memory.py -s
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.chat_memory import get_chat_history, save_message  # noqa: E402
from rag.vector_store import retrieve_similar, store_embedding  # noqa: E402
from utils.redis_client import get_redis_client, is_redis_available  # noqa: E402


@pytest.fixture
def session_id():
    if not is_redis_available():
        pytest.skip("Redis not available (ping failed)")
    sid = f"pytest-{uuid.uuid4().hex}"
    yield sid
    client = get_redis_client()
    client.delete(f"chat:{sid}", f"vec:{sid}")


def test_redis_chat_history_and_similarity(session_id: str) -> None:
    save_message(session_id, "user", "First user question")
    save_message(session_id, "user", "Second user question")
    save_message(session_id, "assistant", "Assistant reply")

    history = get_chat_history(session_id)
    print("\n--- get_chat_history ---\n", history, "\n--- end ---\n", sep="")

    assert len(history) == 3
    assert [h["role"] for h in history] == ["user", "user", "assistant"]

    texts = [
        "Patient reports mild headache for two days.",
        "Blood pressure reading was 128 over 82.",
        "No fever; follow-up in one week.",
    ]
    for t in texts:
        item = store_embedding(session_id, t)
        print("stored_embedding:", {"text": item["text"], "dim": len(item["vector"])})

    similar = retrieve_similar(session_id, "headache symptoms", top_k=3)
    print("\n--- retrieve_similar ---\n", similar, "\n--- end ---\n", sep="")

    assert isinstance(similar, list)
    assert len(similar) > 0
    assert all("text" in x and "score" in x for x in similar)
