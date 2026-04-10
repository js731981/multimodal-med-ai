from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class StorageBackend(Protocol):
    def save_message(self, session_id: str, role: str, content: str) -> None: ...
    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]: ...
    def store_embedding(self, session_id: str, item: Dict[str, Any]) -> Dict[str, Any]: ...
    def retrieve_items(self, session_id: str) -> List[Dict[str, Any]]: ...


_MAX_MESSAGES = 20


def _chat_key(session_id: str) -> str:
    return f"chat:{session_id}"


def _vec_key(session_id: str) -> str:
    return f"vec:{session_id}"


@dataclass(frozen=True)
class _InMemoryState:
    lock: threading.RLock
    chats: Dict[str, List[Dict[str, str]]]
    vecs: Dict[str, List[Dict[str, Any]]]


_MEM = _InMemoryState(lock=threading.RLock(), chats={}, vecs={})


class InMemoryBackend:
    def save_message(self, session_id: str, role: str, content: str) -> None:
        with _MEM.lock:
            items = _MEM.chats.get(session_id)
            if items is None:
                items = []
                _MEM.chats[session_id] = items
            items.append({"role": str(role), "content": str(content)})
            if len(items) > _MAX_MESSAGES:
                _MEM.chats[session_id] = items[-_MAX_MESSAGES:]

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        with _MEM.lock:
            items = _MEM.chats.get(session_id, [])
            return [dict(x) for x in items if isinstance(x, dict)]

    def store_embedding(self, session_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        with _MEM.lock:
            items = _MEM.vecs.get(session_id)
            if items is None:
                items = []
                _MEM.vecs[session_id] = items
            items.append(item)
        return item

    def retrieve_items(self, session_id: str) -> List[Dict[str, Any]]:
        with _MEM.lock:
            items = _MEM.vecs.get(session_id, [])
            return [dict(x) for x in items if isinstance(x, dict)]


class RedisBackend:
    def __init__(self) -> None:
        import redis  # local import to keep import-time flexible

        from utils.redis_client import get_redis_client

        self._redis = redis
        self._get_client = get_redis_client

    def save_message(self, session_id: str, role: str, content: str) -> None:
        key = _chat_key(session_id)
        payload = json.dumps({"role": str(role), "content": str(content)}, ensure_ascii=False)
        try:
            client = self._get_client()
            pipe = client.pipeline(transaction=True)
            pipe.rpush(key, payload)
            pipe.ltrim(key, -_MAX_MESSAGES, -1)
            pipe.execute()
        except (self._redis.RedisError, OSError) as e:
            # If Redis flakes at runtime, degrade gracefully rather than
            # breaking the calling layer.
            logger.warning("Redis error while saving message; falling back to in-memory.", exc_info=e)
            InMemoryBackend().save_message(session_id, role, content)

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        key = _chat_key(session_id)
        try:
            raw_items = self._get_client().lrange(key, 0, -1)
        except (self._redis.RedisError, OSError) as e:
            logger.warning("Redis error while loading history; falling back to in-memory.", exc_info=e)
            return InMemoryBackend().get_chat_history(session_id)

        if not raw_items:
            return []

        out: List[Dict[str, str]] = []
        for item in raw_items:
            try:
                obj: Any = json.loads(item)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            role = obj.get("role")
            content = obj.get("content")
            if isinstance(role, str) and isinstance(content, str):
                out.append({"role": role, "content": content})
        return out

    def store_embedding(self, session_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        key = _vec_key(session_id)
        try:
            client = self._get_client()
            raw = client.get(key)
            items: List[Dict[str, Any]] = []
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        items = [x for x in parsed if isinstance(x, dict)]
                except json.JSONDecodeError:
                    items = []
            items.append(item)
            client.set(key, json.dumps(items))
            return item
        except (self._redis.RedisError, OSError) as e:
            logger.warning("Redis error while storing embedding; falling back to in-memory.", exc_info=e)
            return InMemoryBackend().store_embedding(session_id, item)

    def retrieve_items(self, session_id: str) -> List[Dict[str, Any]]:
        key = _vec_key(session_id)
        try:
            raw = self._get_client().get(key)
        except (self._redis.RedisError, OSError) as e:
            logger.warning("Redis error while retrieving embeddings; falling back to in-memory.", exc_info=e)
            return InMemoryBackend().retrieve_items(session_id)

        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict)]


_BACKEND: Optional[StorageBackend] = None
_BACKEND_LOCK = threading.Lock()
_LOGGED_SELECTION = False


def get_storage_backend() -> StorageBackend:
    global _BACKEND, _LOGGED_SELECTION
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            return _BACKEND

        try:
            from utils.redis_client import is_redis_available

            if is_redis_available():
                _BACKEND = RedisBackend()
                if not _LOGGED_SELECTION:
                    logger.info("Using Redis")
                    _LOGGED_SELECTION = True
                return _BACKEND
        except Exception:
            # Any import/ping failures should result in safe fallback.
            pass

        _BACKEND = InMemoryBackend()
        if not _LOGGED_SELECTION:
            logger.info("Using in-memory fallback")
            _LOGGED_SELECTION = True
        return _BACKEND

