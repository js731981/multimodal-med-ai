"""Lazy Redis client from environment (redis-py)."""

from __future__ import annotations

import os
from typing import Optional

import redis

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6379
_SOCKET_CONNECT_TIMEOUT = 1

_client: Optional[redis.Redis] = None


def _env_host() -> str:
    raw = os.environ.get("REDIS_HOST", DEFAULT_HOST)
    stripped = raw.strip()
    return stripped if stripped else DEFAULT_HOST


def _env_port() -> int:
    raw = os.environ.get("REDIS_PORT", str(DEFAULT_PORT))
    try:
        return int(str(raw).strip())
    except ValueError:
        return DEFAULT_PORT


def get_redis_client() -> redis.Redis:
    """Return a shared ``Redis`` client (lazy connect; no I/O on first call)."""
    global _client
    if _client is None:
        _client = redis.Redis(
            host=_env_host(),
            port=_env_port(),
            decode_responses=True,
            socket_connect_timeout=_SOCKET_CONNECT_TIMEOUT,
        )
    return _client


def is_redis_available() -> bool:
    """Return True if Redis accepts a PING; never raises."""
    try:
        get_redis_client().ping()
        return True
    except (redis.RedisError, OSError):
        return False
