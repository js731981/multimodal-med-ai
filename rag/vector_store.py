from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from rag.storage_backend import get_storage_backend


Vector = List[float]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Pure-Python cosine similarity for two same-length vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    denom = math.sqrt(na) * math.sqrt(nb)
    return float(dot / denom) if denom else 0.0


def _normalize(v: Sequence[float]) -> Vector:
    n2 = 0.0
    out = [float(x) for x in v]
    for x in out:
        n2 += x * x
    n = math.sqrt(n2)
    if not n:
        return [0.0 for _ in out]
    return [x / n for x in out]


class _Embedder:
    """Minimal text->vector embedder with sentence-transformers optional."""

    def __init__(self, *, dim: int = 384, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._dim = int(dim)
        self._model_name = model_name
        self._model = None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            # Fall back to deterministic stub embeddings if sentence-transformers
            # isn't installed/available.
            self._model = None

    @property
    def dim(self) -> int:
        return self._dim

    def embed_one(self, text: str) -> Vector:
        text = (text or "").strip()
        if not text:
            return [0.0] * self._dim

        if self._model is not None:
            emb = self._model.encode([text], convert_to_numpy=False, show_progress_bar=False)[0]
            return _normalize([float(x) for x in emb])

        # Stub: hash -> pseudo-random but deterministic dense vector in [-1, 1].
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        out: List[float] = []
        counter = 0
        while len(out) < self._dim:
            block = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
            counter += 1
            for byte in block:
                out.append((byte / 127.5) - 1.0)
                if len(out) >= self._dim:
                    break
        return _normalize(out)


_EMBEDDER = _Embedder()


def store_embedding(session_id: str, text: str) -> Dict[str, Any]:
    """
    Embed `text` and append to the session store.

    Storage:
      key: vec:{session_id}
      value: JSON list of {text, vector}
    """
    item = {"text": text, "vector": _EMBEDDER.embed_one(text)}
    return get_storage_backend().store_embedding(session_id, item)


def retrieve_similar(session_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Return top-k most similar items for `query` from this session.

    Output items:
      {text, vector, score}
    """
    if top_k <= 0:
        return []

    qv = _EMBEDDER.embed_one(query)
    items = get_storage_backend().retrieve_items(session_id)

    scored: List[Dict[str, Any]] = []
    for it in items:
        vec = it.get("vector")
        txt = it.get("text")
        if not isinstance(txt, str) or not isinstance(vec, list):
            continue
        score = _cosine_similarity(qv, vec)
        scored.append({"text": txt, "vector": vec, "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: min(top_k, len(scored))]

