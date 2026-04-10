from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence


Vector = list[float]


class Embedder(ABC):
    """Abstract embedding generator.

    Keep this interface stable so embedding backends can be swapped without touching
    retriever/vector-store code.
    """

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Return one vector per input text."""

    def embed_one(self, text: str) -> Vector:
        return self.embed([text])[0]


@dataclass(frozen=True, slots=True)
class HashEmbedder(Embedder):
    """Deterministic placeholder embedder (no ML dependencies).

    Not semantically meaningful, but useful to keep the RAG pipeline runnable while
    you wire a real medical embedding model later.
    """

    dimension: int = 256

    @property
    def dim(self) -> int:
        return self.dimension

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_text(t) for t in texts]

    def _embed_text(self, text: str) -> Vector:
        # Produce a stable pseudo-random vector from SHA256 blocks.
        # Values are normalized to unit length for cosine similarity usage.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        needed = self.dimension
        out: list[float] = []

        counter = 0
        while len(out) < needed:
            block = hashlib.sha256(digest + counter.to_bytes(4, "little")).digest()
            for b in block:
                # map [0,255] -> [-1,1]
                out.append((b / 127.5) - 1.0)
                if len(out) >= needed:
                    break
            counter += 1

        return _l2_normalize(out)


def _l2_normalize(v: Iterable[float]) -> Vector:
    vv = list(v)
    n = math.sqrt(sum(x * x for x in vv)) or 1.0
    return [x / n for x in vv]

