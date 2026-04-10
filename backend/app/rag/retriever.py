from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

from backend.app.rag.embedder import Embedder, Vector


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    document: Document
    score: float


class VectorStore(ABC):
    """Abstract vector store.

    This interface is intentionally small so it can be backed by Qdrant, FAISS,
    pgvector, Pinecone, etc.
    """

    @abstractmethod
    def upsert(self, *, vectors: Sequence[Vector], documents: Sequence[Document]) -> None: ...

    @abstractmethod
    def query(self, *, vector: Vector, top_k: int) -> list[RetrievedDocument]: ...


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, *, query: str, top_k: int = 5) -> list[RetrievedDocument]: ...


@dataclass(slots=True)
class SimpleRetriever(Retriever):
    embedder: Embedder
    store: VectorStore

    def retrieve(self, *, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        qvec = self.embedder.embed_one(query)
        return self.store.query(vector=qvec, top_k=top_k)


@dataclass(slots=True)
class InMemoryVectorStore(VectorStore):
    """In-memory store for local/dev runs."""

    _vectors: list[Vector] = field(default_factory=list)
    _docs: list[Document] = field(default_factory=list)

    def upsert(self, *, vectors: Sequence[Vector], documents: Sequence[Document]) -> None:
        if len(vectors) != len(documents):
            raise ValueError("vectors and documents must have the same length")
        self._vectors.extend([list(v) for v in vectors])
        self._docs.extend(list(documents))

    def query(self, *, vector: Vector, top_k: int) -> list[RetrievedDocument]:
        if not self._vectors:
            return []

        scores: list[tuple[int, float]] = []
        for i, v in enumerate(self._vectors):
            scores.append((i, _cosine_sim(vector, v)))

        scores.sort(key=lambda x: x[1], reverse=True)
        out: list[RetrievedDocument] = []
        for i, s in scores[: max(0, top_k)]:
            out.append(RetrievedDocument(document=self._docs[i], score=float(s)))
        return out


@dataclass(slots=True)
class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store.

    This is intentionally a thin wrapper around the Qdrant client. It can be used
    as-is, but you may want to move collection creation/migrations elsewhere.
    """

    collection_name: str
    url: str | None = None
    api_key: str | None = None
    prefer_grpc: bool = False

    _client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        # Lazy import so local runs don't require qdrant running.
        from qdrant_client import QdrantClient  # type: ignore

        self._client = QdrantClient(url=self.url, api_key=self.api_key, prefer_grpc=self.prefer_grpc)
        return self._client

    def upsert(self, *, vectors: Sequence[Vector], documents: Sequence[Document]) -> None:
        if len(vectors) != len(documents):
            raise ValueError("vectors and documents must have the same length")

        client = self._get_client()

        # NOTE: This assumes the collection exists and is configured with the right vector size.
        # You can add collection creation logic here later if desired.
        from qdrant_client.http.models import PointStruct  # type: ignore

        points = []
        for vec, doc in zip(vectors, documents, strict=True):
            payload = {
                "text": doc.text,
                "source": doc.source,
                "metadata": dict(doc.metadata or {}),
            }
            points.append(PointStruct(id=doc.id, vector=list(vec), payload=payload))

        client.upsert(collection_name=self.collection_name, points=points)

    def query(self, *, vector: Vector, top_k: int) -> list[RetrievedDocument]:
        client = self._get_client()

        hits = client.search(
            collection_name=self.collection_name,
            query_vector=list(vector),
            limit=max(0, top_k),
            with_payload=True,
        )

        out: list[RetrievedDocument] = []
        for h in hits:
            payload = getattr(h, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", None)
            metadata = payload.get("metadata", {}) or {}
            doc = Document(
                id=str(getattr(h, "id", "")),
                text=str(text),
                source=source if source is None else str(source),
                metadata=metadata if isinstance(metadata, Mapping) else {"metadata": metadata},
            )
            out.append(RetrievedDocument(document=doc, score=float(getattr(h, "score", 0.0))))
        return out


def _cosine_sim(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if len(av) != len(bv):
        raise ValueError("cosine similarity requires vectors of same length")

    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av)) or 1.0
    nb = math.sqrt(sum(y * y for y in bv)) or 1.0
    return float(dot / (na * nb))

