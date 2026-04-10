from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag.embedder import Embedder
from rag.knowledge_base import MedicalDocument, MedicalKnowledgeBase


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    """One retrieved knowledge snippet with similarity score."""

    document_id: str
    text: str
    score: float


class MedicalRetriever:
    """Retrieve top-k knowledge base passages for a predicted disease (query string)."""

    def __init__(
        self,
        knowledge_base: MedicalKnowledgeBase,
        embedder: Embedder,
        *,
        query_prefix: str = "chest x-ray imaging findings diagnosis ",
    ) -> None:
        self._kb = knowledge_base
        self._embedder = embedder
        self._query_prefix = query_prefix
        self._docs: list[MedicalDocument] = knowledge_base.as_medical_documents()
        self._doc_matrix: np.ndarray | None = None

    def index(self) -> None:
        """Embed all knowledge base texts. Call after construction or when KB changes."""
        texts = [d.text for d in self._docs]
        if texts:
            self._embedder.fit(texts)
        self._doc_matrix = self._embedder.embed(texts) if texts else np.zeros((0, self._embedder.dim), dtype=np.float32)

    def retrieve(self, predicted_disease: str, *, top_k: int = 3) -> list[RetrievedPassage]:
        if self._doc_matrix is None:
            raise RuntimeError("Call retriever.index() before retrieve().")
        if top_k <= 0 or not self._docs:
            return []

        query_text = f"{self._query_prefix}{predicted_disease}".strip()
        q = self._embedder.embed([query_text])
        sims = cosine_similarity(q, self._doc_matrix)[0]
        order = np.argsort(-sims)
        k = min(top_k, len(self._docs))
        out: list[RetrievedPassage] = []
        for i in order[:k]:
            idx = int(i)
            doc = self._docs[idx]
            out.append(RetrievedPassage(document_id=doc.id, text=doc.text, score=float(sims[idx])))
        return out
