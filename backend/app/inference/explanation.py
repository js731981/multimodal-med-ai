from __future__ import annotations

from typing import Protocol

from rag.generator import MedicalRAGService


class MultimodalExplanationProvider(Protocol):
    """Post-fusion explanation (e.g. RAG). Runs only after disease/confidence are known."""

    def explain(self, *, disease: str, symptoms: str | None) -> str: ...


class MedicalRAGExplanationProvider:
    """Adapts ``MedicalRAGService`` to the pipeline protocol (retrieve + generate text)."""

    def __init__(self, rag: MedicalRAGService) -> None:
        self._rag = rag

    def explain(self, *, disease: str, symptoms: str | None) -> str:
        out = self._rag.explain(prediction=disease, symptoms=symptoms)
        return str(out.get("explanation", ""))
