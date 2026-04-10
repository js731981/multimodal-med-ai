from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

from backend.app.rag.retriever import RetrievedDocument, Retriever


class ExplanationGenerator(ABC):
    """Abstract explanation generator (LLM-based or otherwise)."""

    @abstractmethod
    async def generate(self, *, predicted_disease: str, evidence: Sequence[RetrievedDocument]) -> str: ...


@dataclass(slots=True)
class TemplateExplanationGenerator(ExplanationGenerator):
    """Non-LLM placeholder generator to keep the pipeline runnable."""

    async def generate(self, *, predicted_disease: str, evidence: Sequence[RetrievedDocument]) -> str:
        evidence_lines = []
        for i, r in enumerate(evidence[:10], start=1):
            src = f" ({r.document.source})" if r.document.source else ""
            snippet = r.document.text.strip().replace("\n", " ")
            snippet = snippet[:240] + ("…" if len(snippet) > 240 else "")
            evidence_lines.append(f"{i}. score={r.score:.3f}{src} — {snippet}")

        joined = "\n".join(evidence_lines) if evidence_lines else "No supporting documents were retrieved."
        return (
            f"Predicted condition: {predicted_disease}\n\n"
            "Medical explanation (retrieval-augmented draft):\n"
            "- This explanation is generated from retrieved reference snippets and should be reviewed by a clinician.\n\n"
            "Retrieved evidence:\n"
            f"{joined}\n"
        )


@dataclass(slots=True)
class LLMExplanationGenerator(ExplanationGenerator):
    """LLM-based generator that accepts an injected chat callable.

    The callable should take a list of {role, content} messages and return a string.
    This keeps the module independent of any specific LLM SDK.
    """

    llm_chat: Callable[[list[dict[str, str]]], "str | object"]
    system_prompt: str = (
        "You are a medical assistant. Explain the predicted disease using only the provided context. "
        "If evidence is insufficient, say so and avoid making up facts. Include safety caveats."
    )

    async def generate(self, *, predicted_disease: str, evidence: Sequence[RetrievedDocument]) -> str:
        context = "\n\n".join(
            f"[doc {i} | score={r.score:.3f} | source={r.document.source or 'unknown'}]\n{r.document.text}"
            for i, r in enumerate(evidence[:10], start=1)
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Predicted disease: {predicted_disease}\n\n"
                    "Context documents:\n"
                    f"{context}\n\n"
                    "Task: Provide a clear medical explanation and tie statements to the context. "
                    "Add a short 'When to seek care' section and a brief uncertainty note."
                ),
            },
        ]

        result = self.llm_chat(messages)
        return getattr(result, "content", None) or str(result)


@dataclass(slots=True)
class RAGExplanationPipeline:
    """End-to-end flow: predicted disease -> retrieve docs -> generate explanation."""

    retriever: Retriever
    generator: ExplanationGenerator
    top_k: int = 5

    async def explain(self, *, predicted_disease: str) -> dict[str, object]:
        evidence = self.retriever.retrieve(query=predicted_disease, top_k=self.top_k)
        explanation = await self.generator.generate(predicted_disease=predicted_disease, evidence=evidence)
        return {
            "predicted_disease": predicted_disease,
            "explanation": explanation,
            "evidence": [
                {
                    "id": r.document.id,
                    "score": r.score,
                    "source": r.document.source,
                    "text": r.document.text,
                    "metadata": dict(r.document.metadata or {}),
                }
                for r in evidence
            ],
        }

