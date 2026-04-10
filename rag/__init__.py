"""Medical RAG: knowledge base, embeddings, retrieval, and explanation generation."""

from rag.conversational import (
    ChatTurn,
    conversational_reply_with_rag,
    generate_conversational_response,
    reply_for_chat_turn,
)
from rag.embedder import Embedder, SentenceTransformerEmbedder, TfidfEmbedder
from rag.generator import (
    LLMMedicalExplanationGenerator,
    MedicalExplanationGenerator,
    MedicalRAGService,
    TemplateMedicalExplanationGenerator,
)
from rag.knowledge_base import DEFAULT_MEDICAL_DOCUMENTS, MedicalDocument, MedicalKnowledgeBase
from rag.retriever import MedicalRetriever, RetrievedPassage

__all__ = [
    "ChatTurn",
    "DEFAULT_MEDICAL_DOCUMENTS",
    "Embedder",
    "LLMMedicalExplanationGenerator",
    "MedicalDocument",
    "MedicalExplanationGenerator",
    "MedicalKnowledgeBase",
    "MedicalRAGService",
    "conversational_reply_with_rag",
    "generate_conversational_response",
    "reply_for_chat_turn",
    "MedicalRetriever",
    "RetrievedPassage",
    "SentenceTransformerEmbedder",
    "TemplateMedicalExplanationGenerator",
    "TfidfEmbedder",
]


def build_default_medical_rag(
    *,
    knowledge_base: MedicalKnowledgeBase | None = None,
    top_k: int = 3,
) -> MedicalRAGService:
    """Convenience: default KB, TF-IDF embedder, template generator."""
    kb = knowledge_base or MedicalKnowledgeBase.default()
    embedder = TfidfEmbedder()
    retriever = MedicalRetriever(kb, embedder)
    retriever.index()
    gen = TemplateMedicalExplanationGenerator()
    return MedicalRAGService(retriever, gen, default_top_k=top_k)
