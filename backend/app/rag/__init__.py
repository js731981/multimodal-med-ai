"""Retrieval-Augmented Generation (RAG) building blocks.

This package is interface-first so you can swap embedding models, vector stores,
retrievers, and generation backends independently.
"""

from .embedder import Embedder, HashEmbedder
from .generator import ExplanationGenerator, RAGExplanationPipeline, TemplateExplanationGenerator
from .retriever import (
    Document,
    RetrievedDocument,
    Retriever,
    VectorStore,
    InMemoryVectorStore,
    QdrantVectorStore,
    SimpleRetriever,
)

__all__ = [
    "Document",
    "RetrievedDocument",
    "Embedder",
    "HashEmbedder",
    "VectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "Retriever",
    "SimpleRetriever",
    "ExplanationGenerator",
    "TemplateExplanationGenerator",
    "RAGExplanationPipeline",
]

