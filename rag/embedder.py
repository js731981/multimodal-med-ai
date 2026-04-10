from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer


Vector = list[float]


class Embedder(ABC):
    """Text → dense vectors. Subclasses may require ``fit`` on a corpus first."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    def fit(self, corpus: Sequence[str]) -> None:
        """Optional one-time fit on reference texts (no-op for stateless embedders)."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return shape (n_texts, dim) float32 array."""


class TfidfEmbedder(Embedder):
    """Simple bag-of-words embeddings via TF-IDF (scikit-learn)."""

    def __init__(self, *, max_features: int = 4096, ngram_range: tuple[int, int] = (1, 2)) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._max_features = max_features
        self._ngram_range = ngram_range
        self._vectorizer: TfidfVectorizer | None = None

    @property
    def dim(self) -> int:
        if self._vectorizer is None:
            return self._max_features
        return len(self._vectorizer.vocabulary_)

    def fit(self, corpus: Sequence[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            ngram_range=self._ngram_range,
            lowercase=True,
            strip_accents="unicode",
        )
        self._vectorizer.fit(list(corpus))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("TfidfEmbedder.embed() called before fit(); fit on knowledge base texts first.")
        m = self._vectorizer.transform(list(texts))
        return np.asarray(m.toarray(), dtype=np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Dense embeddings via sentence-transformers when installed; swap-in for better retrieval.

    Install: ``pip install sentence-transformers``. Until then, use ``TfidfEmbedder``.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)
