"""MVP clinical symptom text → fixed-size embedding via TF-IDF (replace with ClinicalBERT later)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.app.models.interfaces import ModelInfo, TextModel


def _default_fit_corpus() -> list[str]:
    """Synthetic but English clinical-style lines so common symptom tokens appear in the TF-IDF vocabulary."""
    lines = [
        "fever chills night sweats rigors malaise fatigue weakness lethargy",
        "dry cough productive cough hemoptysis wheezing dyspnea orthopnea tachypnea",
        "chest pain pleuritic pain substernal pressure tightness palpitations syncope",
        "nausea vomiting diarrhea constipation abdominal pain dyspepsia melena hematochezia",
        "headache photophobia neck stiffness confusion altered mental status seizure",
        "sore throat odynophagia dysphagia hoarseness nasal congestion rhinorrhea sinus pressure",
        "joint pain swelling stiffness erythema warmth limited range of motion back pain",
        "rash pruritus urticaria vesicles petechiae jaundice cyanosis clubbing edema",
        "dysuria frequency urgency hematuria flank pain suprapubic pain incontinence retention",
        "weight loss anorexia early satiety lymphadenopathy night pain bone pain",
        "shortness of breath on exertion pleuritic chest pain orthopnea paroxysmal nocturnal dyspnea",
        "acute onset chronic course progressive worsening intermittent symptoms resolved spontaneously",
        "patient denies chest pain denies shortness of breath denies fever denies cough",
        "history of asthma copd congestive heart failure diabetes hypertension hyperlipidemia",
        "recent travel sick contacts hospitalization surgery antibiotic use immunosuppression pregnancy",
        "oxygen saturation blood pressure heart rate respiratory rate temperature general appearance",
        "lung sounds crackles wheezes diminished breath sounds pleural rub egophony",
        "abdominal tenderness rebound guarding distension bowel sounds hypoactive hyperactive",
        "extremity swelling calf pain unilateral bilateral erythema warmth dvt concern",
        "vision changes hearing loss tinnitus vertigo focal weakness numbness tingling gait instability",
    ]
    # Expand with deterministic n-gram-like phrases for richer vocabulary coverage.
    extra: list[str] = []
    for a in ("mild", "moderate", "severe", "acute", "chronic"):
        for b in ("cough", "fever", "pain", "fatigue", "nausea", "headache", "rash", "swelling"):
            extra.append(f"{a} {b} reported by patient")
            extra.append(f"{a} {b} worsened over several days")
    return lines + extra


class TextModelImpl(TextModel):
    """
    TF-IDF bag-of-words embedding for symptom / clinical free text.

    - ``load_model()`` fits (or loads) a :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    - ``predict`` / ``encode`` return float32 CPU tensors of shape ``[B, embedding_dim]``.

    The default fit corpus is a static in-module list so the service starts without an external
    artifact. For production, fit on your corpus, persist the vectorizer (e.g. joblib), and load
    via ``vectorizer_path`` or replace this class with a ClinicalBERT-based implementation that
    keeps the same ``TextModel`` contract.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        vectorizer_path: str | Path | None = None,
        fit_corpus: Sequence[str] | None = None,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        self._vectorizer_path = Path(vectorizer_path) if vectorizer_path else None
        self._fit_corpus: Sequence[str] | None = fit_corpus
        self._vectorizer: TfidfVectorizer | None = None
        self.info = ModelInfo(
            name="tfidf-clinical-text",
            version="0.1.0",
            extra={
                "embedding_dim": self.embedding_dim,
                "vectorizer_path": str(self._vectorizer_path) if self._vectorizer_path else None,
            },
        )

    def load_model(self) -> None:
        if self._vectorizer is not None:
            return

        if self._vectorizer_path is not None and self._vectorizer_path.is_file():
            import joblib

            loaded = joblib.load(self._vectorizer_path)
            if not isinstance(loaded, TfidfVectorizer):
                raise TypeError(f"Expected TfidfVectorizer at {self._vectorizer_path}, got {type(loaded)}")
            self._vectorizer = loaded
            return

        corpus = list(self._fit_corpus) if self._fit_corpus is not None else _default_fit_corpus()
        if not corpus:
            raise ValueError("TF-IDF fit corpus is empty.")

        vec = TfidfVectorizer(
            max_features=self.embedding_dim,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            min_df=1,
            ngram_range=(1, 2),
        )
        vec.fit(corpus)
        self._vectorizer = vec

    def _ensure_loaded(self) -> None:
        if self._vectorizer is None:
            self.load_model()

    def _batch_strings(self, text: str | Sequence[str]) -> list[str]:
        if isinstance(text, str):
            return [text]
        return [str(t) for t in text]

    def _to_fixed_dense(self, X) -> np.ndarray:
        """``[B, vocab]`` → ``[B, embedding_dim]`` (pad or truncate)."""
        assert self._vectorizer is not None
        d = X.toarray().astype(np.float32, copy=False)
        n = d.shape[1]
        target = self.embedding_dim
        if n < target:
            pad = np.zeros((d.shape[0], target - n), dtype=np.float32)
            d = np.concatenate([d, pad], axis=1)
        elif n > target:
            d = d[:, :target]
        return d

    def predict(self, text: str | Sequence[str]) -> torch.Tensor:
        self._ensure_loaded()
        assert self._vectorizer is not None

        batch = self._batch_strings(text)
        # Empty strings → zero row (TF-IDF would return empty matrix with wrong width in some versions).
        cleaned = [s.strip() if s else "" for s in batch]
        if not cleaned:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32)

        X = self._vectorizer.transform(cleaned)
        dense = self._to_fixed_dense(X)
        # Explicit zeros for all-empty inputs (no vocabulary hits).
        for i, s in enumerate(cleaned):
            if not s:
                dense[i] = 0.0
        return torch.from_numpy(dense)

    def encode(self, text: str | Sequence[str]) -> torch.Tensor:
        """Alias for :meth:`predict` (multimodal pipeline expects ``encode``)."""
        return self.predict(text)
