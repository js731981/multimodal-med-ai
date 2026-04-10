"""
RAG explanation smoke test: prediction + symptoms -> non-empty, keyword-grounded text.

Run from repo root with stdout visible:

  python -m pytest tests/test_rag.py -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag import build_default_medical_rag  # noqa: E402


@pytest.fixture
def rag_service():
    return build_default_medical_rag(top_k=3)


def test_rag_explanation_pneumonia_with_symptoms(rag_service):
    disease = "pneumonia"
    symptoms = "fever, cough"

    out = rag_service.explain(prediction=disease, symptoms=symptoms)
    explanation = str(out.get("explanation", "")).strip()

    print("\n--- RAG explanation (inspection) ---\n", explanation, "\n--- end ---\n", sep="")

    assert explanation, "Explanation must not be empty"
    lower = explanation.lower()
    assert "pneumonia" in lower, "Explanation should name or reflect the predicted disease"
    disease_related = (
        "lung",
        "infection",
        "infiltrat",
        "consolidat",
        "cough",
        "fever",
        "dyspnea",
    )
    assert any(k in lower for k in disease_related), (
        f"Explanation should contain disease- or symptom-related terms; got snippet: {lower[:200]!r}..."
    )
    assert "recommended next steps" in lower


def test_rag_explanation_normal_when_symptoms_suggest_pneumonia(rag_service):
    """Template (and LLM) paths should surface imaging–symptom discordance and recommend correlation."""
    out = rag_service.explain(prediction="normal", symptoms="high fever and productive cough for 3 days")
    explanation = str(out.get("explanation", "")).strip()
    assert explanation, "Explanation must not be empty"
    lower = explanation.lower()
    assert "although symptoms suggest" in lower
    assert "image analysis" in lower
    assert "clinical correlation recommended" in lower
    assert "recommended next steps" in lower
