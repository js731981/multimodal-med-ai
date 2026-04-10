"""Template conversational RAG reply: prediction + symptoms + RAG + question."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag import (  # noqa: E402
    ChatTurn,
    build_default_medical_rag,
    conversational_reply_with_rag,
    generate_conversational_response,
    reply_for_chat_turn,
)


def test_generate_conversational_opener_normal_and_symptoms():
    rag_text = (
        "Early pneumonia may not show infiltrates immediately. "
        "Clinical correlation recommended when fever and cough persist."
    )
    out = generate_conversational_response(
        user_question="What does this mean for me?",
        chat_history=(),
        latest_prediction="normal",
        symptoms=["fever", "cough"],
        rag_explanation=rag_text,
    )
    low = out.lower()
    assert "normal lungs" in low
    assert "fever" in low and "cough" in low
    assert "reference-backed explanation" in low
    assert "early pneumonia" in low or "infiltrates" in low
    assert "if symptoms persist" in low
    assert "consider consulting a doctor" in low
    assert "educational" in low


def test_generate_conversational_chat_continuity():
    history = (
        ChatTurn("user", "I have had a cough for a week."),
        ChatTurn("assistant", "Thanks for sharing."),
        ChatTurn("user", "Could it be pneumonia?"),
    )
    out = generate_conversational_response(
        user_question="Should I worry?",
        chat_history=history,
        latest_prediction="pneumonia",
        symptoms="fever",
        rag_explanation="Pneumonia often presents with fever and cough.",
    )
    assert any(
        phrase in out
        for phrase in (
            "Following up on your earlier message",
            "From what you told me earlier",
            "Based on what you shared earlier",
            "Picking up where we left off",
        )
    )
    assert "pneumonia" in out.lower()


def test_generate_conversational_includes_confidence():
    out = generate_conversational_response(
        user_question="What does this mean?",
        chat_history=(),
        latest_prediction="pneumonia",
        symptoms=None,
        rag_explanation="Some RAG text.",
        model_confidence=0.82,
    )
    assert "82%" in out or "82" in out
    assert "fairly confident" in out.lower()


def test_generate_conversational_low_confidence_adds_uncertainty_statement():
    out = generate_conversational_response(
        user_question="What does this mean?",
        chat_history=(),
        latest_prediction="pneumonia",
        symptoms=None,
        rag_explanation="Some RAG text.",
        model_confidence=0.59,
    )
    assert "The model is not highly confident" in out


def test_generate_conversational_mid_confidence_adds_no_confidence_statement():
    out = generate_conversational_response(
        user_question="What does this mean?",
        chat_history=(),
        latest_prediction="pneumonia",
        symptoms=None,
        rag_explanation="Some RAG text.",
        model_confidence=0.65,
    )
    low = out.lower()
    assert "the model is not highly confident" not in low
    assert "fairly confident" not in low


def test_generate_conversational_high_confidence_threshold_is_strict():
    # > 0.7 should reassure; exactly 0.7 should not.
    out_eq = generate_conversational_response(
        user_question="What does this mean?",
        chat_history=(),
        latest_prediction="pneumonia",
        symptoms=None,
        rag_explanation="Some RAG text.",
        model_confidence=0.70,
    )
    out_hi = generate_conversational_response(
        user_question="What does this mean?",
        chat_history=(),
        latest_prediction="pneumonia",
        symptoms=None,
        rag_explanation="Some RAG text.",
        model_confidence=0.71,
    )
    assert "fairly confident" not in out_eq.lower()
    assert "fairly confident" in out_hi.lower()


def test_reply_for_chat_turn_uses_cached_explanation_without_second_rag(monkeypatch):
    calls = {"n": 0}
    rag = build_default_medical_rag(top_k=2)

    def _spy_explain(*args: object, **kwargs: object) -> dict[str, str]:
        calls["n"] += 1
        return rag.explain(*args, **kwargs)

    monkeypatch.setattr(rag, "explain", _spy_explain)
    out = reply_for_chat_turn(
        user_question="Why pneumonia?",
        chat_history=(),
        disease="pneumonia",
        cached_rag_explanation="Cached from predict.",
        symptoms_at_predict="cough",
        model_confidence=0.9,
        fill_empty_explanation_with_rag=rag,
        rag_top_k=2,
    )
    assert calls["n"] == 0
    assert "Cached from predict" in out
    assert "pneumonia" in out.lower()


def test_reply_for_chat_turn_fills_empty_when_rag_provided():
    rag = build_default_medical_rag(top_k=2)
    out = reply_for_chat_turn(
        user_question="Explain",
        chat_history=(),
        disease="pneumonia",
        cached_rag_explanation="",
        symptoms_at_predict="fever",
        fill_empty_explanation_with_rag=rag,
        rag_top_k=2,
    )
    assert len(out) > 50
    assert "pneumonia" in out.lower() or "fever" in out.lower()


def test_conversational_reply_with_rag_end_to_end():
    rag = build_default_medical_rag(top_k=2)
    result = conversational_reply_with_rag(
        rag,
        user_question="Why did the model say pneumonia?",
        chat_history=(),
        latest_prediction="pneumonia",
        symptoms="fever and cough",
    )
    assert "explanation" in result and "reply" in result
    assert result["explanation"]
    assert result["reply"]
    assert "fever" in result["reply"].lower()
    assert "pneumonia" in result["reply"].lower()
