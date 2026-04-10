from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rag.generator import MedicalRAGService


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One message in the session (e.g. from Streamlit chat_state)."""

    role: str
    content: str


def _format_symptoms(symptoms: str | Sequence[str] | None) -> str:
    if symptoms is None:
        return ""
    if isinstance(symptoms, str):
        s = symptoms.strip()
        return s
    parts = [str(s).strip() for s in symptoms if s and str(s).strip()]
    return ", ".join(parts)


def _humanize_prediction(prediction: str) -> str:
    p = (prediction or "").strip()
    if not p:
        return "no single leading label from the model yet"
    low = p.lower()
    if low == "normal" or low.startswith("normal "):
        return "normal lungs on this model's read"
    return p


def _rag_snippet(rag_explanation: str, *, max_chars: int = 380) -> str:
    text = (rag_explanation or "").strip()
    if not text:
        return "No reference-backed narrative was available for this turn."
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    cut = collapsed[: max_chars - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _confidence_statement(model_confidence: float | None) -> str:
    """
    Optional one-sentence confidence messaging for chat responses.

    Requirements:
    - If confidence < 0.6: include the exact sentence lead-in "The model is not highly confident..."
    - If confidence > 0.7: include a short reassurance
    """
    if model_confidence is None:
        return ""
    try:
        c = float(model_confidence)
    except (TypeError, ValueError):
        return ""
    c = max(0.0, min(1.0, c))

    if c < 0.6:
        return (
            "The model is not highly confident in this prediction, so treat it as a tentative signal; "
            "other labels may be similarly plausible."
        )
    if c > 0.7:
        return (
            "The model is fairly confident in this prediction, though it should still be interpreted alongside "
            "symptoms, image quality, and clinician judgment."
        )
    return ""


def _format_similar_memory(similar: Sequence[Any] | None, *, max_items: int = 3, max_chars: int = 240) -> str:
    """
    Format retrieved similar items (e.g. from ``rag.vector_store.retrieve_similar``) as a short inline snippet.

    Accepts items like ``{"text": str, "score": float}`` or plain strings.
    """
    if not similar:
        return ""

    parts: list[str] = []
    for item in list(similar)[: max(0, int(max_items))]:
        text: str | None = None
        score: float | None = None

        if isinstance(item, str):
            text = item
        elif isinstance(item, Mapping):
            raw_text = item.get("text")
            if isinstance(raw_text, str):
                text = raw_text
            raw_score = item.get("score")
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)

        if not text:
            continue

        collapsed = " ".join(text.split())
        if len(collapsed) > max_chars:
            cut = collapsed[: max_chars - 1]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            collapsed = cut + "…"

        if score is None:
            parts.append(f"“{collapsed}”")
        else:
            parts.append(f"“{collapsed}” (score {score:.2f})")

    return "; ".join(parts)


def _stable_choice(seed_text: str, options: Sequence[str]) -> str:
    if not options:
        return ""
    if len(options) == 1:
        return str(options[0])
    h = hashlib.sha256((seed_text or "").encode("utf-8", errors="ignore")).digest()
    idx = int.from_bytes(h[:8], "big") % len(options)
    return str(options[idx])


def _last_user_snippet(history: Sequence[ChatTurn], *, max_chars: int = 100) -> str:
    for turn in reversed(history):
        if turn.role.lower() == "user" and (turn.content or "").strip():
            last_user = turn.content.strip()
            return last_user if len(last_user) <= max_chars else last_user[: max_chars - 1] + "…"
    return ""


def _chat_continuity_prefix(history: Sequence[ChatTurn], *, seed_text: str) -> str:
    if not history:
        return ""
    snippet = _last_user_snippet(history)
    if snippet and len(history) >= 2:
        # Include the user's requested phrasings among the options.
        leadins = (
            f'Following up on your earlier message (“{snippet}”), ',
            f'From what you told me earlier (“{snippet}”), ',
            f'Based on what you shared earlier (“{snippet}”), ',
            f'Picking up where we left off (“{snippet}”), ',
        )
        return _stable_choice(seed_text, leadins)
    return _stable_choice(seed_text, ("Building on our conversation, ", "Continuing from earlier, "))


def _question_followup(user_question: str) -> str:
    q = (user_question or "").strip()
    if not q:
        return (
            "If you have a specific question, ask it here and we can connect it to your symptoms "
            "and the model read in plain language."
        )
    low = q.lower()
    if any(x in low for x in ("why", "how come", "reason", "because")):
        return (
            f'You asked “{q}”: the model combines your symptoms with the image pattern it was trained on; '
            "the explanation above describes typical patterns from reference notes—not proof of your diagnosis."
        )
    if any(
        x in low
        for x in (
            "worried",
            "worry",
            "serious",
            "danger",
            "emergency",
            "should i go",
            "see a doctor",
        )
    ):
        return (
            f'You asked “{q}”: this assistant only summarizes model output and reference text for education. '
            "Seek urgent or emergency care for severe shortness of breath, chest pain, confusion, fainting, "
            "or rapidly worsening symptoms; otherwise follow up with a clinician who can examine you."
        )
    if any(x in low for x in ("what does", "what is", "explain", "mean")):
        return (
            f'You asked “{q}”: in short, treat the predicted label as one line of evidence alongside your exam, '
            "vitals, and course of illness—the longer explanation above expands on that with reference context."
        )
    return (
        f'You asked “{q}”: the paragraphs above tie your latest model read and symptoms to reference material; '
        "use them as context only, not as a substitute for in-person medical advice."
    )


def generate_conversational_response(
    *,
    user_question: str,
    chat_history: Sequence[ChatTurn],
    latest_prediction: str,
    symptoms: str | Sequence[str] | None,
    rag_explanation: str,
    model_confidence: float | None = None,
    similar: Sequence[Any] | None = None,
) -> str:
    """
    Template-based conversational reply: merges prediction, symptoms, RAG text, and the user's question.

    No LLM: callers should obtain ``rag_explanation`` from ``MedicalRAGService.explain`` (or equivalent).
    """
    sym = _format_symptoms(symptoms)
    pred_phrase = _humanize_prediction(latest_prediction)
    conf_bit = ""
    if model_confidence is not None:
        c = max(0.0, min(1.0, float(model_confidence)))
        conf_bit = f" (about {c:.0%} model confidence)"

    seed = "|".join(
        [
            (user_question or "").strip(),
            (pred_phrase or "").strip(),
            sym,
            _last_user_snippet(chat_history),
        ]
    )
    continuity = _chat_continuity_prefix(chat_history, seed_text=seed)

    if sym:
        openers = (
            f"{continuity}Based on your symptoms so far ({sym}) and the model’s latest read of {pred_phrase}{conf_bit}, here’s how to think about it.",
            f"{continuity}Based on your symptoms so far ({sym}), and with the model reading this as {pred_phrase}{conf_bit}, here’s a plain-language walkthrough.",
            f"{continuity}With {sym} in the picture, and the model currently leaning toward {pred_phrase}{conf_bit}, here’s the short version.",
            f"{continuity}Given what you’ve shared ({sym}) and a model label of {pred_phrase}{conf_bit}, here’s a concise explanation.",
        )
    else:
        openers = (
            f"{continuity}With the model’s latest read of {pred_phrase}{conf_bit}, here’s a concise walkthrough.",
            f"{continuity}Given the current model label of {pred_phrase}{conf_bit}, here’s the plain-language summary.",
            f"{continuity}Here’s how I’d summarize the current model output ({pred_phrase}{conf_bit}) in context.",
        )

    opener = _stable_choice(seed, openers).strip()
    if opener:
        opener = opener[0].upper() + opener[1:]

    conf_stmt = _confidence_statement(model_confidence)
    rag_part = _rag_snippet(rag_explanation)
    similar_snippet = _format_similar_memory(similar)
    memory_bit = ""
    if similar_snippet:
        memory_bit = _stable_choice(
            seed,
            (
                f" Relevant prior context from this session: {similar_snippet}.",
                f" From earlier in this chat, the most relevant context was: {similar_snippet}.",
            ),
        )
    body = " ".join(
        [
            opener,
            conf_stmt,
            f"The reference-backed explanation covers: {rag_part}",
            memory_bit,
            _question_followup(user_question),
            "If symptoms persist, consider following up with a clinician for an in-person assessment.",
            "Consider consulting a doctor, especially if you’re unsure how to interpret the result or symptoms change.",
            "This is educational context only—not a diagnosis or treatment plan.",
        ]
    )
    return " ".join(body.split())


def conversational_reply_with_rag(
    rag: MedicalRAGService,
    *,
    user_question: str,
    chat_history: Sequence[ChatTurn],
    latest_prediction: str,
    symptoms: str | Sequence[str] | None = None,
    top_k: int | None = None,
    similar: Sequence[Any] | None = None,
) -> dict[str, str]:
    """
    Retrieve + template RAG explanation, then produce a single conversational string.

    Returns ``{"explanation": ..., "reply": ...}`` so callers can show both full RAG and chat text if desired.
    """
    out = rag.explain(prediction=latest_prediction, symptoms=symptoms, top_k=top_k)
    explanation = str(out.get("explanation", "")).strip()
    reply = generate_conversational_response(
        user_question=user_question,
        chat_history=chat_history,
        latest_prediction=latest_prediction,
        symptoms=symptoms,
        rag_explanation=explanation,
        similar=similar,
    )
    return {"explanation": explanation, "reply": reply}


def reply_for_chat_turn(
    *,
    user_question: str,
    chat_history: Sequence[ChatTurn],
    disease: str,
    cached_rag_explanation: str,
    symptoms_at_predict: str | Sequence[str] | None,
    model_confidence: float | None = None,
    fill_empty_explanation_with_rag: MedicalRAGService | None = None,
    rag_top_k: int | None = None,
    similar: Sequence[Any] | None = None,
) -> str:
    """
    One assistant turn: uses the last **Predict** RAG explanation when present (no fusion re-run).

    If the cache is empty and ``fill_empty_explanation_with_rag`` is set, runs a single
    ``MedicalRAGService.explain`` (retrieval + template/LLM generator only).
    """
    rag_text = (cached_rag_explanation or "").strip()
    if not rag_text and fill_empty_explanation_with_rag is not None:
        out = fill_empty_explanation_with_rag.explain(
            prediction=disease,
            symptoms=symptoms_at_predict,
            top_k=rag_top_k,
        )
        rag_text = str(out.get("explanation", "")).strip()
    if not rag_text:
        rag_text = (
            "No reference-backed explanation is stored for this session yet. "
            "Run **Predict** with RAG enabled to attach retrieval context to the result."
        )
    return generate_conversational_response(
        user_question=user_question,
        chat_history=chat_history,
        latest_prediction=disease,
        symptoms=symptoms_at_predict,
        rag_explanation=rag_text,
        model_confidence=model_confidence,
        similar=similar,
    )
