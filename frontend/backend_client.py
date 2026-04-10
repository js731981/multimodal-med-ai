"""
In-process calls to the multimodal inference pipeline (same stack as the FastAPI backend).

Run the app from the repository root so imports resolve, e.g.:
  streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from rag.conversational import ChatTurn


class InferenceClientError(Exception):
    """User-facing inference failure (message is safe to show in the UI)."""


@dataclass(frozen=True, slots=True)
class MultimodalPredictionResult:
    disease: str
    confidence: float
    explanation: str
    gradcam_path: str | None
    scores: Mapping[str, float] | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_import_path() -> Path:
    root = _repo_root()
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    return root


def run_multimodal_prediction(
    *,
    image: Image.Image | None,
    symptoms: str,
) -> MultimodalPredictionResult:
    """
    Run ``MultimodalInferencePipeline`` and, when an image is present, write a Grad-CAM
    overlay via ``ImageModelImpl.save_gradcam`` when available.
    """
    _ensure_import_path()

    from backend.app.api.deps import get_pipeline
    from backend.app.inference.errors import InvalidImageInputError
    from backend.app.inference.schemas import MultimodalInput

    text = (symptoms or "").strip()
    if image is None and not text:
        raise InferenceClientError("Provide a chest X-ray image and/or symptom text, then click Predict.")

    try:
        pipeline = get_pipeline()
        result = pipeline.run(MultimodalInput(image=image, clinical_text=text or None))
    except InvalidImageInputError as e:
        raise InferenceClientError(str(e)) from e
    except FileNotFoundError as e:
        raise InferenceClientError(
            "A required model file is missing. Ensure checkpoints are under `models/` "
            f"(details: {e})"
        ) from e
    except ValueError as e:
        raise InferenceClientError(str(e)) from e

    gradcam_path = _try_save_gradcam(pipeline.image_model, image, result.disease)
    return MultimodalPredictionResult(
        disease=result.disease,
        confidence=result.confidence,
        explanation=result.explanation,
        gradcam_path=gradcam_path,
        scores=result.scores,
    )


def _try_save_gradcam(
    image_model: Any,
    image: Image.Image | None,
    fused_label: str,
) -> str | None:
    if image is None:
        return None
    save = getattr(image_model, "save_gradcam", None)
    if not callable(save):
        return None
    try:
        out = save(image, target_class=fused_label)
        return str(Path(out).resolve())
    except (ValueError, TypeError):
        try:
            out = save(image, target_class=None)
            return str(Path(out).resolve())
        except Exception:
            return None
    except (OSError, RuntimeError):
        return None


def _history_to_chat_turns(messages: Sequence[Mapping[str, str]]) -> list["ChatTurn"]:
    from rag.conversational import ChatTurn

    out: list[ChatTurn] = []
    for m in messages:
        role = str(m.get("role", "")).strip().lower() or "user"
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role not in ("user", "assistant"):
            role = "user"
        out.append(ChatTurn(role=role, content=content))
    return out


def run_chat_reply(
    *,
    session_id: str,
    user_question: str,
    chat_history: list[dict[str, str]] | None,
    similar: list[dict[str, object]] | None,
    last: MultimodalPredictionResult | None,
    symptoms_at_predict: str | None,
) -> str:
    """
    Follow-up chat grounded on the last **Predict** result: reuses stored RAG explanation
    and does not re-run image/text/fusion inference.
    """
    _ensure_import_path()

    from backend.app.config import get_settings
    from rag import build_default_medical_rag, reply_for_chat_turn
    from rag.chat_memory import get_chat_history as redis_get_chat_history

    q = (user_question or "").strip()
    if last is None:
        return (
            "There is no prediction loaded yet. Add a chest X-ray and/or symptoms, then click **Predict**. "
            "After that, chat can use your last result, its explanation, and this thread.\n\n"
            "_Educational demo only — not medical advice._"
        )

    s = get_settings()
    fill_rag = None
    if s.rag_enabled and not (last.explanation or "").strip():
        fill_rag = build_default_medical_rag(top_k=s.rag_top_k)

    history_src = chat_history
    try:
        # Prefer Redis session history when available.
        history_src = redis_get_chat_history(session_id)
    except Exception:
        pass

    history_src = history_src or []
    turns = _history_to_chat_turns(history_src[-3:])
    body = reply_for_chat_turn(
        user_question=q,
        chat_history=turns,
        disease=last.disease,
        cached_rag_explanation=last.explanation or "",
        symptoms_at_predict=(symptoms_at_predict or "").strip() or None,
        model_confidence=last.confidence,
        fill_empty_explanation_with_rag=fill_rag,
        rag_top_k=s.rag_top_k,
        similar=similar,
    )
    return body + "\n\n_Educational use only — not a diagnosis or treatment recommendation._"
