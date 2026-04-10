"""
Streamlit UI for multimodal chest X-ray + clinical text inference.

Run from the repository root:
  streamlit run frontend/app.py
"""

from __future__ import annotations

import html
import io
import os
import sys
from pathlib import Path

# Repo root on path so `frontend.*` and `backend.*` imports work when Streamlit sets cwd/script dir.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from PIL import Image, UnidentifiedImageError

from frontend.backend_client import (
    InferenceClientError,
    run_chat_reply,
    run_multimodal_prediction,
)
from rag.chat_memory import get_chat_history, save_message
from rag.vector_store import retrieve_similar, store_embedding


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
        .mm-title { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.15rem; color: #0f172a; }
        .mm-sub { color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; line-height: 1.5; }
        .mm-section { font-size: 1.05rem; font-weight: 600; color: #334155; margin: 0 0 0.75rem 0; padding-bottom: 0.35rem; border-bottom: 1px solid #e2e8f0; }
        .mm-placeholder {
            border: 1px dashed #cbd5e1; border-radius: 12px; padding: 1.5rem 1.25rem; text-align: center;
            color: #64748b; background: #f8fafc; font-size: 0.95rem; line-height: 1.55; margin-top: 0.25rem;
        }
        .prediction-banner {
            padding: 1rem 1.15rem; border-radius: 12px; margin: 0 0 0.5rem 0;
            border: 1px solid transparent; line-height: 1.45;
        }
        .prediction-banner--ok {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border-color: #6ee7b7; color: #065f46;
        }
        .prediction-banner--alert {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-color: #fca5a5; color: #991b1b;
        }
        .prediction-banner .pb-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.85; font-weight: 600; }
        .prediction-banner .pb-value { font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; }
        div[data-testid="stExpander"] { background-color: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0; }
        div[data-testid="column"] > div { min-height: 0; }
        .mm-confidence {
            border-radius: 12px; padding: 1rem 1.1rem; margin: 0.75rem 0 1rem 0;
            border: 1px solid #e2e8f0; background: #f8fafc;
        }
        .mm-confidence--high { border-color: #6ee7b7; background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%); }
        .mm-confidence--mid { border-color: #fcd34d; background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%); }
        .mm-confidence--low { border-color: #fca5a5; background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%); }
        .mm-confidence-header { display: flex; align-items: center; gap: 0.35rem; flex-wrap: wrap; margin-bottom: 0.35rem; }
        .mm-confidence-label { font-size: 0.95rem; font-weight: 600; color: #334155; }
        .mm-confidence-tip {
            cursor: help; font-size: 0.85rem; width: 1.15rem; height: 1.15rem; line-height: 1.1rem;
            text-align: center; border-radius: 999px; border: 1px solid #cbd5e1; color: #64748b;
            background: rgba(255,255,255,0.85); user-select: none;
        }
        .mm-confidence-band { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.15rem; }
        .mm-confidence--high .mm-confidence-band { color: #047857; }
        .mm-confidence--mid .mm-confidence-band { color: #b45309; }
        .mm-confidence--low .mm-confidence-band { color: #b91c1c; }
        .mm-confidence-value { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; margin: 0.2rem 0 0.55rem 0; }
        .mm-confidence--high .mm-confidence-value { color: #065f46; }
        .mm-confidence--mid .mm-confidence-value { color: #92400e; }
        .mm-confidence--low .mm-confidence-value { color: #991b1b; }
        .mm-confidence-track { height: 8px; border-radius: 999px; background: rgba(15, 23, 42, 0.08); overflow: hidden; }
        .mm-confidence-fill { height: 100%; border-radius: 999px; transition: width 0.2s ease; }
        .mm-confidence--high .mm-confidence-fill { background: linear-gradient(90deg, #10b981, #34d399); }
        .mm-confidence--mid .mm-confidence-fill { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
        .mm-confidence--low .mm-confidence-fill { background: linear-gradient(90deg, #ef4444, #f87171); }
        .mm-confidence-legend { font-size: 0.78rem; color: #64748b; margin-top: 0.55rem; line-height: 1.45; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_CONFIDENCE_TOOLTIP = (
    "This value is the model's estimated probability for the predicted class—not clinical certainty. "
    "Lower scores mean the model is less decisive; other labels may be nearly as likely. "
    "Interpret together with image quality, symptoms, and clinician judgment."
)


def _render_confidence_level(confidence: float) -> None:
    """Show tier-colored confidence (green >0.7, yellow 0.4–0.7, red <0.4) with label and tooltip."""
    c = max(0.0, min(1.0, float(confidence)))
    conf_pct = c * 100.0
    if c > 0.7:
        tier_class = "mm-confidence--high"
        band = "High confidence"
    elif c >= 0.4:
        tier_class = "mm-confidence--mid"
        band = "Uncertain"
    else:
        tier_class = "mm-confidence--low"
        band = "Low confidence"
    tip_attr = html.escape(_CONFIDENCE_TOOLTIP, quote=True)
    st.markdown(
        f'<div class="mm-confidence {tier_class}">'
        '<div class="mm-confidence-header">'
        '<span class="mm-confidence-label">Model Confidence Level</span>'
        f'<span class="mm-confidence-tip" title="{tip_attr}">?</span>'
        "</div>"
        f'<div class="mm-confidence-band">{html.escape(band)}</div>'
        f'<div class="mm-confidence-value">{conf_pct:.1f}%</div>'
        '<div class="mm-confidence-track">'
        f'<div class="mm-confidence-fill" style="width: {conf_pct:.1f}%;"></div>'
        "</div>"
        '<p class="mm-confidence-legend">'
        "Green &gt;70% · Yellow 40–70% · Red &lt;40% (probability for the predicted class)."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _is_benign_prediction(disease: str) -> bool:
    """Treat typical negative CXR labels as benign (green); other classes as alert (red)."""
    d = (disease or "").strip().lower()
    benign = frozenset({"normal", "no finding", "healthy", "negative"})
    return d in benign or (d.startswith("normal") and len(d) <= 12)


def _load_uploaded_image(uploaded) -> Image.Image | None:
    if uploaded is None:
        return None
    try:
        data = uploaded.getvalue()
        with Image.open(io.BytesIO(data)) as im:
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise InferenceClientError("Could not read the uploaded file as an image. Try PNG or JPEG.")


def main() -> None:
    st.set_page_config(
        page_title="Multimodal Medical AI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    try:
        from backend.app.inference.image_model_impl import resolve_image_checkpoint_path

        _ckpt = resolve_image_checkpoint_path(None)
        st.sidebar.caption(f"Image checkpoint: `{_ckpt}`")
    except FileNotFoundError:
        st.sidebar.caption(
            "Image checkpoint: not found — add `resnet_model.pth` or `dummy_smoke.pth` under "
            "`models/image_model/`, or set MMEDAI_IMAGE_CHECKPOINT_PATH."
        )
    _inject_styles()

    st.markdown('<p class="mm-title">Multimodal Medical AI System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mm-sub">Fused chest X-ray and clinical text inference with optional Grad-CAM.</p>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "session_id" not in st.session_state:
        # Streamlit session-scoped stable identifier for Redis-backed chat memory.
        import uuid

        st.session_state["session_id"] = uuid.uuid4().hex

    if os.environ.get("MMEDAI_DEBUG_SESSION_ID", "").strip().lower() in ("1", "true", "yes", "on"):
        st.sidebar.caption(f"Session ID: `{st.session_state['session_id']}`")
    else:
        if st.sidebar.checkbox("Show session id (debug)", value=False, key="mm_show_session_id"):
            st.sidebar.caption(f"Session ID: `{st.session_state['session_id']}`")

    for key, default in (
        ("mm_result", None),
        ("mm_error", None),
        ("mm_symptoms_at_predict", None),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    col_input, col_results = st.columns(2, gap="large")

    with col_input:
        st.markdown('<p class="mm-section">📥 Input</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Chest X-ray image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Optional if you provide sufficient clinical text; required for Grad-CAM.",
            key="mm_cxr_uploader",
        )
        symptoms = st.text_area(
            "Symptoms",
            placeholder="e.g. fever and productive cough for 5 days; decreased breath sounds right base",
            max_chars=4000,
            height=140,
        )
        predict = st.button("Predict", type="primary", use_container_width=True)

    if predict:
        st.session_state.mm_error = None
        st.session_state.mm_result = None
        st.session_state.mm_symptoms_at_predict = None
        text_stripped = (symptoms or "").strip()
        try:
            pil = _load_uploaded_image(uploaded) if uploaded is not None else None
        except InferenceClientError as e:
            st.session_state.mm_error = str(e)
            pil = None

        if st.session_state.mm_error is None:
            if pil is None and not text_stripped:
                st.session_state.mm_error = (
                    "Please provide at least a chest X-ray image or symptom text before running Predict."
                )
            else:
                try:
                    with st.spinner("Processing..."):
                        st.session_state.mm_result = run_multimodal_prediction(
                            image=pil,
                            symptoms=symptoms,
                        )
                        st.session_state.mm_symptoms_at_predict = text_stripped or None
                except InferenceClientError as e:
                    st.session_state.mm_error = str(e)
                except Exception:
                    st.session_state.mm_error = (
                        "Something went wrong while running inference. "
                        "Check that models are available and try again."
                    )

    err = st.session_state.mm_error
    res = st.session_state.mm_result

    with col_results:
        st.markdown('<p class="mm-section">📊 Results</p>', unsafe_allow_html=True)
        if err:
            st.error(err)
        elif res is None:
            st.markdown(
                '<div class="mm-placeholder">'
                "<strong>No input yet</strong><br/>"
                "Upload a chest X-ray and/or enter symptoms on the left, then click <strong>Predict</strong> "
                "to see the fused prediction and optional Grad-CAM."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            benign = _is_benign_prediction(res.disease)
            safe_label = html.escape(res.disease)
            banner_class = "prediction-banner--ok" if benign else "prediction-banner--alert"
            st.markdown(
                f'<div class="prediction-banner {banner_class}">'
                f'<span class="pb-label">Predicted condition</span>'
                f'<div class="pb-value">{safe_label}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            pred_msg = f"**Predicted condition:** {res.disease}"
            if benign:
                st.success(pred_msg)
            else:
                st.error(pred_msg)
            st.caption(
                "Green vs red on the banner reflects the predicted **class** in this demo, not app health. "
                "The confidence card below uses a separate scale (model certainty for the top label)."
            )
            _render_confidence_level(res.confidence)

            st.markdown("##### Visual explanation")
            img_a, img_b = st.columns(2, gap="medium")
            with img_a:
                if uploaded is not None:
                    try:
                        show_pil = _load_uploaded_image(uploaded)
                        st.image(show_pil, caption="Original", use_container_width=True)
                    except InferenceClientError:
                        st.warning("Could not display the uploaded image.")
                else:
                    st.caption("**Original** — no image in this run (text-only inference).")

            with img_b:
                if res.gradcam_path and Path(res.gradcam_path).is_file():
                    st.image(res.gradcam_path, caption="Grad-CAM", use_container_width=True)
                elif uploaded is None:
                    st.caption("**Grad-CAM** — upload a chest X-ray to enable heatmaps.")
                else:
                    st.caption("**Grad-CAM** — no overlay was returned for this input.")

            if res.scores:
                with st.expander("Per-class scores", expanded=False):
                    for label, score in sorted(res.scores.items(), key=lambda x: -x[1]):
                        st.caption(f"{label}: {score:.3f}")
                        st.progress(float(score))

            with st.expander("🧠 Clinical Explanation", expanded=False):
                expl = (res.explanation or "").strip()
                if not expl:
                    st.info(
                        "No explanation text was returned (RAG may be disabled or returned empty)."
                    )
                else:
                    st.markdown(expl)
                st.caption(
                    "Educational context only—not a diagnosis. Follow institutional protocols "
                    "and clinician judgment."
                )

    st.divider()
    st.caption(
        "Demo stack: fused image + text embeddings • Grad-CAM targets the image head "
        "(class aligned with the fused label when possible)."
    )

    st.markdown('<p class="mm-section">💬 Ask Follow-up Questions</p>', unsafe_allow_html=True)
    st.caption(
        "Uses your last **Predict** result, its RAG explanation, and this thread—no second fusion run. "
        "Educational context only."
    )

    _, clear_col = st.columns([5, 1])
    with clear_col:
        if st.button("Clear chat", key="mm_clear_chat"):
            st.session_state["chat_history"] = []

    with st.container(height=400, border=True):
        session_id = str(st.session_state.get("session_id") or "").strip()
        redis_history: list[dict[str, str]] = []
        if session_id:
            try:
                redis_history = get_chat_history(session_id)
            except Exception:
                redis_history = []

        render_history = redis_history or st.session_state["chat_history"]
        if not render_history:
            st.markdown(
                '<p class="mm-placeholder" style="margin:0;">'
                "No messages yet — send a message with the chat input below."
                "</p>",
                unsafe_allow_html=True,
            )
        for msg in render_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    st.markdown("")  # small vertical breathing room above the fixed input
    if prompt := st.chat_input("Ask follow-up..."):
        msg = prompt.strip()
        if session_id:
            try:
                save_message(session_id, "user", msg)
                store_embedding(session_id, msg)
            except Exception:
                st.session_state["chat_history"].append({"role": "user", "content": msg})
        else:
            st.session_state["chat_history"].append({"role": "user", "content": msg})

        history = []
        similar = []
        if session_id:
            try:
                history = get_chat_history(session_id)
            except Exception:
                history = st.session_state["chat_history"]
            try:
                similar = retrieve_similar(session_id, msg)
            except Exception:
                similar = []
        else:
            history = st.session_state["chat_history"]

        reply = run_chat_reply(
            session_id=session_id,
            user_question=msg,
            chat_history=history,
            similar=similar,
            last=st.session_state.get("mm_result"),
            symptoms_at_predict=st.session_state.get("mm_symptoms_at_predict"),
        )

        if session_id:
            try:
                save_message(session_id, "assistant", reply)
                store_embedding(session_id, reply)
            except Exception:
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        else:
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()
