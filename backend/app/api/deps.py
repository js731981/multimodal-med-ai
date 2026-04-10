import base64
import binascii
import io
import logging
from functools import lru_cache
from typing import Any

from PIL import Image, UnidentifiedImageError

from backend.app.inference.errors import InvalidImageInputError
from backend.app.inference.explanation import MedicalRAGExplanationProvider
from backend.app.inference.loading import build_from_import_path
from backend.app.inference.pipeline import MultimodalInferencePipeline
from backend.app.inference.schemas import MultimodalInput
from backend.app.inference.settings import get_inference_settings
from rag import build_default_medical_rag


logger = logging.getLogger(__name__)


def decode_b64_image(image_b64: str | None) -> bytes | None:
    if not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64.strip())
    except binascii.Error as e:
        raise InvalidImageInputError("Invalid base64 image data.") from e
    if not raw:
        raise InvalidImageInputError("Empty image payload after base64 decode.")
    return raw


def _bytes_to_pil_rgb(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise InvalidImageInputError("Invalid or corrupted image data.") from e


@lru_cache
def get_pipeline() -> MultimodalInferencePipeline:
    s = get_inference_settings()

    image_model = build_from_import_path(s.image_model_path, kwargs=s.image_model_kwargs)
    text_model = build_from_import_path(s.text_model_path, kwargs=s.text_model_kwargs)
    fusion_kw = dict(s.fusion_model_kwargs)
    if "labels" not in fusion_kw:
        fusion_kw["labels"] = s.labels
    fusion_model = build_from_import_path(s.fusion_model_path, kwargs=fusion_kw)

    # Normalize text model to have `.encode(...)` for pipeline protocol.
    if text_model is not None and not hasattr(text_model, "encode") and hasattr(text_model, "predict"):
        text_model.encode = text_model.predict  # type: ignore[attr-defined]

    explanation_provider = None
    if s.rag_enabled:
        explanation_provider = MedicalRAGExplanationProvider(
            build_default_medical_rag(top_k=s.rag_top_k),
        )

    pipeline = MultimodalInferencePipeline(
        image_model=image_model,
        text_model=text_model,
        fusion_model=fusion_model,
        labels=s.labels,
        explanation_provider=explanation_provider,
    )
    pipeline.load()
    logger.info("inference_pipeline_ready")
    return pipeline


def to_pipeline_input(text: str | None, image_b64: str | None) -> MultimodalInput:
    image: Any | None = None
    if image_b64 is not None and str(image_b64).strip():
        raw = decode_b64_image(image_b64)
        if raw is not None:
            image = _bytes_to_pil_rgb(raw)
    return MultimodalInput(image=image, clinical_text=text)
