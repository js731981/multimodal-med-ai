from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch
from PIL import UnidentifiedImageError

from backend.app.inference.errors import InvalidImageInputError
from backend.app.inference.explanation import MultimodalExplanationProvider
from backend.app.inference.fusion_ops import fuse_modal_embeddings
from backend.app.inference.schemas import MultimodalInput, MultimodalPrediction


logger = logging.getLogger(__name__)


Tensor = torch.Tensor


class ImageEmbeddingModel(Protocol):
    def load_model(self) -> None: ...

    def encode(self, image: Any) -> Tensor: ...

    @property
    def info(self) -> Any: ...


class TextEmbeddingModel(Protocol):
    def load_model(self) -> None: ...

    def encode(self, text: str | Sequence[str]) -> Tensor: ...

    @property
    def info(self) -> Any: ...


class FusionClassifier(Protocol):
    """Consumes a **fused** embedding (e.g. concatenated image + text), shape ``[B, D]``."""

    def load_model(self) -> None: ...

    def predict(self, fused_embedding: Tensor) -> Tensor: ...

    @property
    def info(self) -> Any: ...


@dataclass(slots=True)
class MultimodalInferencePipeline:
    """
    Modular pipeline:
    1) image model -> embedding
    2) text model -> embedding
    3) fusion model -> disease + confidence
    4) optional explanation provider (e.g. RAG) using disease + clinical text
    """

    image_model: ImageEmbeddingModel | None
    text_model: TextEmbeddingModel | None
    fusion_model: FusionClassifier
    labels: Sequence[str]
    explanation_provider: MultimodalExplanationProvider | None = None

    def load(self) -> None:
        t0 = time.perf_counter()
        if self.image_model is not None:
            self.image_model.load_model()
        if self.text_model is not None:
            self.text_model.load_model()
        self.fusion_model.load_model()
        logger.info(
            "pipeline_loaded",
            extra={
                "image_model": getattr(getattr(self.image_model, "info", None), "name", None)
                if self.image_model is not None
                else None,
                "text_model": getattr(getattr(self.text_model, "info", None), "name", None)
                if self.text_model is not None
                else None,
                "fusion_model": getattr(getattr(self.fusion_model, "info", None), "name", None),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        )

    def run(self, inp: MultimodalInput) -> MultimodalPrediction:
        if not inp.image and not (inp.clinical_text and inp.clinical_text.strip()):
            raise ValueError("At least one modality must be provided (image and/or clinical_text).")

        t0 = time.perf_counter()

        logger.info(
            "pipeline_input_received",
            extra={
                "has_image": inp.image is not None,
                "text_len": len(inp.clinical_text) if inp.clinical_text else 0,
            },
        )

        img_emb: Tensor | None = None
        txt_emb: Tensor | None = None

        if inp.image is not None:
            if self.image_model is None:
                raise ValueError("Image provided but image_model is not configured.")
            t_img0 = time.perf_counter()
            try:
                img_emb = self.image_model.encode(inp.image)
            except FileNotFoundError:
                raise
            except (OSError, UnidentifiedImageError, TypeError, ValueError) as e:
                logger.warning("image_encode_failed", extra={"reason": str(e)})
                raise InvalidImageInputError(str(e)) from e
            logger.info(
                "image_embedding_done",
                extra={
                    "elapsed_ms": round((time.perf_counter() - t_img0) * 1000, 2),
                    "image_embedding_shape": list(img_emb.shape),
                },
            )

        if inp.clinical_text is not None and inp.clinical_text.strip():
            if self.text_model is None:
                raise ValueError("Clinical text provided but text_model is not configured.")
            t_txt0 = time.perf_counter()
            txt_emb = self.text_model.encode(inp.clinical_text)
            logger.info(
                "text_embedding_done",
                extra={
                    "elapsed_ms": round((time.perf_counter() - t_txt0) * 1000, 2),
                    "text_embedding_shape": list(txt_emb.shape),
                },
            )

        t_fus0 = time.perf_counter()
        img_dim, txt_dim = self._fusion_embedding_dims()
        fused = fuse_modal_embeddings(
            img_emb,
            txt_emb,
            image_feature_dim=img_dim,
            text_feature_dim=txt_dim,
        )
        logger.info(
            "multimodal_fused",
            extra={"fused_embedding_shape": list(fused.shape)},
        )
        probs = self.fusion_model.predict(fused)
        logger.info(
            "fusion_predict_done",
            extra={"elapsed_ms": round((time.perf_counter() - t_fus0) * 1000, 2)},
        )

        probs = self._as_1d_probs(probs)
        if len(self.labels) != probs.numel():
            raise ValueError(f"labels length ({len(self.labels)}) must match num_classes ({probs.numel()})")

        fusion_out = self._fusion_output_dict(probs)
        disease = str(fusion_out["disease"])
        confidence = float(fusion_out["confidence"])
        scores = self._scores_dict(probs, self.labels)

        logger.info(
            "fusion_output",
            extra={"fusion_output": dict(fusion_out)},
        )
        logger.info(
            "pipeline_prediction",
            extra={
                "disease": disease,
                "confidence": confidence,
                "scores": dict(scores),
            },
        )
        symptoms_for_rag = (
            inp.clinical_text.strip()
            if inp.clinical_text and inp.clinical_text.strip()
            else None
        )
        explanation = ""
        if self.explanation_provider is not None:
            t_rag0 = time.perf_counter()
            explanation = self.explanation_provider.explain(
                disease=disease,
                symptoms=symptoms_for_rag,
            )
            logger.info(
                "pipeline_rag_done",
                extra={
                    "elapsed_ms": round((time.perf_counter() - t_rag0) * 1000, 2),
                    "explanation_len": len(explanation),
                },
            )

        logger.info(
            "pipeline_infer_done",
            extra={
                "prediction": disease,
                "confidence": confidence,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        )

        return MultimodalPrediction(
            disease=disease,
            confidence=confidence,
            explanation=explanation,
            scores=scores,
        )

    def _fusion_embedding_dims(self) -> tuple[int, int]:
        fm = self.fusion_model
        idim = getattr(fm, "image_feature_dim", None)
        tdim = getattr(fm, "text_feature_dim", None)
        if idim is not None and tdim is not None:
            return int(idim), int(tdim)
        raise ValueError(
            "fusion_model must expose integer attributes image_feature_dim and text_feature_dim "
            "so the pipeline can build fused embeddings."
        )

    def _fusion_output_dict(self, probs_1d: Tensor) -> dict[str, str | float]:
        decoder = getattr(self.fusion_model, "to_fusion_output", None)
        if callable(decoder):
            raw = decoder(probs_1d)
            if isinstance(raw, dict) and "disease" in raw and "confidence" in raw:
                return {"disease": str(raw["disease"]), "confidence": float(raw["confidence"])}
        best_idx = int(torch.argmax(probs_1d).item())
        return {
            "disease": str(self.labels[best_idx]),
            "confidence": float(probs_1d[best_idx].item()),
        }

    @staticmethod
    def _as_1d_probs(x: Tensor) -> Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("fusion_model.predict must return a torch.Tensor")
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("Expected batch size 1 for fusion output")
            x = x[0]
        if x.ndim != 1:
            raise ValueError("Expected fusion output probabilities as 1D tensor [num_classes] (or [1,num_classes])")
        s = float(x.sum().item())
        if not (0.999 <= s <= 1.001):
            x = torch.softmax(x, dim=-1)
        return x.to(dtype=torch.float32, device="cpu")

    @staticmethod
    def _scores_dict(probs: Tensor, labels: Sequence[str]) -> Mapping[str, float]:
        return {str(labels[i]): float(probs[i].item()) for i in range(probs.numel())}

