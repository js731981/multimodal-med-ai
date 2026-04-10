"""Inference-time MLP fusion: concatenated embeddings → class probabilities + structured output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from backend.app.inference.fusion_ops import fuse_modal_embeddings
from backend.app.models.fusion_model_impl import FusionMLPModule
from backend.app.models.interfaces import ModelInfo


def fusion_prediction_dict(probs: Tensor, labels: Sequence[str]) -> dict[str, Any]:
    """Map a probability vector to ``{"disease": str, "confidence": float}``."""
    if not isinstance(probs, torch.Tensor):
        raise TypeError("probs must be a torch.Tensor")
    p = probs.detach().float().cpu().flatten()
    if p.numel() != len(labels):
        raise ValueError(f"probs length ({p.numel()}) must match len(labels) ({len(labels)}).")
    idx = int(torch.argmax(p).item())
    return {"disease": str(labels[idx]), "confidence": float(p[idx].item())}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class FusionModelImpl:
    """
    Production MLP fusion on **pre-concatenated** multimodal vectors (built by the pipeline).

    The multimodal pipeline is responsible for: image encode → text encode →
    :func:`~backend.app.inference.fusion_ops.fuse_modal_embeddings` → :meth:`predict`.

    :meth:`predict` accepts either a fused tensor ``[B, D]`` or ``(image_embedding, text_embedding)``
    for convenience; the canonical path is fused-only to keep fusion strategy explicit in the pipeline.

    Exposes ``image_feature_dim`` / ``text_feature_dim`` for alignment with upstream encoders.

    The final linear layer has ``len(labels)`` outputs; ``labels[i]`` names softmax dimension ``i``.
    For the default CXR setup, use ``["normal", "pneumonia"]`` in the same order as
    :class:`~backend.app.inference.image_model_impl.ImageModelImpl` checkpoint ``classes``.
    """

    def __init__(
        self,
        labels: Sequence[str],
        *,
        image_feature_dim: int | None = None,
        text_feature_dim: int | None = None,
        image_dim: int | None = None,
        text_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        **extra: Any,
    ) -> None:
        _ = extra
        idim = image_feature_dim if image_feature_dim is not None else image_dim
        tdim = text_feature_dim if text_feature_dim is not None else text_dim
        if idim is None:
            idim = 2048
        if tdim is None:
            tdim = 256
        self.labels = [str(x) for x in labels]
        if len(self.labels) < 1:
            raise ValueError("labels must be non-empty.")
        self.image_feature_dim = int(idim)
        self.text_feature_dim = int(tdim)
        if self.image_feature_dim < 1 or self.text_feature_dim < 1:
            raise ValueError("image_feature_dim and text_feature_dim must be positive.")
        self._hidden_dim = hidden_dim
        self._dropout = float(dropout)
        self._weights_path = Path(weights_path) if weights_path else None
        self._device_pref = device
        self._device: torch.device | None = None
        self._net: FusionMLPModule | None = None
        fused_dim = self.image_feature_dim + self.text_feature_dim
        self.info = ModelInfo(
            name="fusion-mlp",
            version="1.0.0",
            extra={
                "fused_input_dim": fused_dim,
                "num_classes": len(self.labels),
                "weights_path": str(self._weights_path) if self._weights_path else None,
            },
        )

    def load_model(self) -> None:
        if self._net is not None:
            return
        self._device = _resolve_device(self._device_pref)
        fused_dim = self.image_feature_dim + self.text_feature_dim
        net = FusionMLPModule(
            fused_input_dim=fused_dim,
            num_classes=len(self.labels),
            hidden_dim=self._hidden_dim,
            dropout=self._dropout,
        )
        if self._weights_path is not None and self._weights_path.is_file():
            state = torch.load(str(self._weights_path), map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            net.load_state_dict(state, strict=True)
        net = net.to(self._device).eval()
        self._net = net

    def _ensure_loaded(self) -> None:
        if self._net is None:
            self.load_model()

    def build_fused_embedding(
        self,
        image_embedding: Tensor | None,
        text_embedding: Tensor | None,
    ) -> Tensor:
        """Concatenate / pad modalities (CPU float32). Used by the pipeline for a clear fusion step."""
        return fuse_modal_embeddings(
            image_embedding,
            text_embedding,
            image_feature_dim=self.image_feature_dim,
            text_feature_dim=self.text_feature_dim,
        )

    @torch.inference_mode()
    def predict(self, fused_embedding: Tensor) -> Tensor:
        """
        Run the MLP on fused embeddings. Returns probabilities ``[B, num_classes]`` (float32, CPU).
        """
        self._ensure_loaded()
        assert self._net is not None and self._device is not None
        x = fused_embedding.to(self._device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        out = self._net(x)
        return out.detach().to(dtype=torch.float32, device="cpu")

    def predict_from_modalities(
        self,
        image_embedding: Tensor | None,
        text_embedding: Tensor | None,
    ) -> Tensor:
        """Fuse then classify (optional helper; pipeline normally calls :meth:`build_fused_embedding` then :meth:`predict`)."""
        fused = self.build_fused_embedding(image_embedding, text_embedding)
        return self.predict(fused)

    def to_fusion_output(self, probs: Tensor) -> dict[str, Any]:
        """Structured fusion output ``{"disease": str, "confidence": float}`` from model probabilities."""
        if probs.ndim == 2:
            if probs.shape[0] != 1:
                raise ValueError("Expected batch size 1 for fusion output dict.")
            probs = probs[0]
        return fusion_prediction_dict(probs, self.labels)
