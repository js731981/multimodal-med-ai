from __future__ import annotations

from typing import Any

import torch

from backend.app.models.interfaces import FusionModel, ModelInfo
from backend.app.models.pytorch_placeholders import TorchFusionModelPlaceholder


class TorchFusionClassifier:
    """
    Adapter around :class:`~backend.app.models.pytorch_placeholders.TorchFusionModelPlaceholder`.

    Exposes ``predict(fused_embedding)`` for the multimodal pipeline (concatenation done upstream).
    """

    def __init__(
        self,
        fusion: FusionModel | None = None,
        *,
        image_dim: int | None = None,
        image_feature_dim: int | None = None,
        text_dim: int | None = None,
        text_feature_dim: int | None = None,
        fused_dim: int = 256,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        weights_path: str | None = None,
        **extra: Any,
    ) -> None:
        _ = extra
        idim = image_feature_dim if image_feature_dim is not None else (image_dim if image_dim is not None else 2048)
        tdim = text_feature_dim if text_feature_dim is not None else (text_dim if text_dim is not None else 256)
        self.fusion: FusionModel = fusion or TorchFusionModelPlaceholder(
            image_dim=int(idim),
            text_dim=int(tdim),
            fused_dim=int(fused_dim),
            num_classes=int(num_classes),
            device=device,
            weights_path=weights_path,
        )
        self.info = ModelInfo(name="torch-fusion-classifier-adapter", version="0.2.0")

    @property
    def image_feature_dim(self) -> int:
        return int(self.fusion.image_dim)

    @property
    def text_feature_dim(self) -> int:
        return int(self.fusion.text_dim)

    def load_model(self) -> None:
        self.fusion.load_model()

    @torch.inference_mode()
    def predict(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        self.fusion.load_model()
        pfn = getattr(self.fusion, "predict_from_fused", None)
        if not callable(pfn):
            raise TypeError("Wrapped fusion model must implement predict_from_fused(fused_embedding).")
        return pfn(fused_embedding)
