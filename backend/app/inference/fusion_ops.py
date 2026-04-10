"""Reusable multimodal embedding alignment and fusion (concat + optional future strategies)."""

from __future__ import annotations

import torch
from torch import Tensor


def _as_batch_row(x: Tensor) -> Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected embedding of rank 1 or 2, got shape {tuple(x.shape)}.")


def fuse_modal_embeddings(
    image_embedding: Tensor | None,
    text_embedding: Tensor | None,
    *,
    image_feature_dim: int,
    text_feature_dim: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Concatenate image and text embeddings, padding missing modalities with zeros.

    Returns a CPU float tensor of shape ``[B, image_feature_dim + text_feature_dim]``.
    """
    if image_embedding is None and text_embedding is None:
        raise ValueError("At least one of image_embedding or text_embedding must be provided.")
    if image_feature_dim < 1 or text_feature_dim < 1:
        raise ValueError("image_feature_dim and text_feature_dim must be positive.")

    dev = torch.device("cpu")

    if text_embedding is None:
        img = _as_batch_row(image_embedding).to(dev, dtype=dtype)
        z_txt = torch.zeros((img.shape[0], text_feature_dim), device=dev, dtype=dtype)
        if img.shape[-1] != image_feature_dim:
            raise ValueError(f"Expected image embedding dim {image_feature_dim}, got {img.shape[-1]}")
        return torch.cat([img, z_txt], dim=-1)

    if image_embedding is None:
        txt = _as_batch_row(text_embedding).to(dev, dtype=dtype)
        z_img = torch.zeros((txt.shape[0], image_feature_dim), device=dev, dtype=dtype)
        if txt.shape[-1] != text_feature_dim:
            raise ValueError(f"Expected text embedding dim {text_feature_dim}, got {txt.shape[-1]}")
        return torch.cat([z_img, txt], dim=-1)

    img = _as_batch_row(image_embedding).to(dev, dtype=dtype)
    txt = _as_batch_row(text_embedding).to(dev, dtype=dtype)
    if img.shape[0] != txt.shape[0]:
        raise ValueError("Batch size mismatch between image and text embeddings.")
    if img.shape[-1] != image_feature_dim:
        raise ValueError(f"Expected image embedding dim {image_feature_dim}, got {img.shape[-1]}")
    if txt.shape[-1] != text_feature_dim:
        raise ValueError(f"Expected text embedding dim {text_feature_dim}, got {txt.shape[-1]}")
    return torch.cat([img, txt], dim=-1)
