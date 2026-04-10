"""Grad-CAM for ResNet-style classifiers (last conv block)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _jet_rgb(x: np.ndarray) -> np.ndarray:
    """Map float [H, W] in ~[0, 1] to uint8 RGB (jet-like, numpy-only)."""
    t = np.clip(x.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def compute_gradcam_2d(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    class_idx: int,
    *,
    target_layer: torch.nn.Module | None = None,
) -> np.ndarray:
    """
    Return a 2D saliency map [H, W] float32 in [0, 1] at conv spatial resolution.

    ``input_batch`` must be shape [1, 3, H, W] on the same device as ``model``,
    with ``requires_grad=True`` so gradients reach the target layer.
    """
    if target_layer is None:
        target_layer = model.layer4[-1]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _fwd(_m: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        activations.append(out)

    def _bwd(_m: torch.nn.Module, _gi: tuple, go: tuple[torch.Tensor, ...]) -> None:
        gradients.append(go[0])

    h1 = target_layer.register_forward_hook(_fwd)
    h2 = target_layer.register_full_backward_hook(_bwd)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_batch)
        score = logits[0, class_idx]
        score.backward()
    finally:
        h1.remove()
        h2.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

    a = activations[0][0]
    g = gradients[0][0]
    weights = g.mean(dim=(1, 2), keepdim=True)
    cam = (weights * a).sum(dim=0)
    cam = F.relu(cam)
    cam_np = cam.detach().float().cpu().numpy()
    c_min, c_max = float(cam_np.min()), float(cam_np.max())
    if c_max - c_min < 1e-8:
        return np.zeros_like(cam_np, dtype=np.float32)
    cam_np = (cam_np - c_min) / (c_max - c_min)
    return cam_np.astype(np.float32)


class GradCAM:
    """Grad-CAM over the last ResNet conv block (or a custom ``target_layer``)."""

    def __init__(self, model: torch.nn.Module, *, target_layer: torch.nn.Module | None = None) -> None:
        self._model = model
        self._target_layer = target_layer

    def generate(self, input_batch: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Compute a 2D saliency map for ``class_idx``.

        ``input_batch`` must be shape ``[1, 3, H, W]`` on the model device with
        ``requires_grad=True`` so gradients reach the target layer.
        """
        return compute_gradcam_2d(
            self._model, input_batch, class_idx, target_layer=self._target_layer
        )


def overlay_gradcam_on_pil(
    pil_rgb: Image.Image,
    cam_hw: np.ndarray,
    *,
    alpha: float = 0.45,
) -> Image.Image:
    """Resize CAM to original image size, colorize, and alpha-blend over RGB."""
    orig = pil_rgb.convert("RGB")
    w, h = orig.size
    cam_u8 = (np.clip(cam_hw, 0.0, 1.0) * 255.0).astype(np.uint8)
    cam_img = Image.fromarray(cam_u8, mode="L").resize((w, h), Image.BILINEAR)
    cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat_rgb = _jet_rgb(cam_arr)
    base = np.asarray(orig, dtype=np.float32)
    out = alpha * heat_rgb + (1.0 - alpha) * base
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")
