"""Visualization helpers (e.g. Grad-CAM heatmap overlays)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_GRADCAM_OUTPUT = Path("outputs/gradcam_result.jpg")


def _heatmap_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    h = np.asarray(heatmap, dtype=np.float64)
    if h.ndim > 2:
        h = np.squeeze(h)
    if h.ndim != 2:
        raise ValueError(f"heatmap must be 2D (or squeezable to 2D), got shape {np.asarray(heatmap).shape}")
    lo, hi = float(h.min()), float(h.max())
    if hi > lo:
        h = (h - lo) / (hi - lo)
    else:
        h = np.zeros_like(h, dtype=np.float64)
    return (h * 255.0).round().astype(np.uint8)


def overlay_gradcam_and_save(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    output_path: str | Path = DEFAULT_GRADCAM_OUTPUT,
    blend_alpha: float = 0.5,
) -> Image.Image:
    """Resize heatmap to image size, apply JET colormap, blend with original, save as JPEG.

    Args:
        image: Original image (RGB or RGBA; converted to RGB).
        heatmap: 2D numpy array; spatial size may differ from ``image``.
        output_path: Destination path; parent directories are created if needed.
        blend_alpha: Weight of the original image in ``cv2.addWeighted`` (0–1);
            colormap weight is ``1 - blend_alpha``.

    Returns:
        Blended result as a PIL RGB image (same as what is saved).
    """
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError("blend_alpha must be in [0, 1]")

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]

    heat_u8 = _heatmap_to_uint8(heatmap)
    heat_resized = cv2.resize(heat_u8, (width, height), interpolation=cv2.INTER_LINEAR)
    jet_bgr = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)

    orig_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blended_bgr = cv2.addWeighted(
        orig_bgr,
        blend_alpha,
        jet_bgr,
        1.0 - blend_alpha,
        0.0,
    )
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(blended_rgb)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest, format="JPEG", quality=95)
    return out
