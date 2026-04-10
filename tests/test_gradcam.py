"""
Grad-CAM integration test: load smoke ResNet checkpoint on CPU, predict, save overlay.

Run from repository root:

  python -m pytest tests/test_gradcam.py -s

Uses ``models/image_model/dummy_smoke.pth`` (created if absent) and an image under
``data/processed/`` (first match by sorted path; a minimal PNG is written if the tree is empty).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.inference.image_model_impl import ImageModelImpl  # noqa: E402

_SMOKE_CKPT = _PROJECT_ROOT / "models" / "image_model" / "dummy_smoke.pth"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def _synthetic_rgb_pil(*, seed: int = 42) -> Image.Image:
    import numpy as np

    rng = np.random.default_rng(seed)
    h, w = 512, 512
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    base = 1.0 - np.clip(r / (min(h, w) * 0.45), 0, 1)
    noise = rng.normal(0, 0.04, (h, w))
    g = np.clip(base * 0.7 + noise, 0, 1)
    arr = (g * 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _ensure_dummy_smoke_checkpoint() -> Path:
    """
    Ensure ``dummy_smoke.pth`` exists: a valid ResNet50 two-class head matching ``ImageModelImpl``.
    """
    _SMOKE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if _SMOKE_CKPT.is_file():
        return _SMOKE_CKPT
    torch.manual_seed(0)
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    payload = {
        "state_dict": model.state_dict(),
        "classes": ["normal", "pneumonia"],
    }
    try:
        torch.save(payload, _SMOKE_CKPT)
    except OSError as e:
        raise RuntimeError(f"Could not write smoke checkpoint to {_SMOKE_CKPT}: {e}") from e
    if not _SMOKE_CKPT.is_file():
        raise RuntimeError(f"Smoke checkpoint was not created: {_SMOKE_CKPT}")
    return _SMOKE_CKPT


def _iter_processed_images() -> list[Path]:
    if not _PROCESSED_DIR.is_dir():
        return []
    found: list[Path] = []
    for p in _PROCESSED_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            found.append(p)
    return sorted(found)


def _ensure_sample_under_processed() -> None:
    """If ``data/processed`` has no raster images, add a minimal PNG for CI and fresh clones."""
    if _iter_processed_images():
        return
    try:
        _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        normal_dir = _PROCESSED_DIR / "normal"
        normal_dir.mkdir(parents=True, exist_ok=True)
        out = normal_dir / "gradcam_smoke_sample.png"
        _synthetic_rgb_pil().save(out, format="PNG")
    except OSError as e:
        raise RuntimeError(
            f"Could not create sample image under {_PROCESSED_DIR}: {e}. "
            "Create data/processed/**.png or run: python scripts/generate_dummy_cxr_dataset.py"
        ) from e


def _pick_processed_image_path() -> Path:
    _ensure_sample_under_processed()
    paths = _iter_processed_images()
    if not paths:
        raise FileNotFoundError(
            f"No image files (suffixes: {', '.join(sorted(_IMAGE_SUFFIXES))}) under {_PROCESSED_DIR}. "
            "Run: python scripts/generate_dummy_cxr_dataset.py"
        )
    return paths[0]


def test_gradcam_prediction_and_save(tmp_path: Path) -> None:
    ImageModelImpl.reset_singleton_for_testing()

    ckpt = _ensure_dummy_smoke_checkpoint()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Image model checkpoint not found: {ckpt}")

    image_path = _pick_processed_image_path()
    if not image_path.is_file():
        raise FileNotFoundError(f"Selected processed image is not a file: {image_path}")

    model = ImageModelImpl(checkpoint_path=ckpt, device="cpu")
    model.load_model()

    with Image.open(image_path) as im:
        image = im.convert("RGB")

    _, probs = model.predict(image)
    pred_idx = int(probs[0].argmax().item())
    pred_name = str(model._classes[pred_idx]) if model._classes else str(pred_idx)

    out_path = tmp_path / "gradcam_test.jpg"
    saved = model.save_gradcam(image, target_class=pred_idx, output_path=out_path)

    assert saved.resolve() == out_path.resolve()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0

    with Image.open(out_path) as heat_im:
        heat_im.load()
        assert heat_im.format == "JPEG"
        assert heat_im.size[0] > 0 and heat_im.size[1] > 0

    heatmap_generated = True

    print()
    print(f"predicted class: {pred_name}")
    print(f"heatmap generated: {heatmap_generated}")
    print(f"saved file path: {out_path.resolve()}")
    print(f"input image: {image_path.resolve()}")
    print()
