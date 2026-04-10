"""
Multimodal pipeline integration test: processed CXR + clinical text.

Loads the first image under ``data/processed`` (recursive). If none exists, writes a
minimal PNG under ``data/processed/normal/`` so CI and fresh clones can run without
manual steps. Override with ``MMEDAI_TEST_PROCESSED_IMAGE_PATH``.

Uses ``models/image_model/dummy_smoke.pth`` for the ResNet head (created on first run
if absent). Run from repo root:

  python -m pytest tests/test_multimodal_pipeline.py -s
"""

from __future__ import annotations

import os
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

from backend.app.config import get_settings  # noqa: E402
from backend.app.inference.fusion_ops import fuse_modal_embeddings  # noqa: E402
from backend.app.inference.image_model_impl import ImageModelImpl  # noqa: E402
from backend.app.inference.loading import build_from_import_path  # noqa: E402
from backend.app.inference.pipeline import MultimodalInferencePipeline  # noqa: E402
from backend.app.inference.schemas import MultimodalInput  # noqa: E402

_PROCESSED_ROOT = _PROJECT_ROOT / "data" / "processed"
_SAMPLE_TEXT = "fever, cough, chest pain"
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

_SMOKE_CKPT_REL = Path("models") / "image_model" / "dummy_smoke.pth"
_FALLBACK_SAMPLE_REL = Path("data") / "processed" / "normal" / "pytest_smoke_sample.png"


def _smoke_checkpoint_path() -> Path:
    return (_PROJECT_ROOT / _SMOKE_CKPT_REL).resolve()


def _ensure_smoke_checkpoint() -> Path:
    path = _smoke_checkpoint_path()
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    payload = {
        "state_dict": model.state_dict(),
        "classes": ["normal", "pneumonia"],
    }
    torch.save(payload, path)
    return path


def _iter_images_under_processed() -> list[Path]:
    if not _PROCESSED_ROOT.is_dir():
        return []
    found: list[Path] = []
    for p in _PROCESSED_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            found.append(p)
    return sorted(found)


def _write_minimal_processed_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (64, 64), color=(120, 140, 160))
    im.save(path, format="PNG")


def _resolve_processed_sample_image() -> Path:
    env = os.environ.get("MMEDAI_TEST_PROCESSED_IMAGE_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"MMEDAI_TEST_PROCESSED_IMAGE_PATH is not a file: {p.resolve() if p.exists() else p}"
            )
        return p.resolve()

    for candidate in _iter_images_under_processed():
        return candidate.resolve()

    fallback = (_PROJECT_ROOT / _FALLBACK_SAMPLE_REL).resolve()
    try:
        _write_minimal_processed_png(fallback)
    except OSError as e:
        raise FileNotFoundError(
            f"No image under {_PROCESSED_ROOT} and could not write fallback sample at {fallback}: {e}. "
            "Create data/processed/**.png (e.g. run: python scripts/generate_dummy_cxr_dataset.py) "
            "or set MMEDAI_TEST_PROCESSED_IMAGE_PATH to an existing image file."
        ) from e

    if not fallback.is_file():
        raise FileNotFoundError(
            f"No image under {_PROCESSED_ROOT}. Expected at least one file with extension "
            f"{sorted(_IMAGE_EXTS)}. Run: python scripts/generate_dummy_cxr_dataset.py "
            "or set MMEDAI_TEST_PROCESSED_IMAGE_PATH."
        )
    return fallback


def _load_processed_sample_pil() -> tuple[Image.Image, Path]:
    path = _resolve_processed_sample_image()
    with Image.open(path) as im:
        return im.convert("RGB"), path


def _build_pipeline(*, checkpoint_path: Path) -> MultimodalInferencePipeline:
    s = get_settings()
    image_kwargs = {**dict(s.image_model_kwargs), "checkpoint_path": str(checkpoint_path)}
    image_model = build_from_import_path(s.image_model_path, kwargs=image_kwargs)
    text_model = build_from_import_path(s.text_model_path, kwargs=s.text_model_kwargs)
    fusion_kw = dict(s.fusion_model_kwargs)
    if "labels" not in fusion_kw:
        fusion_kw["labels"] = s.labels
    fusion_model = build_from_import_path(s.fusion_model_path, kwargs=fusion_kw)
    if text_model is not None and not hasattr(text_model, "encode") and hasattr(text_model, "predict"):
        text_model.encode = text_model.predict  # type: ignore[attr-defined]
    return MultimodalInferencePipeline(
        image_model=image_model,
        text_model=text_model,
        fusion_model=fusion_model,
        labels=s.labels,
    )


def _image_scores_from_probs(
    pipeline: MultimodalInferencePipeline, img_probs: torch.Tensor
) -> dict[str, float] | dict[str, list]:
    classes = getattr(pipeline.image_model, "_classes", None)
    if classes:
        flat = img_probs[0] if img_probs.ndim == 2 else img_probs
        return {str(classes[i]): float(flat[i].item()) for i in range(len(classes))}
    return {"raw_tensor": img_probs.detach().cpu().tolist()}


def test_multimodal_pipeline_processed_image_and_text() -> None:
    ImageModelImpl.reset_singleton_for_testing()

    ckpt = _ensure_smoke_checkpoint()
    assert ckpt.is_file(), f"Smoke checkpoint must exist after ensure: {ckpt}"

    image, image_path = _load_processed_sample_pil()
    pipeline = _build_pipeline(checkpoint_path=ckpt)
    pipeline.load()
    assert pipeline.image_model is not None
    assert pipeline.text_model is not None

    img_emb = pipeline.image_model.encode(image)
    txt_emb = pipeline.text_model.encode(_SAMPLE_TEXT)
    assert isinstance(img_emb, torch.Tensor)
    assert isinstance(txt_emb, torch.Tensor)
    assert img_emb.numel() > 0 and txt_emb.numel() > 0

    fm = pipeline.fusion_model
    idim = int(getattr(fm, "image_feature_dim"))
    tdim = int(getattr(fm, "text_feature_dim"))
    fused = fuse_modal_embeddings(
        img_emb,
        txt_emb,
        image_feature_dim=idim,
        text_feature_dim=tdim,
    )

    _, img_probs = pipeline.image_model.predict(image)
    image_scores = _image_scores_from_probs(pipeline, img_probs)

    final = pipeline.run(MultimodalInput(image=image, clinical_text=_SAMPLE_TEXT))

    assert final.disease in ("normal", "pneumonia")
    assert 0.0 <= final.confidence <= 1.0
    assert isinstance(final.explanation, str)
    assert final.scores is not None
    assert set(final.scores) == {"normal", "pneumonia"}

    print()
    print("=== Sample path ===")
    print(image_path)
    print()
    print("=== Image prediction (ResNet head) ===")
    print(image_scores)
    print()
    print("=== Embedding shapes ===")
    print(f"image embedding shape:  {tuple(img_emb.shape)}")
    print(f"text embedding shape:   {tuple(txt_emb.shape)}")
    print(f"fused embedding shape: {tuple(fused.shape)}")
    print()
    print("=== Final multimodal output ===")
    print(f"disease:     {final.disease}")
    print(f"confidence:  {final.confidence:.6f}")
    print(f"scores:      {dict(final.scores)}")
    print()
