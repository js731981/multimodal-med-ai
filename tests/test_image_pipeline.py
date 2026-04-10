"""
Integration smoke test for the real ResNet chest X-ray model + multimodal fusion.

Run from repository root (so ``backend`` is importable):

  python -m pytest tests/test_image_pipeline.py -s

Or without pytest:

  python tests/test_image_pipeline.py

Sample image resolution order:
1. ``MMEDAI_TEST_CXR_PATH`` — path to any PNG/JPEG chest X-ray
2. ``tests/fixtures/sample_cxr.png`` if present
3. Otherwise a small synthetic grayscale image (pipeline wiring only)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

# Repo root: multimodal-med-ai/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.inference.image_model_impl import (  # noqa: E402
    ImageModelImpl,
    resolve_image_checkpoint_path,
)
from backend.app.inference.loading import build_from_import_path  # noqa: E402
from backend.app.inference.pipeline import MultimodalInferencePipeline  # noqa: E402
from backend.app.inference.schemas import MultimodalInput  # noqa: E402


_FIXTURE_CXR = Path(__file__).resolve().parent / "fixtures" / "sample_cxr.png"
_DUMMY_CLINICAL_TEXT = "Patient reports mild cough; no fever."


def _image_checkpoint_available() -> bool:
    try:
        resolve_image_checkpoint_path(None)
        return True
    except FileNotFoundError:
        return False


def _synthetic_chest_xray_pil():
    """Deterministic pseudo-CXR (RGB) for environments without a real file."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
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


def _load_sample_cxr():
    """Return a PIL RGB image for inference."""
    from PIL import Image

    env_path = os.environ.get("MMEDAI_TEST_CXR_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_file():
            raise FileNotFoundError(f"MMEDAI_TEST_CXR_PATH is not a file: {p}")
        with Image.open(p) as im:
            return im.convert("RGB")
    if _FIXTURE_CXR.is_file():
        with Image.open(_FIXTURE_CXR) as im:
            return im.convert("RGB")
    return _synthetic_chest_xray_pil()


def _build_pipeline() -> MultimodalInferencePipeline:
    s = get_settings()
    image_model = build_from_import_path(s.image_model_path, kwargs=s.image_model_kwargs)
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


def _run_image_pipeline_integration() -> None:
    ImageModelImpl.reset_singleton_for_testing()

    ckpt = resolve_image_checkpoint_path(None)
    assert ckpt.is_file(), f"Image model checkpoint not found (expected at {ckpt})"

    image = _load_sample_cxr()

    pipeline = _build_pipeline()
    t0 = time.perf_counter()
    pipeline.load()
    load_ms = (time.perf_counter() - t0) * 1000.0

    assert pipeline.image_model is not None

    t_img0 = time.perf_counter()
    img_emb, img_probs = pipeline.image_model.predict(image)
    img_predict_ms = (time.perf_counter() - t_img0) * 1000.0

    assert img_emb.ndim == 2 and img_emb.shape[0] == 1
    assert img_probs.ndim == 2 and img_probs.shape[0] == 1

    classes = getattr(pipeline.image_model, "_classes", None)
    if classes:
        flat = img_probs[0] if img_probs.ndim == 2 else img_probs
        image_scores = {str(classes[i]): float(flat[i].item()) for i in range(len(classes))}
    else:
        image_scores = {"raw_tensor": img_probs.detach().cpu().tolist()}

    t_run0 = time.perf_counter()
    fused = pipeline.run(
        MultimodalInput(image=image, clinical_text=_DUMMY_CLINICAL_TEXT),
    )
    run_ms = (time.perf_counter() - t_run0) * 1000.0

    print()
    print("=== Latency ===")
    print(f"pipeline.load():           {load_ms:8.2f} ms")
    print(f"image_model.predict():     {img_predict_ms:8.2f} ms")
    print(f"pipeline.run() (fusion): {run_ms:8.2f} ms")
    print()
    print("=== Image prediction (ResNet head, checkpoint classes) ===")
    print(image_scores)
    print()
    assert fused.disease in ("normal", "pneumonia")
    assert isinstance(fused.explanation, str)
    assert fused.scores is not None
    assert set(fused.scores) == {"normal", "pneumonia"}

    print("=== Final fused multimodal output ===")
    print(f"disease:     {fused.disease}")
    print(f"confidence:  {fused.confidence:.6f}")
    print(f"scores:      {dict(fused.scores) if fused.scores else None}")
    print(f"explanation: {fused.explanation[:200] + '...' if len(fused.explanation) > 200 else fused.explanation}")
    print()


@pytest.mark.skipif(not _image_checkpoint_available(), reason="image checkpoint missing")
def test_image_pipeline_integration() -> None:
    _run_image_pipeline_integration()


def main() -> int:
    try:
        resolve_image_checkpoint_path(None)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    _run_image_pipeline_integration()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
