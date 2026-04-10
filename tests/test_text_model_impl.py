"""Unit tests for TF-IDF :class:`TextModelImpl`."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.inference.text_model_impl import TextModelImpl  # noqa: E402


def test_text_model_impl_single_string_fixed_shape() -> None:
    m = TextModelImpl(embedding_dim=128)
    m.load_model()
    out = m.encode("fever and productive cough for three days")
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 128)
    assert out.dtype == torch.float32


def test_text_model_impl_batch() -> None:
    m = TextModelImpl(embedding_dim=64)
    m.load_model()
    out = m.predict(["fever", "chest pain on exertion"])
    assert out.shape == (2, 64)


def test_text_model_impl_empty_string_zeros() -> None:
    m = TextModelImpl(embedding_dim=32)
    m.load_model()
    out = m.encode("   ")
    assert torch.allclose(out, torch.zeros(1, 32))
