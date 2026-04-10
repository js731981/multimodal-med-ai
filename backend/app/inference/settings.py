from __future__ import annotations

"""
Backwards-compatible shim.

Inference settings are now centralized in `backend.app.config.get_settings()`.
This module remains to avoid breaking older imports.
"""

from backend.app.config import Settings, get_settings


InferenceSettings = Settings


def get_inference_settings() -> Settings:
    return get_settings()

