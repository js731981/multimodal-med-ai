from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MultimodalInput:
    image: Any | None
    clinical_text: str | None


@dataclass(frozen=True, slots=True)
class MultimodalPrediction:
    disease: str
    confidence: float
    explanation: str = ""
    scores: Mapping[str, float] | None = None

