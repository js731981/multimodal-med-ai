from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class MedicalDocument:
    """One retrievable unit: disease key (id) and explanatory text."""

    id: str
    text: str


# Minimal in-code default matching the spec; replace or load from JSON in production.
DEFAULT_MEDICAL_DOCUMENTS: dict[str, str] = {
    "pneumonia": (
        "Pneumonia is a lung infection causing inflammation of the air sacs. "
        "On imaging, it may appear as consolidation or infiltrates. "
        "Clinical correlation with fever, cough, and dyspnea is common."
    ),
    "normal": (
        "Normal chest X-ray shows clear lungs without focal consolidation, "
        "pleural effusion, or pneumothorax. The cardiac silhouette and mediastinum "
        "are typically within expected limits for the patient's age and technique."
    ),
}


class MedicalKnowledgeBase:
    """Stores simple disease-keyed medical text snippets (JSON or dict)."""

    def __init__(self, documents: dict[str, str]) -> None:
        self._documents: dict[str, str] = dict(documents)

    @classmethod
    def from_json_path(cls, path: str | Path) -> MedicalKnowledgeBase:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Knowledge base JSON must be an object mapping keys to strings")
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("All keys and values must be strings")
        return cls(data)

    @classmethod
    def default(cls) -> MedicalKnowledgeBase:
        return cls(DEFAULT_MEDICAL_DOCUMENTS)

    def get(self, disease_key: str) -> str | None:
        return self._documents.get(disease_key)

    def items(self) -> Iterator[tuple[str, str]]:
        yield from self._documents.items()

    def as_medical_documents(self) -> list[MedicalDocument]:
        return [MedicalDocument(id=k, text=v) for k, v in self._documents.items()]

    def __len__(self) -> int:
        return len(self._documents)
