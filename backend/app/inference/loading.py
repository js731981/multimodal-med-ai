from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ImportPath:
    module: str
    attr: str

    @classmethod
    def parse(cls, value: str) -> "ImportPath":
        if ":" not in value:
            raise ValueError("Import path must be in the form 'module.submodule:attribute'")
        module, attr = value.split(":", 1)
        module = module.strip()
        attr = attr.strip()
        if not module or not attr:
            raise ValueError("Import path must be in the form 'module.submodule:attribute'")
        return cls(module=module, attr=attr)


def import_object(path: str) -> Any:
    spec = ImportPath.parse(path)
    mod = importlib.import_module(spec.module)
    try:
        return getattr(mod, spec.attr)
    except AttributeError as e:
        raise ImportError(f"Attribute '{spec.attr}' not found in module '{spec.module}'") from e


def build_from_import_path(path: str, *, kwargs: dict[str, Any] | None = None) -> Any:
    obj = import_object(path)
    if callable(obj):
        return obj(**(kwargs or {}))
    return obj

