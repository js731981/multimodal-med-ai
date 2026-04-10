"""Chest X-ray image dataset from folder layout: root/class_name/*.ext."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import Subset
from torchvision.datasets import ImageFolder


def build_imagefolder(
    data_root: Path | str,
    transform: Callable | None = None,
) -> ImageFolder:
    """Load images with torchvision ``ImageFolder`` (one folder per class)."""
    root = Path(data_root).resolve()
    return ImageFolder(str(root), transform=transform)


def stratified_indices(
    targets: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Train/val index split with equal ``val_fraction`` per class."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    from collections import defaultdict
    import random

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, t in enumerate(targets):
        by_class[t].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls, idxs in sorted(by_class.items()):
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(n * val_fraction))
            n_val = max(1, n_val)
            n_val = min(n_val, n - 1)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    if not train_idx:
        raise ValueError(
            "Train split is empty; reduce val_fraction or add more images per class."
        )
    return train_idx, val_idx


def train_val_subsets(
    dataset: ImageFolder,
    val_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Stratified ``Subset`` for train and validation."""
    targets = dataset.targets
    train_i, val_i = stratified_indices(targets, val_fraction, seed)
    return Subset(dataset, train_i), Subset(dataset, val_i)
