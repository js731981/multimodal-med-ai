"""Create a tiny synthetic ImageFolder dataset for local training pipeline smoke tests.

Layout (matches ``training.train`` default ``--data-dir``):

    data/processed/normal/*.png
    data/processed/pneumonia/*.png

``training.dataset.stratified_indices`` needs at least two images per class so the
default validation split (``val_fraction=0.2``) is non-empty. This script defaults
to eight images per class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
CLASSES = ("normal", "pneumonia")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Root directory for class subfolders (default: <repo>/data/processed)",
    )
    p.add_argument(
        "--per-class",
        type=int,
        default=8,
        metavar="N",
        help="Number of dummy PNGs per class (min 2 for default val split; default: 8)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=256,
        metavar="PX",
        help="Width and height of each synthetic image (default: 256)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible noise (default: 42)",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip importing the training dataset helpers and checking splits",
    )
    return p.parse_args()


def _base_rgb(class_name: str) -> tuple[int, int, int]:
    """Distinct base colors so classes are not identical tensors."""
    if class_name == "normal":
        return (180, 200, 220)
    if class_name == "pneumonia":
        return (200, 140, 120)
    raise ValueError(f"Unknown class: {class_name}")


def write_class_images(
    class_dir: Path,
    class_name: str,
    n: int,
    size: int,
    rng: np.random.Generator,
) -> list[Path]:
    class_dir.mkdir(parents=True, exist_ok=True)
    base = np.array(_base_rgb(class_name), dtype=np.float32)
    paths: list[Path] = []
    for i in range(n):
        noise = rng.standard_normal((size, size, 3), dtype=np.float32) * 12.0
        arr = np.clip(base + noise, 0.0, 255.0).astype(np.uint8)
        im = Image.fromarray(arr, mode="RGB")
        path = class_dir / f"dummy_{i:03d}.png"
        im.save(path, format="PNG")
        paths.append(path)
    return paths


def verify_training_splits(data_root: Path, val_fraction: float, seed: int) -> None:
    repo = REPO_ROOT
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from training.dataset import build_imagefolder, train_val_subsets

    root = data_root.resolve()
    ref = build_imagefolder(root, transform=None)
    assert len(ref.classes) >= 2, "Expected at least two classes"
    train_s, val_s = train_val_subsets(ref, val_fraction, seed)
    assert len(train_s) > 0 and len(val_s) > 0, (
        "Train or val split empty; increase --per-class or lower val_fraction"
    )


def main() -> None:
    args = _parse_args()
    if args.per_class < 2:
        raise SystemExit("--per-class must be at least 2 for default training val split")

    out = args.output_dir.resolve()
    rng = np.random.default_rng(args.seed)
    all_written: list[Path] = []
    for name in CLASSES:
        sub = out / name
        written = write_class_images(sub, name, args.per_class, args.size, rng)
        all_written.extend(written)

    print(f"Wrote {len(all_written)} images under {out}")
    for name in CLASSES:
        print(f"  {name}/: {args.per_class} files")

    if not args.no_verify:
        verify_training_splits(out, val_fraction=0.2, seed=42)
        print("Dataset loader + stratified train/val split: OK")


if __name__ == "__main__":
    main()
