"""Run a single-image forward pass through a saved ResNet50 CXR checkpoint.

Uses the same preprocessing as validation / production inference
(``training.utils.build_val_transforms``): resize 224, ImageNet normalize, no augmentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "image_model" / "dummy_smoke.pth"
DEFAULT_NORMAL_DIR = REPO_ROOT / "data" / "processed" / "normal"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Training checkpoint .pth (default: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image path; if omitted, uses the first image in --normal-dir",
    )
    p.add_argument(
        "--normal-dir",
        type=Path,
        default=DEFAULT_NORMAL_DIR,
        help=f"Folder to pick a sample from when --image is omitted (default: {DEFAULT_NORMAL_DIR})",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu, cuda, or cuda:0 (default: cuda if available else cpu)",
    )
    return p.parse_args()


def _first_image_in_dir(d: Path) -> Path:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")
    candidates = sorted(p for p in d.iterdir() if p.suffix.lower() in exts and p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No images ({', '.join(sorted(exts))}) in {d}")
    return candidates[0]


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict checkpoint at {path}")
    state = raw.get("state_dict", raw)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid state_dict in {path}")
    classes = raw.get("classes")
    if classes is None:
        num = raw.get("num_classes")
        if isinstance(num, int) and num > 0:
            classes = [str(i) for i in range(num)]
        else:
            classes = ["normal", "pneumonia"]
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"Checkpoint must define non-empty 'classes' list: {path}")
    return state, list(classes)


def main() -> None:
    args = _parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from training.utils import build_val_transforms

    ckpt_path = args.checkpoint.resolve()
    image_path = args.image.resolve() if args.image else _first_image_in_dir(args.normal_dir.resolve())

    state_dict, classes = _load_checkpoint(ckpt_path)
    num_classes = len(classes)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    transform = build_val_transforms()
    with Image.open(image_path) as im:
        tensor = transform(im.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor.to(device, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

    print(f"checkpoint: {ckpt_path}")
    print(f"image:      {image_path}")
    print(f"device:     {device}")
    print("probabilities:")
    for name, p in zip(classes, probs.tolist(), strict=True):
        print(f"  {name}: {p:.6f}")


if __name__ == "__main__":
    main()
