"""Train ResNet50 for chest X-ray folder-based classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from training.dataset import build_imagefolder, train_val_subsets
from training.utils import (
    MetricTracker,
    accuracy_from_logits,
    build_resnet50_classifier,
    build_train_transforms,
    build_val_transforms,
    set_backbone_trainable,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chest X-ray ResNet50 transfer learning")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root with class subfolders (e.g. pneumonia/, normal/)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/image_model/resnet_model.pth"),
        help="Checkpoint path (.pth)",
    )
    p.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
        help="ReduceLROnPlateau patience (epochs without val loss improvement)",
    )
    p.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="LR multiply factor when plateau scheduler triggers",
    )
    p.add_argument(
        "--unfreeze-backbone-epoch",
        type=int,
        default=None,
        help="1-based epoch index after which backbone is unfrozen (optional)",
    )
    p.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="LR for backbone after unfreeze; defaults to --lr * 0.1",
    )
    return p.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    tracker = MetricTracker()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        acc = accuracy_from_logits(logits.detach(), targets)
        tracker.update(loss.item(), acc, images.size(0))
    return tracker.avg_loss, tracker.avg_accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    tracker = MetricTracker()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        acc = accuracy_from_logits(logits, targets)
        tracker.update(loss.item(), acc, images.size(0))
    return tracker.avg_loss, tracker.avg_accuracy


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr: float | None,
    backbone_unfrozen: bool,
) -> Adam:
    if backbone_unfrozen:
        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("fc."):
                head_params.append(p)
            else:
                backbone_params.append(p)
        blr = backbone_lr if backbone_lr is not None else lr * 0.1
        params = []
        if backbone_params:
            params.append({"params": backbone_params, "lr": blr})
        if head_params:
            params.append({"params": head_params, "lr": lr})
        return Adam(params, weight_decay=weight_decay)
    return Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = args.data_dir.resolve()
    ref_ds = build_imagefolder(data_root, transform=None)
    num_classes = len(ref_ds.classes)
    train_tf, val_tf = build_train_transforms(), build_val_transforms()
    train_split, val_split = train_val_subsets(ref_ds, args.val_fraction, args.seed)
    train_ds = Subset(build_imagefolder(data_root, transform=train_tf), train_split.indices)
    val_ds = Subset(build_imagefolder(data_root, transform=val_tf), val_split.indices)
    if len(train_ds) == 0:
        raise ValueError("Train split is empty; check data-dir and val-fraction.")
    if len(val_ds) == 0:
        raise ValueError(
            "Validation split is empty; add more images per class or lower val-fraction."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet50_classifier(num_classes, freeze_backbone=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    backbone_unfrozen = False
    optimizer = build_optimizer(
        model, args.lr, args.weight_decay, args.backbone_lr, backbone_unfrozen
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    best_val_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if (
            args.unfreeze_backbone_epoch is not None
            and epoch == args.unfreeze_backbone_epoch
            and not backbone_unfrozen
        ):
            set_backbone_trainable(model, True)
            backbone_unfrozen = True
            optimizer = build_optimizer(
                model,
                args.lr,
                args.weight_decay,
                args.backbone_lr,
                backbone_unfrozen,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "class_to_idx": ref_ds.class_to_idx,
                "classes": ref_ds.classes,
                "num_classes": num_classes,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            torch.save(checkpoint, args.output)
            meta_path = args.output.with_suffix(".json")
            meta_path.write_text(
                json.dumps(
                    {
                        "classes": ref_ds.classes,
                        "class_to_idx": ref_ds.class_to_idx,
                        "num_classes": num_classes,
                        "best_val_loss": val_loss,
                        "best_val_acc": val_acc,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    print(f"Best val_loss={best_val_loss:.4f} — saved to {args.output}")


if __name__ == "__main__":
    main()
