"""Transforms, model factory, metrics, and training helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_val_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Toggle gradients for all parameters except the final linear layer."""
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            continue
        param.requires_grad = trainable


def build_resnet50_classifier(
    num_classes: int,
    *,
    freeze_backbone: bool = True,
    weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V1,
) -> nn.Module:
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    set_backbone_trainable(model, not freeze_backbone)
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def accuracy_from_logits(logits: Tensor, targets: Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


@dataclass
class MetricTracker:
    """Running average for loss and accuracy over one pass."""

    loss_sum: float = 0.0
    acc_sum: float = 0.0
    n_batches: int = 0
    n_samples: int = 0

    def update(self, loss: float, acc: float, batch_size: int) -> None:
        self.loss_sum += loss
        self.acc_sum += acc * batch_size
        self.n_batches += 1
        self.n_samples += batch_size

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(self.n_batches, 1)

    @property
    def avg_accuracy(self) -> float:
        return self.acc_sum / max(self.n_samples, 1)
