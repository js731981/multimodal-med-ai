"""Production ResNet50 chest X-ray classifier inference (matches `training/` pipeline)."""

from __future__ import annotations

import io
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from backend.app.inference.gradcam import GradCAM, overlay_gradcam_on_pil
from backend.app.models.interfaces import ImageModel, ModelInfo

# Same as `training.utils` — validation / inference preprocessing (no augmentation).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_user_checkpoint_path(raw: str | Path, *, project_root: Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    return p


def resolve_image_checkpoint_path(
    explicit: str | Path | None = None,
    *,
    project_root: Path | None = None,
) -> Path:
    """
    Resolve the image classifier checkpoint path (cross-platform, no hardcoded absolute paths).

    Priority:
    1. ``explicit`` — constructor / ``MMEDAI_IMAGE_MODEL_KWARGS`` ``checkpoint_path`` (must exist).
    2. ``MMEDAI_IMAGE_CHECKPOINT_PATH`` / ``Settings.image_checkpoint_path`` (must exist if set).
    3. ``models/image_model/resnet_model.pth`` then ``models/image_model/dummy_smoke.pth``
       under the project root (first existing file wins).

    Raises ``FileNotFoundError`` with a clear message if nothing usable is found.
    """
    root = project_root or _project_root()

    if explicit is not None and str(explicit).strip():
        p = _normalize_user_checkpoint_path(explicit, project_root=root)
        if not p.is_file():
            raise FileNotFoundError(
                "Image model checkpoint not found at the configured path.\n\n"
                f"  Path: {p}\n\n"
                "Set ``checkpoint_path`` in ``MMEDAI_IMAGE_MODEL_KWARGS`` to a valid .pth file, "
                "or use MMEDAI_IMAGE_CHECKPOINT_PATH, or place resnet_model.pth / dummy_smoke.pth "
                "under models/image_model/."
            )
        return p

    from backend.app.config import get_settings

    cfg = get_settings().image_checkpoint_path
    if cfg is not None and str(cfg).strip():
        p = _normalize_user_checkpoint_path(cfg, project_root=root)
        if not p.is_file():
            raise FileNotFoundError(
                "Image model checkpoint not found (MMEDAI_IMAGE_CHECKPOINT_PATH / "
                "IMAGE_CHECKPOINT_PATH).\n\n"
                f"  Path: {p}\n\n"
                "Point the variable at an existing .pth file, or unset it to use the default "
                "search under models/image_model/."
            )
        return p

    primary = (root / "models" / "image_model" / "resnet_model.pth").resolve()
    fallback = (root / "models" / "image_model" / "dummy_smoke.pth").resolve()
    for cand in (primary, fallback):
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        "No image model checkpoint found.\n\n"
        "Searched under the project root:\n"
        f"  • {primary}\n"
        f"  • {fallback}\n\n"
        "Place ``resnet_model.pth`` (trained) or ``dummy_smoke.pth`` (smoke test) in "
        "``models/image_model/``, or set MMEDAI_IMAGE_CHECKPOINT_PATH to an existing .pth "
        "(absolute path or path relative to the project root)."
    )


def _default_checkpoint_path() -> Path:
    """Primary default candidate (``resnet_model.pth``); prefer :func:`resolve_image_checkpoint_path`."""
    return _project_root() / "models" / "image_model" / "resnet_model.pth"


def _default_gradcam_output_path() -> Path:
    return _project_root() / "outputs" / "gradcam_result.jpg"


def _build_inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Image model checkpoint not found: {path}")
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint at {path} must be a dict with 'state_dict' and 'classes'.")
    state = raw.get("state_dict", raw)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid state_dict in checkpoint: {path}")
    classes = raw.get("classes")
    if classes is None:
        classes = ["normal", "pneumonia"]
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ValueError(f"Checkpoint 'classes' must be a list of strings: {path}")
    return state, classes


class ImageModelImpl(ImageModel):
    """
    Singleton ResNet50 binary (or N-class) classifier.

    Checkpoint file is resolved at init via :func:`resolve_image_checkpoint_path` (constructor
    ``checkpoint_path``, then ``MMEDAI_IMAGE_CHECKPOINT_PATH``, then ``resnet_model.pth`` /
    ``dummy_smoke.pth`` under ``models/image_model/``).

    - ``encode(image)`` returns pooled CNN features for the fusion stage, shape ``[1, 2048]``.
    - ``predict(image)`` returns ``(embedding, probabilities)``: pooled ResNet features before ``fc``
      (shape ``[1, 2048]``) and softmax probabilities (shape ``[1, num_classes]``; column order matches
      ``classes`` in the checkpoint). One forward pass through the backbone per call.
    - ``predict_proba_dict(image)`` returns ``{"pneumonia": float, "normal": float}`` for the standard
      two-class checkpoint (class folder names from training).

    Accepts PIL ``Image``, ``bytes``, a filesystem path (``str`` / ``Path``), or a ``[3,H,W]`` float tensor in ``[0,1]``.
    """

    _singleton: ClassVar[ImageModelImpl | None] = None
    _singleton_lock: ClassVar[threading.Lock] = threading.Lock()

    info: ModelInfo

    def __new__(
        cls,
        *,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> ImageModelImpl:
        with cls._singleton_lock:
            if cls._singleton is None:
                instance = super().__new__(cls)
                instance._init_once(
                    checkpoint_path=checkpoint_path,
                    device=device,
                )
                cls._singleton = instance
            return cls._singleton

    def _init_once(
        self,
        *,
        checkpoint_path: str | Path | None,
        device: str | torch.device | None,
    ) -> None:
        explicit = checkpoint_path if (checkpoint_path is not None and str(checkpoint_path).strip()) else None
        self._checkpoint_path = resolve_image_checkpoint_path(explicit)
        self._device_pref = device
        self._device: torch.device | None = None
        self._model: nn.Module | None = None
        self._load_lock = threading.Lock()
        self._transform = _build_inference_transform()
        self._classes: list[str] = []
        self._class_to_output_idx: dict[str, int] = {}
        self.info = ModelInfo(
            name="resnet50-cxr",
            version="1.0.0",
            extra={"checkpoint": str(self._checkpoint_path)},
        )

    @classmethod
    def reset_singleton_for_testing(cls) -> None:
        """Release the singleton (tests only)."""
        with cls._singleton_lock:
            cls._singleton = None

    def load_model(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return

            state, classes = _load_checkpoint_state(self._checkpoint_path)
            num_classes = len(classes)
            if num_classes == 0:
                raise ValueError("Checkpoint lists zero classes.")

            model = resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            model.load_state_dict(state, strict=True)

            self._device = _resolve_device(self._device_pref)
            self._model = model.to(self._device).eval()
            self._classes = list(classes)
            lower = [c.lower() for c in self._classes]
            self._class_to_output_idx = {name: i for i, name in enumerate(lower)}

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load_model()

    def _prepare_pil(self, image: Any) -> Image.Image:
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.is_file():
                raise FileNotFoundError(f"Image file not found: {p}")
            with Image.open(p) as im:
                return im.convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            with Image.open(io.BytesIO(bytes(image))) as im:
                return im.convert("RGB")
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 4:
                if t.shape[0] != 1:
                    raise ValueError("Image tensor batch must have size 1 for inference.")
                t = t[0]
            if t.ndim != 3 or t.shape[0] != 3:
                raise ValueError("Expected image tensor [3, H, W] or [1, 3, H, W].")
            arr = (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
            return Image.fromarray(arr, mode="RGB")
        raise TypeError("Expected PIL.Image.Image, str path, pathlib.Path, bytes, or torch.Tensor.")

    @torch.inference_mode()
    def _embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Pooled ResNet50 features before the classification head (shape [B, 2048])."""
        self._ensure_loaded()
        assert self._model is not None and self._device is not None
        x = batch.to(self._device, dtype=torch.float32, non_blocking=self._device.type == "cuda")
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)
        x = self._model.layer1(x)
        x = self._model.layer2(x)
        x = self._model.layer3(x)
        x = self._model.layer4(x)
        x = self._model.avgpool(x)
        return torch.flatten(x, 1)

    @torch.inference_mode()
    def _forward_tensor(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._embed_batch(batch)
        logits = self._model.fc(emb)
        probs = torch.softmax(logits, dim=-1)
        return emb, probs

    @torch.inference_mode()
    def encode(self, image: Any) -> torch.Tensor:
        """
        Return visual embedding for the fusion model, shape ``[1, 2048]`` (float32, CPU).
        """
        pil = self._prepare_pil(image)
        tensor = self._transform(pil).unsqueeze(0)
        emb = self._embed_batch(tensor)
        return emb.detach().to(dtype=torch.float32, device="cpu")

    @torch.inference_mode()
    def predict(self, image: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference: ResNet pooled embedding (before ``fc``) and class probabilities.

        Returns ``(embedding, probs)`` with shapes ``[1, 2048]`` and ``[1, num_classes]``
        (float32, CPU). Class order matches the checkpoint ``classes`` field.
        """
        pil = self._prepare_pil(image)
        tensor = self._transform(pil).unsqueeze(0)
        emb, probs = self._forward_tensor(tensor)
        return (
            emb.detach().to(dtype=torch.float32, device="cpu"),
            probs.detach().to(dtype=torch.float32, device="cpu"),
        )

    def predict_proba_dict(self, image: Any) -> dict[str, float]:
        """
        Return ``{"pneumonia": p0, "normal": p1}`` using the trained class indices from the checkpoint.
        """
        _, probs = self.predict(image)
        probs = probs[0]
        idx_n = self._class_to_output_idx.get("normal")
        idx_p = self._class_to_output_idx.get("pneumonia")
        if idx_n is None or idx_p is None:
            raise ValueError(
                "Checkpoint classes must include 'normal' and 'pneumonia' (case-insensitive). "
                f"Got: {self._classes}"
            )
        return {
            "pneumonia": float(probs[idx_p].item()),
            "normal": float(probs[idx_n].item()),
        }

    def predict_with_gradcam(
        self,
        image: Any,
        *,
        enable_gradcam: bool = False,
        alpha: float = 0.45,
    ) -> dict[str, Any]:
        """
        Run image classification and optionally save a Grad-CAM overlay.

        When ``enable_gradcam`` is ``False`` (default), only a single inference-mode forward
        pass runs (same cost as ``predict``). Grad-CAM uses an extra backward pass only when
        enabled.

        Returns ``{"prediction": {...}, "gradcam_path": ...}`` where ``gradcam_path`` is
        ``null`` when Grad-CAM is disabled, otherwise a project-relative path like
        ``outputs/gradcam_<timestamp>.jpg``.
        """
        self._ensure_loaded()
        pil = self._prepare_pil(image)
        tensor = self._transform(pil).unsqueeze(0)
        _, probs = self._forward_tensor(tensor)
        probs_cpu = probs.detach().to(dtype=torch.float32, device="cpu")
        pred_idx = int(torch.argmax(probs_cpu[0]).item())

        prob_map: dict[str, float] = {
            str(self._classes[i]): float(probs_cpu[0, i].item()) for i in range(probs_cpu.shape[1])
        }
        prediction: dict[str, Any] = {
            "class_index": pred_idx,
            "class_name": str(self._classes[pred_idx]) if self._classes else str(pred_idx),
            "probabilities": prob_map,
        }

        gradcam_path: str | None = None
        if enable_gradcam:
            assert self._model is not None and self._device is not None
            batch = tensor.to(self._device, dtype=torch.float32, non_blocking=self._device.type == "cuda")
            batch = batch.clone().requires_grad_(True)
            with self._load_lock:
                with torch.enable_grad():
                    cam_hw = GradCAM(self._model).generate(batch, pred_idx)
            overlay = overlay_gradcam_on_pil(pil, cam_hw, alpha=alpha)
            out_dir = _project_root() / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"gradcam_{ts}.jpg"
            out_path = out_dir / filename
            overlay.save(out_path, format="JPEG", quality=92)
            gradcam_path = f"outputs/{filename}"

        return {"prediction": prediction, "gradcam_path": gradcam_path}

    def _resolve_gradcam_class_idx(self, target_class: int | str | None, pil: Image.Image) -> int:
        if target_class is None:
            _, probs = self.predict(pil)
            return int(torch.argmax(probs[0]).item())
        if isinstance(target_class, int):
            if target_class < 0 or target_class >= len(self._classes):
                raise ValueError(f"target_class index out of range: {target_class} (num_classes={len(self._classes)})")
            return target_class
        key = str(target_class).lower()
        idx = self._class_to_output_idx.get(key)
        if idx is None:
            raise ValueError(f"Unknown class '{target_class}'. Known: {self._classes}")
        return idx

    def gradcam_overlay(
        self,
        image: Any,
        *,
        target_class: int | str | None = None,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Grad-CAM over the last ResNet block, resized and blended onto the original image.

        ``target_class`` is an output index, a class name (case-insensitive), or ``None``
        for the argmax of ``predict``.
        """
        self._ensure_loaded()
        assert self._model is not None and self._device is not None
        pil = self._prepare_pil(image)
        class_idx = self._resolve_gradcam_class_idx(target_class, pil)

        batch = self._transform(pil).unsqueeze(0)
        batch = batch.to(self._device, dtype=torch.float32, non_blocking=self._device.type == "cuda")
        batch = batch.clone().requires_grad_(True)

        with self._load_lock:
            with torch.enable_grad():
                cam_hw = GradCAM(self._model).generate(batch, class_idx)

        return overlay_gradcam_on_pil(pil, cam_hw, alpha=alpha)

    def save_gradcam(
        self,
        image: Any,
        *,
        target_class: int | str | None = None,
        output_path: str | Path | None = None,
        alpha: float = 0.45,
    ) -> Path:
        """Write a JPEG explanation to ``outputs/gradcam_result.jpg`` by default."""
        out = Path(output_path) if output_path is not None else _default_gradcam_output_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        overlay = self.gradcam_overlay(image, target_class=target_class, alpha=alpha)
        overlay.save(out, format="JPEG", quality=92)
        return out
