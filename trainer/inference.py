"""Inference helpers for detection and quality models."""

import math
from pathlib import Path

from PIL import Image

try:
    import torch._dynamo.config as _dynamo_config

    _dynamo_config.disable = True
except Exception:
    pass

_model_cache: dict[str, object] = {}
_quality_model_cache: dict[str, tuple[object, int, float]] = {}


def _get_yolo(model_path: Path):
    key = str(model_path.resolve())
    if key not in _model_cache:
        from ultralytics import YOLO

        _model_cache[key] = YOLO(key)
    return _model_cache[key]


def predict_top_box(model_path: Path, image_abs_path: Path) -> list[dict]:
    """
    Run detection and return at most one box (highest confidence) as
    pixel-space x, y, w, h (top-left origin, matching Annotorious).
    """
    model = _get_yolo(model_path)
    results = model.predict(
        source=str(image_abs_path),
        verbose=False,
    )
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    confs = boxes.conf
    xyxy = boxes.xyxy
    if confs is None or xyxy is None:
        return []

    best_i = int(confs.argmax().item())
    x1, y1, x2, y2 = xyxy[best_i].tolist()
    w = x2 - x1
    h = y2 - y1
    return [{"x": float(x1), "y": float(y1), "w": float(w), "h": float(h)}]


def _get_quality_model(model_path: Path):
    key = str(model_path.resolve())
    if key in _quality_model_cache:
        return _quality_model_cache[key]

    import torch
    from torch import nn
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(key, map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid(),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    img_size = int(checkpoint.get("img_size", 224))
    padding_fraction = float(checkpoint.get("padding_fraction", 0.10))
    _quality_model_cache[key] = (model, img_size, padding_fraction)
    return _quality_model_cache[key]


def _crop_box_with_padding(img: Image.Image, box: dict, pad_frac: float) -> Image.Image:
    img_w, img_h = img.size
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])

    pad_x = w * pad_frac
    pad_y = h * pad_frac

    left = max(0, int(math.floor(x - pad_x)))
    top = max(0, int(math.floor(y - pad_y)))
    right = min(img_w, int(math.ceil(x + w + pad_x)))
    bottom = min(img_h, int(math.ceil(y + h + pad_y)))

    if right <= left or bottom <= top:
        return img.copy()
    return img.crop((left, top, right, bottom))


def predict_quality_score(model_path: Path, image_abs_path: Path, box: dict) -> float:
    import torch
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    model, img_size, padding_fraction = _get_quality_model(model_path)
    device = next(model.parameters()).device

    transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with Image.open(image_abs_path) as img:
        rgb = img.convert("RGB")
        crop = _crop_box_with_padding(rgb, box, padding_fraction)

    x = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x).squeeze(1).item()
    return float(y)
