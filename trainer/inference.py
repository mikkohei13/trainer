"""YOLO inference for annotation assist (single top-confidence box)."""

from pathlib import Path

try:
    import torch._dynamo.config as _dynamo_config

    _dynamo_config.disable = True
except Exception:
    pass

_model_cache: dict[str, object] = {}


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
