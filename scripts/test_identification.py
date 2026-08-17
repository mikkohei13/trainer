"""
Run the identification model on images in trainer/data/<project>/test.

Uses the same OD crop + 10% pad as training when a detection is found;
otherwise classifies the full image. Prints filename and top-k genera.

    uv run python scripts/test_identification.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from trainer import db
from trainer.images import IMAGE_EXTS
from trainer.inference import predict_top_box

PROJECT = "auchenorrhyncha"
RUN_ID = "20260817-231140"
TOP_K = 5
BBOX_PADDING_FRACTION = 0.10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "trainer" / "models" / PROJECT / "identification" / RUN_ID
TEST_DIR = PROJECT_ROOT / "trainer" / "data" / PROJECT / "test"
CHECKPOINT = RUN_DIR / "best.pt"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def letterbox(img: Image.Image, size: int, fill: int = 0) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), (fill, fill, fill))
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def crop_box_with_padding(img: Image.Image, box: dict, pad_frac: float) -> Image.Image:
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


def pil_to_tensor(img: Image.Image, img_size: int):
    import torch

    boxed = letterbox(img, img_size, fill=0)
    arr = np.asarray(boxed, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr)


def list_test_images(test_dir: Path) -> list[Path]:
    return sorted(
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_model(ckpt_path: Path, device):
    import torch
    import timm

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
    model = timm.create_model(
        ckpt["base_model"],
        pretrained=False,
        num_classes=len(class_to_idx),
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, idx_to_class, int(ckpt["img_size"])


def prepare_image(path: Path, od_model_path: Path | None) -> tuple[Image.Image, str]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
    if od_model_path is None:
        return rgb, "full (no OD model)"
    boxes = predict_top_box(od_model_path, path)
    if not boxes:
        return rgb, "full (no detection)"
    return crop_box_with_padding(rgb, boxes[0], BBOX_PADDING_FRACTION), "cropped"


def main() -> None:
    import torch
    import torch.nn.functional as F

    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")
    if not TEST_DIR.is_dir():
        raise SystemExit(f"Test directory not found: {TEST_DIR}")

    images = list_test_images(TEST_DIR)
    if not images:
        raise SystemExit(f"No images in {TEST_DIR}")

    device = pick_device()
    model, idx_to_class, img_size = load_model(CHECKPOINT, device)
    od_model_path = db.get_active_model_path_for_taxon(PROJECT)

    print(f"run={RUN_DIR}")
    print(f"test_dir={TEST_DIR}")
    print(f"device={device} classes={len(idx_to_class)} img_size={img_size}")
    print(f"od_model={od_model_path}")
    print()

    for path in images:
        try:
            crop, how = prepare_image(path, od_model_path)
        except (OSError, UnidentifiedImageError) as exc:
            print(f"{path.name}\n  error: {exc}\n")
            continue

        x = pil_to_tensor(crop, img_size).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
        k = min(TOP_K, probs.numel())
        values, indices = torch.topk(probs, k)

        print(f"{path.name}  [{how}]")
        for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            genus = idx_to_class.get(idx, f"?{idx}")
            print(f"  {rank}. {genus}  {score:.4f}")
        print()


if __name__ == "__main__":
    main()
