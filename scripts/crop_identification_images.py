"""
Crop project images with the active object-detection model (+ 10% pad).

Writes to trainer/images_processed/<project>/ mirroring the original
collection/taxon/file layout under trainer/images/<project>/.
Never modifies originals. Skips outputs that already exist.

    uv run python scripts/crop_identification_images.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from trainer import db
from trainer.images import IMAGES_DIR, list_project_image_paths
from trainer.inference import predict_top_box

# ---------------------------------------------------------------------------
# Hardcoded parameters
# ---------------------------------------------------------------------------

PROJECT = "auchenorrhyncha"
BBOX_PADDING_FRACTION = 0.10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "trainer" / "images_processed"

LIMIT = 1000


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


def processed_path_for(rel_path: str) -> Path:
    """
    Map trainer/images-relative path to images_processed.

    rel_path is like: auchenorrhyncha/britishbugs/Acericerus_ribauti/foo.jpg
    """
    return PROCESSED_DIR / rel_path


def save_crop(crop: Image.Image, out_abs: Path) -> None:
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_abs.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        crop.save(out_abs, format="JPEG", quality=95)
    elif suffix == ".png":
        crop.save(out_abs, format="PNG")
    else:
        crop.save(out_abs.with_suffix(".jpg"), format="JPEG", quality=95)


def crop_project_images(project: str) -> None:
    model_path = db.get_active_model_path_for_taxon(project)
    if model_path is None:
        raise SystemExit(
            f"No active object-detection model for project '{project}'. "
            "Activate a finished OD training run first."
        )

    out_root = PROCESSED_DIR / project
    out_root.mkdir(parents=True, exist_ok=True)

    paths = list_project_image_paths(project)
    total = len(paths)
    written = 0
    skipped_exists = 0
    skipped_no_box = 0
    skipped_bad = 0

    print(f"project={project} images={total} model={model_path}")
    print(f"output={out_root}")

    for i, rel_path in enumerate(paths, start=1):
        if i > LIMIT:
            print(f"reached limit {LIMIT}")
            break

        if i % 100 == 0 or i == 1:
            print(f"cropping {i}/{total} …")

        out_abs = processed_path_for(rel_path)
        if out_abs.is_file():
            skipped_exists += 1
            continue

        abs_path = IMAGES_DIR / rel_path
        boxes = predict_top_box(model_path, abs_path)
        if not boxes:
            skipped_no_box += 1
            continue

        try:
            with Image.open(abs_path) as img:
                rgb = img.convert("RGB")
                crop = crop_box_with_padding(rgb, boxes[0], BBOX_PADDING_FRACTION)
        except (OSError, UnidentifiedImageError) as exc:
            print(f"skip bad image {rel_path}: {exc}")
            skipped_bad += 1
            continue

        save_crop(crop, out_abs)
        written += 1

    print(
        f"done: written={written}, skipped_exists={skipped_exists}, "
        f"skipped_no_box={skipped_no_box}, skipped_bad={skipped_bad}"
    )


def main() -> None:
    if db.get_project(PROJECT) is None:
        raise SystemExit(f"Project '{PROJECT}' not found in database")
    crop_project_images(PROJECT)


if __name__ == "__main__":
    main()
