"""
Crop project images with the active object-detection model (+ 10% pad)
and score each crop with the active quality model.

Writes to trainer/images_processed/<project>/ mirroring the original
collection/taxon/file layout under trainer/images/<project>/.
Quality scores go to trainer/images_processed/<project>/quality.json.
Never modifies originals. Existing crops and ratings are skipped.

    uv run python scripts/crop_identification_images.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

from trainer import db
from trainer.images import IMAGES_DIR, list_project_image_paths
from trainer.inference import predict_quality_score, predict_top_box

# ---------------------------------------------------------------------------
# Hardcoded parameters
# ---------------------------------------------------------------------------

PROJECT = "auchenorrhyncha"
BBOX_PADDING_FRACTION = 0.10
QUALITY_JSON_NAME = "quality.json"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "trainer" / "images_processed"

LIMIT = 100000


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


def quality_json_path(project: str) -> Path:
    return PROCESSED_DIR / project / QUALITY_JSON_NAME


def load_quality_ratings(project: str) -> dict[str, float]:
    path = quality_json_path(project)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def save_quality_ratings(project: str, ratings: dict[str, float]) -> None:
    path = quality_json_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ratings, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    quality_model_path = db.get_active_quality_model_path_for_taxon(project)
    if quality_model_path is None:
        raise SystemExit(
            f"No active quality model for project '{project}'. "
            "Activate a finished quality training run first."
        )

    out_root = PROCESSED_DIR / project
    out_root.mkdir(parents=True, exist_ok=True)
    ratings = load_quality_ratings(project)

    paths = list_project_image_paths(project)
    total = len(paths)
    written = 0
    skipped_exists = 0
    skipped_no_box = 0
    skipped_bad = 0
    rated = 0
    skipped_rated = 0
    ratings_dirty = False

    print(f"project={project} images={total} model={model_path}")
    print(f"quality_model={quality_model_path}")
    print(f"output={out_root}")
    print(f"quality_json={quality_json_path(project)}")

    for i, rel_path in enumerate(paths, start=1):
        if i > LIMIT:
            print(f"reached limit {LIMIT}")
            break

        if i % 100 == 0 or i == 1:
            print(f"processing {i}/{total} …")

        out_abs = processed_path_for(rel_path)
        if out_abs.is_file():
            skipped_exists += 1
        else:
            abs_path = IMAGES_DIR / rel_path
            try:
                boxes = predict_top_box(model_path, abs_path)
                if not boxes:
                    skipped_no_box += 1
                    continue

                with Image.open(abs_path) as img:
                    rgb = img.convert("RGB")
                    crop = crop_box_with_padding(rgb, boxes[0], BBOX_PADDING_FRACTION)
                save_crop(crop, out_abs)
            except Exception as exc:
                print(f"skip failed image {rel_path}: {exc}")
                skipped_bad += 1
                continue
            written += 1

        if rel_path in ratings:
            skipped_rated += 1
            continue
        if not out_abs.is_file():
            continue

        try:
            ratings[rel_path] = predict_quality_score(quality_model_path, out_abs)
        except Exception as exc:
            print(f"skip quality score {rel_path}: {exc}")
            skipped_bad += 1
            continue
        rated += 1
        ratings_dirty = True
        if rated % 50 == 0:
            save_quality_ratings(project, ratings)
            ratings_dirty = False

    if ratings_dirty:
        save_quality_ratings(project, ratings)

    print(
        f"done: written={written}, skipped_exists={skipped_exists}, "
        f"skipped_no_box={skipped_no_box}, skipped_bad={skipped_bad}, "
        f"rated={rated}, skipped_rated={skipped_rated}"
    )


def main() -> None:
    if db.get_project(PROJECT) is None:
        raise SystemExit(f"Project '{PROJECT}' not found in database")
    crop_project_images(PROJECT)


if __name__ == "__main__":
    main()
