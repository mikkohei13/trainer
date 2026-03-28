"""YOLO model training: dataset export and fire-and-forget training task."""

import csv
import logging
import multiprocessing
import random
import shutil
import tempfile
from pathlib import Path

try:
    import torch._dynamo.config as _dynamo_config

    _dynamo_config.disable = True
except Exception:
    # Training should still proceed even if dynamo config is unavailable.
    pass

from PIL import Image

from trainer import db
from trainer.images import IMAGES_DIR

MODELS_DIR = Path(__file__).resolve().parent / "models"

BASE_MODEL = "yolo11m.pt" # XL: x, L: l, M: m
TRAIN_EPOCHS = 50
FREEZE_LAYERS = 10
LR0 = 0.001
PATIENCE = 20


def start_training_process(run_id: int, taxon: str) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_blocking_train, args=(run_id, taxon), daemon=True)
    p.start()


def _blocking_train(run_id: int, taxon: str) -> None:
    run_dir = MODELS_DIR / taxon / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    ul_logger = logging.getLogger("ultralytics")
    handler = logging.FileHandler(log_path)
    ul_logger.addHandler(handler)

    print(f"[training run {run_id}] starting for project '{taxon}'")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            train_count, val_count = export_yolo_dataset(taxon, dataset_dir)
            print(f"[training run {run_id}] dataset exported: {train_count} train, {val_count} val images")

            if train_count == 0:
                raise ValueError("No annotated training images available")

            # NumPy before torch/ultralytics avoids some torch._dynamo init edge cases.
            import numpy as np  # noqa: F401
            import torch  # noqa: F401

            from ultralytics import YOLO

            model = YOLO(BASE_MODEL)
            print(f"[training run {run_id}] training started (epochs={TRAIN_EPOCHS}, patience={PATIENCE})")
            model.train(
                data=str(dataset_dir / "dataset.yaml"),
                epochs=TRAIN_EPOCHS,
                freeze=FREEZE_LAYERS,
                lr0=LR0,
                patience=PATIENCE,
                verbose=True,
                project=str(run_dir.parent),
                name=str(run_id),
                exist_ok=True,
                workers=0,
            )

        best_pt = run_dir / "weights" / "best.pt"
        map50, map50_95 = _read_map_from_results(run_dir / "results.csv")
        print(f"[training run {run_id}] done — mAP50={map50}, mAP50-95={map50_95}, model={best_pt}")
        db.finish_training_run(run_id, str(best_pt), map50, map50_95, str(log_path))

    except Exception as exc:
        print(f"[training run {run_id}] failed: {exc}")
        db.fail_training_run(run_id, str(log_path))

    finally:
        handler.close()
        ul_logger.removeHandler(handler)


def export_yolo_dataset(taxon: str, output_dir: Path) -> tuple[int, int]:
    """
    Export annotated images as a YOLO dataset to output_dir.

    Annotated images (≥1 bounding box) get a label file with YOLO-format lines.
    Images marked no-organism get an empty label file (background examples).
    Unannotated images are excluded.

    Returns (train_count, val_count).
    """
    no_set, box_map = db.project_annotation_state(taxon)

    annotated_paths = []
    for p in _list_project_image_paths(taxon):
        if p in no_set or p in box_map:
            annotated_paths.append(p)

    if not annotated_paths:
        return 0, 0

    random.shuffle(annotated_paths)
    split_idx = max(1, int(len(annotated_paths) * 0.8))
    train_paths = annotated_paths[:split_idx]
    val_paths = annotated_paths[split_idx:]

    _write_subset(train_paths, no_set, output_dir, "train")

    if val_paths:
        _write_subset(val_paths, no_set, output_dir, "val")
        val_dir_rel = "images/val"
    else:
        val_dir_rel = "images/train"

    yaml_text = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        f"val: {val_dir_rel}\n"
        "nc: 1\n"
        "names: [organism]\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml_text)

    return len(train_paths), len(val_paths)


def _write_subset(
    image_paths: list[str],
    no_set: set[str],
    output_dir: Path,
    subset: str,
) -> None:
    img_dir = output_dir / "images" / subset
    lbl_dir = output_dir / "labels" / subset
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        src = IMAGES_DIR / image_path
        flat_name = image_path.replace("/", "_")
        dst_img = img_dir / flat_name
        shutil.copy2(src, dst_img)

        label_file = lbl_dir / (Path(flat_name).stem + ".txt")

        if image_path in no_set:
            label_file.write_text("")
        else:
            annotations = db.get_annotations(image_path)
            with Image.open(src) as img:
                img_w, img_h = img.size
            lines = []
            for box in annotations["boxes"]:
                cx = (box["x"] + box["w"] / 2) / img_w
                cy = (box["y"] + box["h"] / 2) / img_h
                bw = box["w"] / img_w
                bh = box["h"] / img_h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_file.write_text("\n".join(lines))


def _list_project_image_paths(taxon: str) -> list[str]:
    from trainer.images import IMAGE_EXTS

    project_dir = IMAGES_DIR / taxon
    if not project_dir.is_dir():
        return []
    paths = []
    for f in project_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            rel = f.relative_to(IMAGES_DIR)
            paths.append(str(rel).replace("\\", "/"))
    paths.sort()
    return paths


def _read_map_from_results(results_csv: Path) -> tuple[float | None, float | None]:
    if not results_csv.exists():
        return None, None

    with results_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None, None

    last = rows[-1]
    cleaned = {k.strip(): v.strip() for k, v in last.items()}

    try:
        map50 = float(cleaned["metrics/mAP50(B)"])
    except (KeyError, ValueError):
        map50 = None

    try:
        map50_95 = float(cleaned["metrics/mAP50-95(B)"])
    except (KeyError, ValueError):
        map50_95 = None

    return map50, map50_95
