"""YOLO model training: dataset export and fire-and-forget training task."""

import csv
import logging
import multiprocessing
import os
import random
import re
import shutil
import traceback
from pathlib import Path

try:
    import torch._dynamo.config as _dynamo_config

    _dynamo_config.disable = True
except Exception:
    # Training should still proceed even if dynamo config is unavailable.
    pass

from PIL import Image, UnidentifiedImageError

from trainer import db
from trainer.images import IMAGES_DIR

MODELS_DIR = Path(__file__).resolve().parent / "models"

BASE_MODEL = "yolo11m.pt"  # XL: x, L: l, M: m
TRAIN_EPOCHS = 100
FREEZE_LAYERS = 10
LR0 = 0.0008
PATIENCE = 20
MAX_TRAIN_ATTEMPTS = 3

_IMAGE_NOT_FOUND_RE = re.compile(
    r"(?:Image Not Found|No such file or directory:?)\s*['\"]?(.+?\.(?:jpg|jpeg|png))['\"]?",
    re.IGNORECASE,
)


def start_training_process(run_id: int, taxon: str) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_blocking_train, args=(run_id, taxon), daemon=False)
    p.start()


def _blocking_train(run_id: int, taxon: str) -> None:
    try:
        os.setsid()
    except OSError:
        pass

    db.set_training_run_pid(run_id, os.getpid())

    run_dir = MODELS_DIR / taxon / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    dataset_dir = run_dir / "dataset"

    ul_logger = logging.getLogger("ultralytics")
    handler = logging.FileHandler(log_path)
    ul_logger.addHandler(handler)

    print(f"[training run {run_id}] starting for project '{taxon}'", flush=True)

    try:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        train_count, val_count = export_yolo_dataset(taxon, dataset_dir)
        print(
            f"[training run {run_id}] dataset exported: {train_count} train, {val_count} val images",
            flush=True,
        )

        if train_count == 0:
            raise ValueError("No annotated training images available")

        # NumPy before torch/ultralytics avoids some torch._dynamo init edge cases.
        import numpy as np  # noqa: F401
        import torch  # noqa: F401

        from ultralytics import YOLO

        train_kwargs = dict(
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

        last_pt = run_dir / "weights" / "last.pt"
        last_exc: Exception | None = None

        for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
            try:
                if last_pt.is_file() and attempt > 1:
                    print(
                        f"[training run {run_id}] resume attempt {attempt}/{MAX_TRAIN_ATTEMPTS} from {last_pt}",
                        flush=True,
                    )
                    model = YOLO(str(last_pt))
                    model.train(resume=True)
                else:
                    print(
                        f"[training run {run_id}] training started "
                        f"(epochs={TRAIN_EPOCHS}, patience={PATIENCE}, attempt={attempt}/{MAX_TRAIN_ATTEMPTS})",
                        flush=True,
                    )
                    model = YOLO(BASE_MODEL)
                    model.train(**train_kwargs)
                last_exc = None
                break
            except (FileNotFoundError, OSError) as exc:
                last_exc = exc
                tb = traceback.format_exc()
                print(f"[training run {run_id}] I/O error on attempt {attempt}: {exc}\n{tb}", flush=True)
                handler.stream.write(tb)
                handler.stream.flush()

                bad = parse_missing_image_path(exc)
                if bad is not None:
                    _drop_dataset_image(dataset_dir, Path(bad))
                    print(f"[training run {run_id}] dropped bad dataset image: {bad}", flush=True)

                if attempt >= MAX_TRAIN_ATTEMPTS:
                    raise
            except Exception as exc:
                msg = str(exc)
                if "Image Not Found" not in msg and "No such file or directory" not in msg:
                    raise
                last_exc = exc
                tb = traceback.format_exc()
                print(f"[training run {run_id}] I/O error on attempt {attempt}: {exc}\n{tb}", flush=True)
                handler.stream.write(tb)
                handler.stream.flush()

                bad = parse_missing_image_path(exc)
                if bad is not None:
                    _drop_dataset_image(dataset_dir, Path(bad))
                    print(f"[training run {run_id}] dropped bad dataset image: {bad}", flush=True)

                if attempt >= MAX_TRAIN_ATTEMPTS:
                    raise

        if last_exc is not None:
            raise last_exc

        best_pt = run_dir / "weights" / "best.pt"
        map50, map50_95 = _read_map_from_results(run_dir / "results.csv")
        print(
            f"[training run {run_id}] done — mAP50={map50}, mAP50-95={map50_95}, model={best_pt}",
            flush=True,
        )
        db.finish_training_run(run_id, str(best_pt), map50, map50_95, str(log_path))

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[training run {run_id}] failed: {exc}\n{tb}", flush=True)
        try:
            handler.stream.write(tb)
            handler.stream.flush()
        except Exception:
            pass
        db.fail_training_run(run_id, str(log_path))

    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        handler.close()
        ul_logger.removeHandler(handler)


def export_yolo_dataset(taxon: str, output_dir: Path) -> tuple[int, int]:
    """
    Export annotated images as a YOLO dataset to output_dir.

    Annotated images (≥1 bounding box) get a label file with YOLO-format lines.
    Images marked no-organism get an empty label file (background examples).
    Unannotated images are excluded.
    Missing or unreadable sources are skipped.

    Returns (train_count, val_count) of images actually written.
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

    train_count = _write_subset(train_paths, no_set, output_dir, "train")

    if val_paths:
        val_count = _write_subset(val_paths, no_set, output_dir, "val")
        val_dir_rel = "images/val"
    else:
        val_count = 0
        val_dir_rel = "images/train"

    yaml_text = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        f"val: {val_dir_rel}\n"
        "nc: 1\n"
        "names: [organism]\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml_text)

    return train_count, val_count


def parse_missing_image_path(exc: BaseException) -> str | None:
    """Extract a dataset image path from a FileNotFoundError / Ultralytics message."""
    text = str(exc)
    if getattr(exc, "filename", None):
        name = str(exc.filename)
        if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return name
    m = _IMAGE_NOT_FOUND_RE.search(text)
    if m:
        return m.group(1).strip().strip("'\"")
    # errno-style: [Errno 2] No such file or directory: 'path'
    m2 = re.search(r":\s*['\"]([^'\"]+\.(?:jpg|jpeg|png))['\"]", text, re.IGNORECASE)
    if m2:
        return m2.group(1)
    return None


def _drop_dataset_image(dataset_dir: Path, image_path: Path) -> None:
    """Remove a bad train/val image, its label, and YOLO label caches."""
    path = image_path
    if not path.is_absolute():
        path = dataset_dir / path
    try:
        path = path.resolve()
        dataset_dir.resolve()
        path.relative_to(dataset_dir.resolve())
    except (ValueError, OSError):
        return

    if path.is_file():
        path.unlink(missing_ok=True)

    # labels/<subset>/<stem>.txt next to images/<subset>/<name>
    parts = path.parts
    try:
        images_idx = parts.index("images")
    except ValueError:
        return
    if images_idx + 2 >= len(parts):
        return
    subset = parts[images_idx + 1]
    label = dataset_dir / "labels" / subset / (path.stem + ".txt")
    if label.is_file():
        label.unlink(missing_ok=True)

    for cache in (dataset_dir / "labels" / subset).glob("*.cache"):
        cache.unlink(missing_ok=True)
    for cache in dataset_dir.glob("*.cache"):
        cache.unlink(missing_ok=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_subset(
    image_paths: list[str],
    no_set: set[str],
    output_dir: Path,
    subset: str,
) -> int:
    img_dir = output_dir / "images" / subset
    lbl_dir = output_dir / "labels" / subset
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_path in image_paths:
        src = IMAGES_DIR / image_path
        flat_name = image_path.replace("/", "_")
        dst_img = img_dir / flat_name
        label_file = lbl_dir / (Path(flat_name).stem + ".txt")

        if not src.is_file():
            print(f"[export] skip missing image: {image_path}", flush=True)
            continue

        try:
            with Image.open(src) as img:
                img_w, img_h = img.size
            _link_or_copy(src, dst_img)
            if image_path in no_set:
                label_file.write_text("")
            else:
                annotations = db.get_annotations(image_path)
                lines = []
                for box in annotations["boxes"]:
                    cx = (box["x"] + box["w"] / 2) / img_w
                    cy = (box["y"] + box["h"] / 2) / img_h
                    bw = box["w"] / img_w
                    bh = box["h"] / img_h
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                label_file.write_text("\n".join(lines))
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            print(f"[export] skip unreadable image {image_path}: {exc}", flush=True)
            dst_img.unlink(missing_ok=True)
            label_file.unlink(missing_ok=True)
            continue

        written += 1

    return written


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
