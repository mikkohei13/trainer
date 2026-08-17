"""Quality regression model training."""

import logging
import math
import multiprocessing
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from trainer import db
from trainer.images import IMAGES_DIR
from trainer.inference import DETECTION_CONF_THRESHOLD, predict_boxes, quality_crop_box

MODELS_DIR = Path(__file__).resolve().parent / "models"

QUALITY_BASE_MODEL = "resnet18"
QUALITY_TRAIN_EPOCHS = 50
QUALITY_PATIENCE = 8
QUALITY_LR = 3e-4
QUALITY_BACKBONE_LR = 1e-4
QUALITY_WEIGHT_DECAY = 1e-3
QUALITY_SMOOTH_L1_BETA = 0.15
QUALITY_BATCH_SIZE = 32
QUALITY_IMG_SIZE = 224
BBOX_PADDING_FRACTION = 0.10
QUALITY_RATINGS = (0.0, 0.333, 0.666, 1.0)


def start_quality_training_process(run_id: int, taxon: str) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_blocking_quality_train, args=(run_id, taxon), daemon=False)
    p.start()


def _blocking_quality_train(run_id: int, taxon: str) -> None:
    try:
        os.setsid()
    except OSError:
        pass

    db.set_quality_training_run_pid(run_id, os.getpid())

    run_dir = MODELS_DIR / taxon / "quality" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    trainer_logger = logging.getLogger(f"quality_training.{run_id}")
    trainer_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    trainer_logger.addHandler(handler)

    def emit(msg: str) -> None:
        trainer_logger.info(msg)
        print(msg, flush=True)

    emit(f"[quality run {run_id}] starting for project '{taxon}'")

    try:
        records = _collect_quality_records(taxon)
        if len(records) < 5:
            raise ValueError(
                "Need at least 5 quality-rated images with an insect detection for training"
            )

        train_records, val_records = _split_records(records)
        emit(
            f"[quality run {run_id}] dataset: train={len(train_records)}, val={len(val_records)} images"
        )
        _log_split_stats(emit, run_id, "train", train_records)
        _log_split_stats(emit, run_id, "val", val_records)

        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import models
        from torchvision.transforms import (
            Compose,
            Normalize,
            RandomHorizontalFlip,
            Resize,
            ToTensor,
        )

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = Compose([
            Resize((QUALITY_IMG_SIZE, QUALITY_IMG_SIZE)),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        val_transform = Compose([
            Resize((QUALITY_IMG_SIZE, QUALITY_IMG_SIZE)),
            ToTensor(),
            normalize,
        ])

        train_ds = _QualityCropDataset(train_records, train_transform)
        val_ds = _QualityCropDataset(val_records, val_transform)
        train_loader = DataLoader(train_ds, batch_size=QUALITY_BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=QUALITY_BATCH_SIZE, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid(),
        )
        model = model.to(device)

        optimizer = _make_optimizer(model, QUALITY_BACKBONE_LR, QUALITY_LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        criterion = nn.SmoothL1Loss(beta=QUALITY_SMOOTH_L1_BETA)

        best_rmse = None
        best_state = None
        stale_epochs = 0
        best_targets: list[float] = []
        best_preds: list[float] = []

        emit(
            f"[quality run {run_id}] training started "
            f"(epochs={QUALITY_TRAIN_EPOCHS}, patience={QUALITY_PATIENCE})"
        )

        for epoch in range(1, QUALITY_TRAIN_EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).squeeze(1)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = x.size(0)
                train_loss_sum += float(loss.detach().cpu().item()) * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(1, train_count)
            val_rmse, val_targets, val_preds = _evaluate(model, val_loader, device)
            scheduler.step(val_rmse)
            emit(
                f"[quality run {run_id}] epoch={epoch} train_loss={train_loss:.6f} val_rmse={val_rmse:.6f}"
            )

            if best_rmse is None or val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_targets = val_targets
                best_preds = val_preds
                stale_epochs = 0
                _log_val_diagnostics(emit, run_id, val_records, best_targets, best_preds)
            else:
                stale_epochs += 1
                if stale_epochs >= QUALITY_PATIENCE:
                    emit(
                        f"[quality run {run_id}] early stopping after {stale_epochs} stale epochs"
                    )
                    break

        if best_state is None or best_rmse is None:
            raise RuntimeError("Training did not produce a model")

        best_path = run_dir / "best.pt"
        torch.save(
            {
                "model_name": QUALITY_BASE_MODEL,
                "img_size": QUALITY_IMG_SIZE,
                "padding_fraction": BBOX_PADDING_FRACTION,
                "state_dict": best_state,
            },
            best_path,
        )

        _log_val_diagnostics(emit, run_id, val_records, best_targets, best_preds)
        emit(f"[quality run {run_id}] done — val_rmse={best_rmse}, model={best_path}")
        db.finish_quality_training_run(run_id, str(best_path), float(best_rmse), str(log_path))
    except Exception as exc:
        trainer_logger.exception("[quality run %s] failed: %s", run_id, exc)
        print(f"[quality run {run_id}] failed: {exc}", flush=True)
        db.fail_quality_training_run(run_id, str(log_path))
    finally:
        handler.close()
        trainer_logger.removeHandler(handler)


class _QualityCropDataset:
    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        src = IMAGES_DIR / rec["image_path"]
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            crop = _crop_box_with_padding(rgb, rec["box"], BBOX_PADDING_FRACTION)
        x = self.transform(crop)

        import torch

        y = torch.tensor(float(rec["quality"]), dtype=torch.float32)
        return x, y


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


def _make_optimizer(model, backbone_lr: float, head_lr: float):
    import torch

    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    return torch.optim.AdamW(
        [
            {"params": head, "lr": head_lr},
            {"params": backbone, "lr": backbone_lr},
        ],
        weight_decay=QUALITY_WEIGHT_DECAY,
    )


def _collect_quality_records(taxon: str) -> list[dict]:
    model_path = db.get_active_model_path_for_taxon(taxon)
    if model_path is None:
        raise ValueError(
            "No active object-detection model; activate a finished detection training run first"
        )

    quality_map = db.get_image_quality_map(taxon)
    records = []
    for image_path, quality in quality_map.items():
        src = IMAGES_DIR / image_path
        if not src.is_file():
            print(f"[quality] skip missing image: {image_path}", flush=True)
            continue
        try:
            with Image.open(src) as img:
                img.verify()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            print(f"[quality] skip unreadable image {image_path}: {exc}", flush=True)
            continue

        boxes = predict_boxes(model_path, src, conf=DETECTION_CONF_THRESHOLD)
        box = quality_crop_box(boxes, DETECTION_CONF_THRESHOLD)
        if box is None:
            print(f"[quality] skip no detection: {image_path}", flush=True)
            continue

        records.append({
            "image_path": image_path,
            "quality": float(quality),
            "box": box,
        })
    records.sort(key=lambda r: r["image_path"])
    return records


def _split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    data = records[:]
    random.Random(42).shuffle(data)
    split = int(len(data) * 0.8)
    split = max(1, min(len(data) - 1, split))
    return data[:split], data[split:]


def _evaluate(model, dataloader, device) -> tuple[float, list[float], list[float]]:
    import torch

    model.eval()
    sq_error_sum = 0.0
    count = 0
    targets: list[float] = []
    preds: list[float] = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).squeeze(1)
            sq_error_sum += float(((pred - y) ** 2).sum().detach().cpu().item())
            count += x.size(0)
            targets.extend(float(v) for v in y.detach().cpu().tolist())
            preds.extend(float(v) for v in pred.detach().cpu().tolist())
    mse = sq_error_sum / max(1, count)
    return math.sqrt(mse), targets, preds


def _image_source(image_path: str) -> str:
    parts = image_path.split("/")
    if len(parts) > 1:
        return parts[1]
    return "unknown"


def _format_counter(counter: Counter, ordered_keys=None) -> str:
    items: list[tuple[object, int]] = []
    seen = set()
    if ordered_keys is not None:
        for key in ordered_keys:
            items.append((key, int(counter.get(key, 0))))
            seen.add(key)
    for key, value in sorted(counter.items(), key=lambda kv: str(kv[0])):
        if key not in seen:
            items.append((key, int(value)))
    return " ".join(f"{key}={value}" for key, value in items)


def _log_split_stats(emit, run_id: int, split_name: str, records: list[dict]) -> None:
    ratings = Counter(_nearest_rating(float(r["quality"])) for r in records)
    sources = Counter(_image_source(r["image_path"]) for r in records)
    emit(
        f"[quality run {run_id}] {split_name} ratings: {_format_counter(ratings, QUALITY_RATINGS)}"
    )
    emit(
        f"[quality run {run_id}] {split_name} sources: {_format_counter(sources)}"
    )


def _nearest_rating(value: float) -> float:
    return min(QUALITY_RATINGS, key=lambda rating: abs(rating - value))


def _group_mae(pairs: list[tuple[float, float]]) -> tuple[float, int]:
    if not pairs:
        return 0.0, 0
    mae = sum(abs(pred - target) for pred, target in pairs) / len(pairs)
    return mae, len(pairs)


def _log_val_diagnostics(
    emit,
    run_id: int,
    records: list[dict],
    targets: list[float],
    preds: list[float],
) -> None:
    if len(targets) != len(preds) or len(targets) != len(records):
        emit(f"[quality run {run_id}] val diagnostics skipped (length mismatch)")
        return

    overall_mae, overall_n = _group_mae(list(zip(preds, targets)))
    emit(f"[quality run {run_id}] val mae={overall_mae:.6f} n={overall_n}")

    by_rating: dict[float, list[tuple[float, float]]] = {r: [] for r in QUALITY_RATINGS}
    for pred, target in zip(preds, targets):
        by_rating[_nearest_rating(target)].append((pred, target))
    rating_parts = []
    for rating in QUALITY_RATINGS:
        mae, n = _group_mae(by_rating[rating])
        rating_parts.append(f"{rating}={mae:.3f}(n={n})")
    emit(f"[quality run {run_id}] val mae by rating: {' '.join(rating_parts)}")

    by_source: dict[str, list[tuple[float, float]]] = {}
    for rec, pred, target in zip(records, preds, targets):
        source = _image_source(rec["image_path"])
        by_source.setdefault(source, []).append((pred, target))
    source_parts = []
    for source in sorted(by_source):
        mae, n = _group_mae(by_source[source])
        source_parts.append(f"{source}={mae:.3f}(n={n})")
    emit(f"[quality run {run_id}] val mae by source: {' '.join(source_parts)}")

    labels = "/".join(str(r) for r in QUALITY_RATINGS)
    emit(f"[quality run {run_id}] val confusion (true\\pred {labels}):")
    for true_rating in QUALITY_RATINGS:
        counts = []
        for pred_rating in QUALITY_RATINGS:
            n = sum(
                1
                for pred, target in zip(preds, targets)
                if _nearest_rating(target) == true_rating
                and _nearest_rating(pred) == pred_rating
            )
            counts.append(str(n))
        emit(f"[quality run {run_id}]   {true_rating}: {' '.join(counts)}")
