"""
Train a genus identification model for a project (v1, decoupled CLI).

Hardcoded parameters — edit constants below, then:

    uv run python scripts/train_identification.py

Never modifies originals under trainer/images/. Crops and artifacts go to
trainer/models/<project>/identification/<run_id>/.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from trainer import db
from trainer.harmonize import harmonization_path, read_harmonization
from trainer.images import IMAGES_DIR, list_project_image_paths
from trainer.inference import predict_top_box

# ---------------------------------------------------------------------------
# Hardcoded parameters
# ---------------------------------------------------------------------------

PROJECT = "auchenorrhyncha"
MIN_IMAGES_PER_CLASS = 10
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10
SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 64
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 12
UNFREEZE_LEAF_MODULES = 40  # within 30–60; later MBConv / classifier leaves
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
PATIENCE = 4
NUM_WORKERS = 0
BBOX_PADDING_FRACTION = 0.10
BASE_MODEL = "tf_efficientnetv2_s.in21k"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "trainer" / "models"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _setup_logging(run_dir: Path) -> logging.Logger:
    log_path = run_dir / "train.log"
    logger = logging.getLogger("train_identification")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _log(logger: logging.Logger, msg: str) -> None:
    logger.info(msg)


# ---------------------------------------------------------------------------
# Label / crop dataset building
# ---------------------------------------------------------------------------


def genus_from_authoritative(authoritative_name: str) -> str | None:
    name = (authoritative_name or "").strip()
    if not name:
        return None
    return name.split()[0]


def folder_name_from_image_path(rel_path: str) -> str:
    return Path(rel_path).parent.name


def build_folder_to_genus(project: str) -> dict[str, str]:
    rows = read_harmonization(harmonization_path(project)) or []
    mapping: dict[str, str] = {}
    for row in rows:
        genus = genus_from_authoritative(row.get("authoritative_name", ""))
        image_name = row.get("image_name") or ""
        if genus and image_name:
            mapping[image_name] = genus
    return mapping


def collect_labeled_paths(project: str) -> list[tuple[str, str]]:
    """Return (relative_image_path, genus) for harmonized images."""
    folder_to_genus = build_folder_to_genus(project)
    labeled: list[tuple[str, str]] = []
    for rel in list_project_image_paths(project):
        folder = folder_name_from_image_path(rel)
        genus = folder_to_genus.get(folder)
        if genus is None:
            continue
        labeled.append((rel, genus))
    return labeled


def filter_by_min_count(
    labeled: list[tuple[str, str]],
    min_count: int,
) -> list[tuple[str, str]]:
    counts = Counter(genus for _, genus in labeled)
    keep = {g for g, n in counts.items() if n >= min_count}
    return [(path, genus) for path, genus in labeled if genus in keep]


def crop_filename_for_source(rel_path: str) -> str:
    """Stable unique filename from relative path (no path separators)."""
    stem = re.sub(r"[^\w.\-]+", "_", rel_path.replace("/", "__"))
    if not stem.lower().endswith((".jpg", ".jpeg", ".png")):
        stem = stem + ".jpg"
    return stem


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


def build_crop_dataset(
    project: str,
    run_dir: Path,
    labeled: list[tuple[str, str]],
    logger: logging.Logger,
) -> list[dict]:
    """
    Run OD on each image, save padded crops under run_dir/crops/, return records.
    Never writes into trainer/images/.
    """
    model_path = db.get_active_model_path_for_taxon(project)
    if model_path is None:
        raise SystemExit(
            f"No active object-detection model for project '{project}'. "
            "Activate a finished OD training run first."
        )

    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = crops_dir / "manifest.jsonl"

    records: list[dict] = []
    skipped_no_box = 0
    skipped_bad_image = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for i, (rel_path, genus) in enumerate(labeled, start=1):
            if i % 100 == 0 or i == 1:
                _log(logger, f"cropping {i}/{len(labeled)} …")

            abs_path = IMAGES_DIR / rel_path
            boxes = predict_top_box(model_path, abs_path)
            if not boxes:
                skipped_no_box += 1
                continue

            try:
                with Image.open(abs_path) as img:
                    rgb = img.convert("RGB")
                    crop = crop_box_with_padding(rgb, boxes[0], BBOX_PADDING_FRACTION)
            except (OSError, UnidentifiedImageError):
                skipped_bad_image += 1
                continue

            out_name = crop_filename_for_source(rel_path)
            out_rel = f"crops/{genus}/{out_name}"
            out_abs = run_dir / out_rel
            out_abs.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_abs, format="JPEG", quality=95)

            rec = {
                "source_path": rel_path,
                "genus": genus,
                "crop_path": out_rel,
            }
            records.append(rec)
            manifest.write(json.dumps(rec) + "\n")

    _log(
        logger,
        f"crops done: kept={len(records)}, "
        f"skipped_no_box={skipped_no_box}, skipped_bad_image={skipped_bad_image}",
    )
    return records


def refilter_records_after_crop(
    records: list[dict],
    min_count: int,
) -> list[dict]:
    """Drop genera that fell below min_count after OD skips."""
    counts = Counter(r["genus"] for r in records)
    keep = {g for g, n in counts.items() if n >= min_count}
    return [r for r in records if r["genus"] in keep]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def stratified_split(
    records: list[dict],
    seed: int = SEED,
) -> dict[str, list[dict]]:
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9
    labels = [r["genus"] for r in records]
    train_recs, rest_recs, _, rest_labels = train_test_split(
        records,
        labels,
        test_size=(VAL_FRAC + TEST_FRAC),
        random_state=seed,
        stratify=labels,
    )
    relative_test = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
    val_recs, test_recs = train_test_split(
        rest_recs,
        test_size=relative_test,
        random_state=seed,
        stratify=rest_labels,
    )
    return {"train": train_recs, "val": val_recs, "test": test_recs}


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def letterbox(img: Image.Image, size: int, fill: int = 0) -> Image.Image:
    """Resize preserving aspect ratio and pad to a square of side `size`."""
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


def _train_augment(img: Image.Image) -> Image.Image:
    angle = random.choice([0, 90, 180, 270])
    if angle:
        img = img.rotate(angle, expand=True)
    if random.random() < 0.8:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if random.random() < 0.8:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)
    return img


# ---------------------------------------------------------------------------
# Focal loss / model helpers
# ---------------------------------------------------------------------------


class FocalLoss:
    """Multiclass focal loss (gamma, alpha). Callable like nn.Module."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -self.alpha * (1.0 - pt) ** self.gamma * log_pt
        return loss.mean()


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_classifier(model) -> None:
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def leaf_modules_with_params(model) -> list:
    leaves = []
    for module in model.modules():
        if any(module.children()):
            continue
        if any(True for _ in module.parameters(recurse=False)):
            leaves.append(module)
    return leaves


def unfreeze_last_leaf_modules(model, n: int) -> int:
    """Unfreeze the last `n` leaf modules (later blocks + head). Returns count."""
    leaves = leaf_modules_with_params(model)
    chosen = leaves[-n:] if n < len(leaves) else leaves
    for module in chosen:
        for p in module.parameters(recurse=False):
            p.requires_grad = True
    return len(chosen)


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset / training
# ---------------------------------------------------------------------------


class CropDataset:
    def __init__(
        self,
        run_dir: Path,
        records: list[dict],
        class_to_idx: dict[str, int],
        train: bool,
        img_size: int = IMG_SIZE,
    ):
        self.run_dir = run_dir
        self.records = records
        self.class_to_idx = class_to_idx
        self.train = train
        self.img_size = img_size
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        rec = self.records[index]
        path = self.run_dir / rec["crop_path"]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        if self.train:
            rgb = _train_augment(rgb)
        boxed = letterbox(rgb, self.img_size, fill=0)
        arr = torch.from_numpy(
            np.asarray(boxed, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        for c in range(3):
            arr[c] = (arr[c] - self.mean[c]) / self.std[c]
        label = self.class_to_idx[rec["genus"]]
        return arr, label


def _macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / num_classes if num_classes else 0.0


def evaluate(model, loader, criterion, device, num_classes: int) -> dict:
    import torch

    model.eval()
    loss_sum = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    return {
        "loss": loss_sum / n if n else 0.0,
        "top1": top1,
        "macro_f1": _macro_f1(y_true, y_pred, num_classes),
    }


def run_epochs(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_classes: int,
    epochs: int,
    patience: int,
    logger: logging.Logger,
    phase: str,
    best_state: dict | None,
    best_f1: float,
) -> tuple[dict | None, float, int]:
    import torch

    stale = 0
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)

        train_loss = loss_sum / n if n else 0.0
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        _log(
            logger,
            f"[{phase}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}",
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                _log(logger, f"[{phase}] early stop at epoch {epoch} (patience={patience})")
                break

    return best_state, best_f1, last_epoch


def train_model(
    run_dir: Path,
    splits: dict[str, list[dict]],
    logger: logging.Logger,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    import timm

    genera = sorted({r["genus"] for split in splits.values() for r in split})
    class_to_idx = {g: i for i, g in enumerate(genera)}
    idx_to_class = {i: g for g, i in class_to_idx.items()}
    num_classes = len(genera)

    label_map_path = run_dir / "label_map.json"
    label_map_path.write_text(
        json.dumps({"class_to_idx": class_to_idx, "idx_to_class": {str(k): v for k, v in idx_to_class.items()}}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    train_ds = CropDataset(run_dir, splits["train"], class_to_idx, train=True)
    val_ds = CropDataset(run_dir, splits["val"], class_to_idx, train=False)
    test_ds = CropDataset(run_dir, splits["test"], class_to_idx, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    device = pick_device()
    _log(logger, f"device={device}, num_classes={num_classes}, base_model={BASE_MODEL}")

    model = timm.create_model(BASE_MODEL, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)

    # Phase A: head only
    freeze_all(model)
    unfreeze_classifier(model)
    _log(logger, f"phase A: trainable params={count_trainable(model)}")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    best_state, best_f1, _ = run_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_classes,
        HEAD_EPOCHS,
        PATIENCE,
        logger,
        "head",
        None,
        -1.0,
    )

    # Phase B: unfreeze last leaf modules
    freeze_all(model)
    n_unfrozen = unfreeze_last_leaf_modules(model, UNFREEZE_LEAF_MODULES)
    unfreeze_classifier(model)
    _log(
        logger,
        f"phase B: unfroze {n_unfrozen} leaf modules "
        f"(requested {UNFREEZE_LEAF_MODULES}), trainable params={count_trainable(model)}",
    )
    if best_state is not None:
        model.load_state_dict(best_state)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FINETUNE,
        weight_decay=WEIGHT_DECAY,
    )
    best_state, best_f1, _ = run_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_classes,
        FINETUNE_EPOCHS,
        PATIENCE,
        logger,
        "finetune",
        best_state,
        best_f1,
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    _log(
        logger,
        f"final val_top1={val_metrics['top1']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_top1={test_metrics['top1']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}",
    )

    ckpt = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "base_model": BASE_MODEL,
        "img_size": IMG_SIZE,
        "project": PROJECT,
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "head_epochs": HEAD_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "unfreeze_leaf_modules": UNFREEZE_LEAF_MODULES,
            "lr_head": LR_HEAD,
            "lr_finetune": LR_FINETUNE,
            "weight_decay": WEIGHT_DECAY,
            "focal_gamma": FOCAL_GAMMA,
            "focal_alpha": FOCAL_ALPHA,
            "seed": SEED,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    torch.save(ckpt, run_dir / "best.pt")

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "num_classes": num_classes,
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
        "best_val_macro_f1": best_f1,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)

    project = db.get_project(PROJECT)
    if project is None:
        raise SystemExit(f"Project '{PROJECT}' not found in database")

    rank = project.get("identification_rank")
    if rank != "genus":
        print(f"warning: identification_rank is '{rank}', script trains genus labels")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = MODELS_DIR / PROJECT / "identification" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(run_dir)
    _log(logger, f"run_dir={run_dir}")
    _log(logger, f"project={PROJECT} identification_rank={rank}")

    labeled = collect_labeled_paths(PROJECT)
    _log(logger, f"harmonized images: {len(labeled)}")
    labeled = filter_by_min_count(labeled, MIN_IMAGES_PER_CLASS)
    genus_counts = Counter(g for _, g in labeled)
    _log(
        logger,
        f"after min_count>={MIN_IMAGES_PER_CLASS}: "
        f"images={len(labeled)} genera={len(genus_counts)}",
    )
    if len(labeled) < 30 or len(genus_counts) < 2:
        raise SystemExit("Not enough labeled images/classes to train")

    records = build_crop_dataset(PROJECT, run_dir, labeled, logger)
    records = refilter_records_after_crop(records, MIN_IMAGES_PER_CLASS)
    genus_counts = Counter(r["genus"] for r in records)
    _log(
        logger,
        f"after crop + refilter: images={len(records)} genera={len(genus_counts)}",
    )
    if len(records) < 30 or len(genus_counts) < 2:
        raise SystemExit("Not enough cropped images/classes to train")

    # Stratified split needs at least 2 samples per class for 3-way split;
    # drop any class that is too small to stratify into train/val/test.
    min_for_split = 3
    too_small = {g for g, n in genus_counts.items() if n < min_for_split}
    if too_small:
        records = [r for r in records if r["genus"] not in too_small]
        _log(logger, f"dropped genera with <{min_for_split} crops: {sorted(too_small)}")

    splits = stratified_split(records, seed=SEED)
    splits_path = run_dir / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                k: [{"source_path": r["source_path"], "genus": r["genus"], "crop_path": r["crop_path"]} for r in v]
                for k, v in splits.items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _log(
        logger,
        f"split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
    )

    metrics = train_model(run_dir, splits, logger)
    _log(logger, f"done. metrics={json.dumps(metrics)}")


if __name__ == "__main__":
    main()
