"""
Train a genus identification model for a project (v2, decoupled CLI).

Expects OD-cropped images from scripts/crop_identification_images.py under
trainer/images_processed/<project>/ (same layout as trainer/images/).

Hardcoded parameters — edit constants below, then:

    uv run python scripts/crop_identification_images.py   # once / incremental
    uv run python scripts/train_identification.py

Never modifies originals under trainer/images/. Training artifacts go to
trainer/models/<project>/identification/<run_id>/.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split

from trainer import db
from trainer.harmonize import harmonization_path, read_harmonization
from trainer.images import IMAGE_EXTS

# ---------------------------------------------------------------------------
# Hardcoded parameters
# ---------------------------------------------------------------------------

PROJECT = "auchenorrhyncha"
MIN_IMAGES_PER_CLASS = 10
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
TEST_FRAC = 0.10
SEED = 42

# Evaluation image size (used for val/test).
IMG_SIZE = 384
# Training image size (used for train, to speed up CPU augmentation + training).
TRAIN_IMG_SIZE = 384
BATCH_SIZE = 128  # drop to 64 if MPS runs out of memory at 384
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 60
UNFREEZE_LEAF_MODULES = 60  # later MBConv blocks; keep head unfrozen too
LR_HEAD = 1e-3
LR_FINETUNE_HEAD = 1e-4
LR_FINETUNE_BACKBONE = 1e-5
LR_FINETUNE_MIN = 1e-6
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
PATIENCE = 8
WORST_CLASSES_TO_LOG = 20
# Keep 0 when images are cached in RAM. DataLoader workers on MPS would copy
# the cache per process and fight the GPU for unified memory.
NUM_WORKERS = 0
CACHE_IMAGES = True
# For RAM cache only: pre-resize training crops so augmentation runs faster
# and cache uses less memory. Validation/test already get letterboxed to
# IMG_SIZE at cache time, so only train benefits from this.
CACHE_TRAIN_RESIZE_FACTOR = 1.2
CACHE_EVAL_AS_UINT8 = True
BASE_MODEL = "tf_efficientnetv2_s.in21k"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "trainer" / "models"
PROCESSED_DIR = PROJECT_ROOT / "trainer" / "images_processed"

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
# Labels from processed crops
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


def list_processed_image_paths(project: str) -> list[str]:
    """Paths relative to PROCESSED_DIR, e.g. project/collection/taxon/file.jpg."""
    project_dir = PROCESSED_DIR / project
    if not project_dir.is_dir():
        return []
    paths = []
    for f in project_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            rel = f.relative_to(PROCESSED_DIR)
            paths.append(str(rel).replace("\\", "/"))
    paths.sort()
    return paths


def collect_labeled_paths(project: str) -> list[tuple[str, str]]:
    """Return (processed_rel_path, genus) for harmonized cropped images."""
    folder_to_genus = build_folder_to_genus(project)
    labeled: list[tuple[str, str]] = []
    for rel in list_processed_image_paths(project):
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


def labeled_to_records(labeled: list[tuple[str, str]]) -> list[dict]:
    return [{"crop_path": path, "genus": genus} for path, genus in labeled]


def inverse_sqrt_sample_weights(records: list[dict]) -> list[float]:
    """Per-example weights 1/sqrt(n_class). Softens 10–1000× imbalance vs 1/n."""
    counts = Counter(r["genus"] for r in records)
    class_w = {g: 1.0 / (n ** 0.5) for g, n in counts.items()}
    return [class_w[r["genus"]] for r in records]


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


def frozen_splits_path(project: str) -> Path:
    return MODELS_DIR / project / "identification" / "splits.json"


def splits_payload(splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {
        k: [{"crop_path": r["crop_path"], "genus": r["genus"]} for r in v]
        for k, v in splits.items()
    }


def write_splits(path: Path, splits: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits_payload(splits), indent=2) + "\n", encoding="utf-8")


def load_splits(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, list[dict]] = {}
    for key in ("train", "val", "test"):
        if key not in data:
            raise ValueError(f"{path} missing '{key}'")
        splits[key] = [
            {"crop_path": r["crop_path"], "genus": r["genus"]} for r in data[key]
        ]
    return splits


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
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle90 = random.choice([0, 90, 180, 270])
    if angle90:
        img = img.rotate(angle90, expand=True)
    small = random.uniform(-30.0, 30.0)
    if abs(small) > 0.5:
        img = img.rotate(small, expand=True, fillcolor=(0, 0, 0))
    scale = random.uniform(0.85, 1.15)
    if abs(scale - 1.0) > 0.01:
        w, h = img.size
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.8:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    return img


def _pil_to_normalized_tensor(img: Image.Image, img_size: int):
    import torch

    boxed = letterbox(img, img_size, fill=0)
    arr = np.asarray(boxed, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr)


def _pil_to_uint8_tensor(img: Image.Image, img_size: int):
    import torch

    boxed = letterbox(img, img_size, fill=0)
    # Keep as uint8 to shrink RAM cache; normalize later on-device.
    arr = np.asarray(boxed, dtype=np.uint8).transpose(2, 0, 1)
    return torch.from_numpy(arr)


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return img
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.BILINEAR)


def _maybe_normalize_on_device(x):
    """If x is a cached uint8 image tensor (BCHW), normalize it for ImageNet."""
    import torch

    if x.dtype != torch.uint8:
        return x
    x = x.to(dtype=torch.float32).div(255.0)
    mean = torch.as_tensor(IMAGENET_MEAN, device=x.device, dtype=torch.float32)
    std = torch.as_tensor(IMAGENET_STD, device=x.device, dtype=torch.float32)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Focal loss / model helpers
# ---------------------------------------------------------------------------


class FocalLoss:
    """Multiclass focal loss (gamma, alpha). Callable like nn.Module."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, logits, targets):
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
        records: list[dict],
        class_to_idx: dict[str, int],
        train: bool,
        img_size: int = IMG_SIZE,
        cache_images: bool = CACHE_IMAGES,
        cache_train_resize_factor: float = CACHE_TRAIN_RESIZE_FACTOR,
        cache_eval_as_uint8: bool = CACHE_EVAL_AS_UINT8,
        logger: logging.Logger | None = None,
    ):
        self.records = records
        self.class_to_idx = class_to_idx
        self.train = train
        self.img_size = img_size
        self.cache_train_resize_factor = cache_train_resize_factor
        self.cache_eval_as_uint8 = cache_eval_as_uint8
        self.cache = None
        if cache_images:
            self.cache = self._build_cache(logger)

    def _build_cache(self, logger: logging.Logger | None):
        cached = []
        n = len(self.records)
        for i, rec in enumerate(self.records, start=1):
            if logger and (i == 1 or i % 2000 == 0 or i == n):
                split = "train" if self.train else "eval"
                _log(logger, f"caching {split} images {i}/{n}")
            path = PROCESSED_DIR / rec["crop_path"]
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                if self.train:
                    # Pre-resize so augmentation runs on a smaller bitmap.
                    max_side = max(1, int(round(self.img_size * self.cache_train_resize_factor)))
                    rgb = _resize_max_side(rgb, max_side)
                    cached.append(rgb.copy())
                else:
                    if self.cache_eval_as_uint8:
                        cached.append(_pil_to_uint8_tensor(rgb, self.img_size))
                    else:
                        cached.append(_pil_to_normalized_tensor(rgb, self.img_size))
        if logger:
            split = "train" if self.train else "eval"
            if self.train:
                pixels = sum(im.width * im.height * 3 for im in cached)
                _log(logger, f"cached {n} {split} RGB crops (~{pixels / 1e9:.2f} GB uncompressed)")
            else:
                kind = "uint8" if self.cache_eval_as_uint8 else "float32 normalized"
                _log(logger, f"cached {n} {split} tensors ({kind})")
        return cached

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, index: int) -> Image.Image:
        if self.cache is not None:
            return self.cache[index]
        rec = self.records[index]
        with Image.open(PROCESSED_DIR / rec["crop_path"]) as img:
            return img.convert("RGB")

    def __getitem__(self, index: int):
        rec = self.records[index]
        if self.train:
            rgb = self._load_rgb(index)
            if self.cache is not None:
                rgb = rgb.copy()
            rgb = _train_augment(rgb)
            arr = _pil_to_normalized_tensor(rgb, self.img_size)
        elif self.cache is not None:
            arr = self.cache[index]
        else:
            arr = _pil_to_normalized_tensor(self._load_rgb(index), self.img_size)
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


def worst_class_recalls(
    y_true: list[int],
    y_pred: list[int],
    idx_to_class: dict[int, str],
    k: int = WORST_CLASSES_TO_LOG,
) -> list[dict]:
    rows = []
    for c, name in idx_to_class.items():
        support = sum(1 for t in y_true if t == c)
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        recall = tp / support if support else 0.0
        rows.append({"genus": name, "recall": recall, "support": support})
    rows.sort(key=lambda r: (r["recall"], r["genus"]))
    return rows[:k]


def _log_worst_recalls(logger: logging.Logger, split: str, rows: list[dict]) -> None:
    parts = [f"{r['genus']}={r['recall']:.2f}(n={r['support']})" for r in rows]
    _log(logger, f"worst {len(rows)} {split} recalls: " + ", ".join(parts))


def evaluate(model, loader, criterion, device, num_classes: int) -> tuple[dict, list[int], list[int]]:
    import torch

    model.eval()
    loss_sum = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x = _maybe_normalize_on_device(x)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    metrics = {
        "loss": loss_sum / n if n else 0.0,
        "top1": top1,
        "macro_f1": _macro_f1(y_true, y_pred, num_classes),
    }
    return metrics, y_true, y_pred


def _hyperparams() -> dict:
    return {
        "batch_size": BATCH_SIZE,
        "img_size_train": TRAIN_IMG_SIZE,
        "img_size_eval": IMG_SIZE,
        "head_epochs": HEAD_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "unfreeze_leaf_modules": UNFREEZE_LEAF_MODULES,
        "lr_head": LR_HEAD,
        "lr_finetune_head": LR_FINETUNE_HEAD,
        "lr_finetune_backbone": LR_FINETUNE_BACKBONE,
        "lr_finetune_min": LR_FINETUNE_MIN,
        "weight_decay": WEIGHT_DECAY,
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": FOCAL_ALPHA,
        "patience": PATIENCE,
        "seed": SEED,
        "sampler": "inverse_sqrt",
    }


def save_best_checkpoint(
    path: Path,
    model,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    val_metrics: dict,
    test_metrics: dict | None = None,
) -> None:
    import torch

    payload = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "base_model": BASE_MODEL,
        "img_size_train": TRAIN_IMG_SIZE,
        "img_size_eval": IMG_SIZE,
        "project": PROJECT,
        "hyperparams": _hyperparams(),
        "val_metrics": val_metrics,
    }
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics
    torch.save(payload, path)


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
    scheduler=None,
    run_dir: Path | None = None,
    class_to_idx: dict[str, int] | None = None,
    idx_to_class: dict[int, str] | None = None,
) -> tuple[dict | None, float, int]:
    stale = 0
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        loss_sum = 0.0
        n = 0
        t_data = 0.0
        t_model = 0.0
        t0 = time.perf_counter()
        for x, y in train_loader:
            t_data += time.perf_counter() - t0
            t1 = time.perf_counter()
            x = x.to(device)
            x = _maybe_normalize_on_device(x)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
            t_model += time.perf_counter() - t1
            t0 = time.perf_counter()

        if scheduler is not None:
            scheduler.step()

        train_loss = loss_sum / n if n else 0.0
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, num_classes)
        lr_msg = ""
        if scheduler is not None:
            lrs = [g["lr"] for g in optimizer.param_groups]
            lr_msg = " " + " ".join(f"lr{i}={lr:.2e}" for i, lr in enumerate(lrs))
        _log(
            logger,
            f"[{phase}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            f" data={t_data:.1f}s model={t_model:.1f}s"
            f"{lr_msg}",
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if run_dir is not None and class_to_idx is not None and idx_to_class is not None:
                save_best_checkpoint(
                    run_dir / "best.pt",
                    model,
                    class_to_idx,
                    idx_to_class,
                    val_metrics,
                )
                _log(logger, f"[{phase}] saved best.pt (val_macro_f1={best_f1:.4f})")
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
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import timm

    genera = sorted({r["genus"] for split in splits.values() for r in split})
    class_to_idx = {g: i for i, g in enumerate(genera)}
    idx_to_class = {i: g for g, i in class_to_idx.items()}
    num_classes = len(genera)

    label_map_path = run_dir / "label_map.json"
    label_map_path.write_text(
        json.dumps(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _log(logger, f"caching images in RAM (CACHE_IMAGES={CACHE_IMAGES})")
    train_ds = CropDataset(
        splits["train"],
        class_to_idx,
        train=True,
        img_size=TRAIN_IMG_SIZE,
        logger=logger,
    )
    # Val/test always use eval size.
    val_ds = CropDataset(splits["val"], class_to_idx, train=False, img_size=IMG_SIZE, logger=logger)
    test_ds = CropDataset(splits["test"], class_to_idx, train=False, img_size=IMG_SIZE, logger=logger)

    sample_weights = inverse_sqrt_sample_weights(splits["train"])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    _log(logger, "train sampler=inverse_sqrt (WeightedRandomSampler)")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    device = pick_device()
    _log(
        logger,
        f"device={device}, num_classes={num_classes}, "
        f"batch_size={BATCH_SIZE}, img_size_train={TRAIN_IMG_SIZE}, img_size_eval={IMG_SIZE}, base_model={BASE_MODEL}",
    )

    model = timm.create_model(BASE_MODEL, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)

    epoch_kwargs = {
        "run_dir": run_dir,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }

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
        **epoch_kwargs,
    )

    # Phase B: unfreeze last leaf modules; lower LR on backbone than head
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

    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_FINETUNE_BACKBONE},
            {"params": head_params, "lr": LR_FINETUNE_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FINETUNE_EPOCHS,
        eta_min=LR_FINETUNE_MIN,
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
        scheduler=scheduler,
        **epoch_kwargs,
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, val_true, val_pred = evaluate(
        model, val_loader, criterion, device, num_classes
    )
    test_metrics, test_true, test_pred = evaluate(
        model, test_loader, criterion, device, num_classes
    )
    val_worst = worst_class_recalls(val_true, val_pred, idx_to_class)
    test_worst = worst_class_recalls(test_true, test_pred, idx_to_class)
    _log(
        logger,
        f"final val_top1={val_metrics['top1']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_top1={test_metrics['top1']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}",
    )
    _log_worst_recalls(logger, "val", val_worst)
    _log_worst_recalls(logger, "test", test_worst)

    save_best_checkpoint(
        run_dir / "best.pt",
        model,
        class_to_idx,
        idx_to_class,
        val_metrics,
        test_metrics=test_metrics,
    )

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "val_worst_recall": val_worst,
        "test_worst_recall": test_worst,
        "num_classes": num_classes,
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
        "best_val_macro_f1": best_f1,
        "hyperparams": _hyperparams(),
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

    rank = project["identification_rank"]
    if rank != "genus":
        print(f"warning: identification_rank is '{rank}', script trains genus labels")

    processed_root = PROCESSED_DIR / PROJECT
    if not processed_root.is_dir():
        raise SystemExit(
            f"No processed images at {processed_root}. "
            "Run: uv run python scripts/crop_identification_images.py"
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = MODELS_DIR / PROJECT / "identification" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(run_dir)
    _log(logger, f"run_dir={run_dir}")
    _log(logger, f"project={PROJECT} identification_rank={rank}")
    _log(logger, f"processed_dir={processed_root}")

    frozen_path = frozen_splits_path(PROJECT)
    if frozen_path.is_file():
        splits = load_splits(frozen_path)
        _log(logger, f"using frozen splits {frozen_path}")
    else:
        labeled = collect_labeled_paths(PROJECT)
        _log(logger, f"harmonized processed images: {len(labeled)}")
        labeled = filter_by_min_count(labeled, MIN_IMAGES_PER_CLASS)
        genus_counts = Counter(g for _, g in labeled)
        _log(
            logger,
            f"after min_count>={MIN_IMAGES_PER_CLASS}: "
            f"images={len(labeled)} genera={len(genus_counts)}",
        )
        if len(labeled) < 30 or len(genus_counts) < 2:
            raise SystemExit(
                "Not enough labeled processed images/classes to train. "
                "Run crop_identification_images.py first if crops are missing."
            )

        records = labeled_to_records(labeled)

        # Stratified split needs at least 2 samples per class for 3-way split.
        min_for_split = 3
        too_small = {g for g, n in genus_counts.items() if n < min_for_split}
        if too_small:
            records = [r for r in records if r["genus"] not in too_small]
            _log(logger, f"dropped genera with <{min_for_split} images: {sorted(too_small)}")

        splits = stratified_split(records, seed=SEED)
        write_splits(frozen_path, splits)
        _log(logger, f"wrote frozen splits {frozen_path}")

    write_splits(run_dir / "splits.json", splits)
    _log(
        logger,
        f"split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
    )

    metrics = train_model(run_dir, splits, logger)
    _log(logger, f"done. metrics={json.dumps(metrics)}")


if __name__ == "__main__":
    main()
