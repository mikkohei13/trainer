"""Model helpers, metrics, training loop, and CLI entry."""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from trainer import db

from . import config
from .data import (
    collect_labeled_paths,
    filter_by_min_count,
    filter_labeled_by_quality,
    filter_splits_by_quality,
    frozen_splits_path,
    inverse_sqrt_sample_weights,
    labeled_to_records,
    load_quality_ratings,
    load_splits,
    log,
    quality_ratings_path,
    setup_logging,
    stratified_split,
    write_splits,
)
from .dataset import CropDataset, maybe_normalize_on_device


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


def _macro_f1(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
    skip_empty: bool = False,
) -> float:
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if skip_empty and (tp + fn) == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    denom = len(f1s) if skip_empty else num_classes
    return sum(f1s) / denom if denom else 0.0


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
    skip_empty: bool = False,
) -> dict:
    n = len(y_true)
    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    present = {t for t in y_true}
    return {
        "top1": top1,
        "macro_f1": _macro_f1(y_true, y_pred, num_classes, skip_empty=skip_empty),
        "n": n,
        "num_classes": len(present),
    }


def filter_eval_by_quality(
    records: list[dict],
    y_true: list[int],
    y_pred: list[int],
    ratings: dict[str, float],
    min_quality: float,
) -> tuple[list[int], list[int]]:
    """Keep eval pairs whose saved quality is at or above min_quality."""
    kept_true: list[int] = []
    kept_pred: list[int] = []
    for rec, t, p in zip(records, y_true, y_pred):
        score = ratings.get(rec["crop_path"])
        if score is not None and score >= min_quality:
            kept_true.append(t)
            kept_pred.append(p)
    return kept_true, kept_pred


def class_recall_rows(
    y_true: list[int],
    y_pred: list[int],
    idx_to_class: dict[int, str],
    min_support: int | None = None,
) -> list[dict]:
    if min_support is None:
        min_support = config.MIN_RECALL_SUPPORT
    rows = []
    for c, name in idx_to_class.items():
        support = sum(1 for t in y_true if t == c)
        if support < min_support:
            continue
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        rows.append({"genus": name, "recall": tp / support, "support": support})
    return rows


def worst_class_recalls(
    y_true: list[int],
    y_pred: list[int],
    idx_to_class: dict[int, str],
    k: int | None = None,
    min_support: int | None = None,
) -> list[dict]:
    if k is None:
        k = config.WORST_CLASSES_TO_LOG
    rows = class_recall_rows(y_true, y_pred, idx_to_class, min_support=min_support)
    rows.sort(key=lambda r: (r["recall"], -r["support"], r["genus"]))
    return rows[:k]


def best_class_recalls(
    y_true: list[int],
    y_pred: list[int],
    idx_to_class: dict[int, str],
    k: int | None = None,
    min_support: int | None = None,
) -> list[dict]:
    if k is None:
        k = config.WORST_CLASSES_TO_LOG
    rows = class_recall_rows(y_true, y_pred, idx_to_class, min_support=min_support)
    rows.sort(key=lambda r: (-r["recall"], -r["support"], r["genus"]))
    return rows[:k]


def _log_class_recalls(
    logger: logging.Logger,
    kind: str,
    split: str,
    rows: list[dict],
    min_support: int | None = None,
) -> None:
    if min_support is None:
        min_support = config.MIN_RECALL_SUPPORT
    if not rows:
        log(logger, f"{kind} {split} recalls (n>={min_support}): (none)")
        return
    parts = [f"{r['genus']}={r['recall']:.2f}(n={r['support']})" for r in rows]
    log(
        logger,
        f"{kind} {len(rows)} {split} recalls (n>={min_support}): " + ", ".join(parts),
    )


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
            x = maybe_normalize_on_device(x)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    cls = classification_metrics(y_true, y_pred, num_classes)
    metrics = {
        "loss": loss_sum / n if n else 0.0,
        "top1": cls["top1"],
        "macro_f1": cls["macro_f1"],
    }
    return metrics, y_true, y_pred


def _hyperparams() -> dict:
    return {
        "batch_size": config.BATCH_SIZE,
        "img_size_train": config.TRAIN_IMG_SIZE,
        "img_size_eval": config.IMG_SIZE,
        "head_epochs": config.HEAD_EPOCHS,
        "finetune_epochs": config.FINETUNE_EPOCHS,
        "unfreeze_leaf_modules": config.UNFREEZE_LEAF_MODULES,
        "lr_head": config.LR_HEAD,
        "lr_finetune_head": config.LR_FINETUNE_HEAD,
        "lr_finetune_backbone": config.LR_FINETUNE_BACKBONE,
        "lr_finetune_min": config.LR_FINETUNE_MIN,
        "weight_decay": config.WEIGHT_DECAY,
        "focal_gamma": config.FOCAL_GAMMA,
        "focal_alpha": config.FOCAL_ALPHA,
        "patience": config.PATIENCE,
        "seed": config.SEED,
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
        "base_model": config.BASE_MODEL,
        "img_size_train": config.TRAIN_IMG_SIZE,
        "img_size_eval": config.IMG_SIZE,
        "project": config.PROJECT,
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
            x = maybe_normalize_on_device(x)
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
        log(
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
                log(logger, f"[{phase}] saved best.pt (val_macro_f1={best_f1:.4f})")
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log(logger, f"[{phase}] early stop at epoch {epoch} (patience={patience})")
                break

    return best_state, best_f1, last_epoch


def _high_quality_report(
    records: list[dict],
    y_true: list[int],
    y_pred: list[int],
    ratings: dict[str, float],
    idx_to_class: dict[int, str],
    num_classes: int,
) -> dict:
    hq_true, hq_pred = filter_eval_by_quality(
        records, y_true, y_pred, ratings, config.HIGH_QUALITY
    )
    metrics = classification_metrics(hq_true, hq_pred, num_classes, skip_empty=True)
    return {
        **metrics,
        "worst_recall": worst_class_recalls(hq_true, hq_pred, idx_to_class),
        "best_recall": best_class_recalls(hq_true, hq_pred, idx_to_class),
    }


def _log_split_recalls(
    logger: logging.Logger,
    split: str,
    y_true: list[int],
    y_pred: list[int],
    idx_to_class: dict[int, str],
    hq: dict,
) -> tuple[list[dict], list[dict]]:
    worst = worst_class_recalls(y_true, y_pred, idx_to_class)
    best = best_class_recalls(y_true, y_pred, idx_to_class)
    _log_class_recalls(logger, "worst", split, worst)
    _log_class_recalls(logger, "best", split, best)
    _log_class_recalls(logger, "worst", f"{split} high_quality", hq["worst_recall"])
    _log_class_recalls(logger, "best", f"{split} high_quality", hq["best_recall"])
    return worst, best


def train_model(
    run_dir: Path,
    splits: dict[str, list[dict]],
    logger: logging.Logger,
    ratings: dict[str, float],
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

    log(logger, f"caching images in RAM (CACHE_IMAGES={config.CACHE_IMAGES})")
    train_ds = CropDataset(
        splits["train"],
        class_to_idx,
        train=True,
        img_size=config.TRAIN_IMG_SIZE,
        logger=logger,
    )
    # Val/test always use eval size.
    val_ds = CropDataset(
        splits["val"], class_to_idx, train=False, img_size=config.IMG_SIZE, logger=logger
    )
    test_ds = CropDataset(
        splits["test"], class_to_idx, train=False, img_size=config.IMG_SIZE, logger=logger
    )

    sample_weights = inverse_sqrt_sample_weights(splits["train"])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    log(logger, "train sampler=inverse_sqrt (WeightedRandomSampler)")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )

    device = pick_device()
    log(
        logger,
        f"device={device}, num_classes={num_classes}, "
        f"batch_size={config.BATCH_SIZE}, img_size_train={config.TRAIN_IMG_SIZE}, "
        f"img_size_eval={config.IMG_SIZE}, base_model={config.BASE_MODEL}",
    )

    model = timm.create_model(config.BASE_MODEL, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    criterion = FocalLoss(gamma=config.FOCAL_GAMMA, alpha=config.FOCAL_ALPHA)

    epoch_kwargs = {
        "run_dir": run_dir,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }

    # Phase A: head only
    freeze_all(model)
    unfreeze_classifier(model)
    log(logger, f"phase A: trainable params={count_trainable(model)}")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR_HEAD,
        weight_decay=config.WEIGHT_DECAY,
    )
    best_state, best_f1, _ = run_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_classes,
        config.HEAD_EPOCHS,
        config.PATIENCE,
        logger,
        "head",
        None,
        -1.0,
        **epoch_kwargs,
    )

    # Phase B: unfreeze last leaf modules; lower LR on backbone than head
    freeze_all(model)
    n_unfrozen = unfreeze_last_leaf_modules(model, config.UNFREEZE_LEAF_MODULES)
    unfreeze_classifier(model)
    log(
        logger,
        f"phase B: unfroze {n_unfrozen} leaf modules "
        f"(requested {config.UNFREEZE_LEAF_MODULES}), trainable params={count_trainable(model)}",
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
            {"params": backbone_params, "lr": config.LR_FINETUNE_BACKBONE},
            {"params": head_params, "lr": config.LR_FINETUNE_HEAD},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.FINETUNE_EPOCHS,
        eta_min=config.LR_FINETUNE_MIN,
    )
    best_state, best_f1, _ = run_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_classes,
        config.FINETUNE_EPOCHS,
        config.PATIENCE,
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
    val_hq = _high_quality_report(
        splits["val"], val_true, val_pred, ratings, idx_to_class, num_classes
    )
    test_hq = _high_quality_report(
        splits["test"], test_true, test_pred, ratings, idx_to_class, num_classes
    )
    log(
        logger,
        f"final val_top1={val_metrics['top1']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_top1={test_metrics['top1']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}",
    )
    log(
        logger,
        f"high_quality (>={config.HIGH_QUALITY}) "
        f"val_n={val_hq['n']} val_classes={val_hq['num_classes']} "
        f"val_top1={val_hq['top1']:.4f} val_macro_f1={val_hq['macro_f1']:.4f} "
        f"test_n={test_hq['n']} test_classes={test_hq['num_classes']} "
        f"test_top1={test_hq['top1']:.4f} test_macro_f1={test_hq['macro_f1']:.4f}",
    )
    val_worst, val_best = _log_split_recalls(
        logger, "val", val_true, val_pred, idx_to_class, val_hq
    )
    test_worst, test_best = _log_split_recalls(
        logger, "test", test_true, test_pred, idx_to_class, test_hq
    )

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
        "val_best_recall": val_best,
        "test_worst_recall": test_worst,
        "test_best_recall": test_best,
        "val_high_quality": val_hq,
        "test_high_quality": test_hq,
        "high_quality_threshold": config.HIGH_QUALITY,
        "num_classes": num_classes,
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
        "best_val_macro_f1": best_f1,
        "hyperparams": _hyperparams(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    random.seed(config.SEED)

    project = db.get_project(config.PROJECT)
    if project is None:
        raise SystemExit(f"Project '{config.PROJECT}' not found in database")

    rank = project["identification_rank"]
    if rank != "genus":
        print(f"warning: identification_rank is '{rank}', script trains genus labels")

    processed_root = config.PROCESSED_DIR / config.PROJECT
    if not processed_root.is_dir():
        raise SystemExit(
            f"No processed images at {processed_root}. "
            "Run: uv run python scripts/crop_identification_images.py"
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.MODELS_DIR / config.PROJECT / "identification" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir)
    log(logger, f"run_dir={run_dir}")
    log(logger, f"project={config.PROJECT} identification_rank={rank}")
    log(logger, f"processed_dir={processed_root}")

    ratings_path = quality_ratings_path(config.PROJECT)
    if not ratings_path.is_file():
        raise SystemExit(
            f"No quality ratings at {ratings_path}. "
            "Run: uv run python scripts/crop_identification_images.py"
        )
    ratings = load_quality_ratings(config.PROJECT)
    log(
        logger,
        f"quality_ratings={ratings_path} images={len(ratings)} "
        f"min_quality={config.MIN_QUALITY} high_quality={config.HIGH_QUALITY}",
    )

    frozen_path = frozen_splits_path(config.PROJECT)
    if frozen_path.is_file():
        splits = load_splits(frozen_path)
        log(logger, f"using frozen splits {frozen_path}")
        splits = filter_splits_by_quality(splits, ratings, logger)
    else:
        labeled = collect_labeled_paths(config.PROJECT)
        log(logger, f"harmonized processed images: {len(labeled)}")
        labeled = filter_labeled_by_quality(labeled, ratings, logger)
        labeled = filter_by_min_count(labeled, config.MIN_IMAGES_PER_CLASS)
        genus_counts = Counter(g for _, g in labeled)
        log(
            logger,
            f"after min_count>={config.MIN_IMAGES_PER_CLASS}: "
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
            log(logger, f"dropped genera with <{min_for_split} images: {sorted(too_small)}")

        splits = stratified_split(records, seed=config.SEED)
        write_splits(frozen_path, splits)
        log(logger, f"wrote frozen splits {frozen_path}")

    write_splits(run_dir / "splits.json", splits)
    log(
        logger,
        f"split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
    )

    metrics = train_model(run_dir, splits, logger, ratings)
    log(logger, f"done. metrics={json.dumps(metrics)}")
