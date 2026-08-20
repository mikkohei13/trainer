"""Labels, quality filters, and train/val/test splits."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from trainer.harmonize import harmonization_path, read_harmonization
from trainer.images import IMAGE_EXTS

from . import config


def setup_logging(run_dir: Path) -> logging.Logger:
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


def log(logger: logging.Logger, msg: str) -> None:
    logger.info(msg)


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
    project_dir = config.PROCESSED_DIR / project
    if not project_dir.is_dir():
        return []
    paths = []
    for f in project_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            rel = f.relative_to(config.PROCESSED_DIR)
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


def quality_ratings_path(project: str) -> Path:
    return config.PROCESSED_DIR / project / "quality.json"


def load_quality_ratings(project: str) -> dict[str, float]:
    path = quality_ratings_path(project)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def filter_labeled_by_quality(
    labeled: list[tuple[str, str]],
    ratings: dict[str, float],
    logger: logging.Logger | None = None,
    min_quality: float | None = None,
) -> list[tuple[str, str]]:
    """Keep labeled crops whose saved quality is strictly above min_quality."""
    if min_quality is None:
        min_quality = config.MIN_QUALITY
    kept: list[tuple[str, str]] = []
    dropped = 0
    missing = 0
    for rel, genus in labeled:
        score = ratings.get(rel)
        if score is None:
            missing += 1
            continue
        if score <= min_quality:
            dropped += 1
            continue
        kept.append((rel, genus))
    if logger:
        log(
            logger,
            f"quality filter (drop score<={min_quality}): "
            f"kept={len(kept)} dropped={dropped} missing={missing}",
        )
    return kept


def filter_splits_by_quality(
    splits: dict[str, list[dict]],
    ratings: dict[str, float],
    logger: logging.Logger | None = None,
    min_quality: float | None = None,
) -> dict[str, list[dict]]:
    labeled = [(r["crop_path"], r["genus"]) for recs in splits.values() for r in recs]
    kept = set(filter_labeled_by_quality(labeled, ratings, logger, min_quality))
    return {
        key: [r for r in recs if (r["crop_path"], r["genus"]) in kept]
        for key, recs in splits.items()
    }


def labeled_to_records(labeled: list[tuple[str, str]]) -> list[dict]:
    return [{"crop_path": path, "genus": genus} for path, genus in labeled]


def inverse_sqrt_sample_weights(records: list[dict]) -> list[float]:
    """Per-example weights 1/sqrt(n_class). Softens 10–1000× imbalance vs 1/n."""
    counts = Counter(r["genus"] for r in records)
    class_w = {g: 1.0 / (n ** 0.5) for g, n in counts.items()}
    return [class_w[r["genus"]] for r in records]


def stratified_split(
    records: list[dict],
    seed: int | None = None,
) -> dict[str, list[dict]]:
    if seed is None:
        seed = config.SEED
    assert abs(config.TRAIN_FRAC + config.VAL_FRAC + config.TEST_FRAC - 1.0) < 1e-9
    labels = [r["genus"] for r in records]
    train_recs, rest_recs, _, rest_labels = train_test_split(
        records,
        labels,
        test_size=(config.VAL_FRAC + config.TEST_FRAC),
        random_state=seed,
        stratify=labels,
    )
    relative_test = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_recs, test_recs = train_test_split(
        rest_recs,
        test_size=relative_test,
        random_state=seed,
        stratify=rest_labels,
    )
    return {"train": train_recs, "val": val_recs, "test": test_recs}


def frozen_splits_path(project: str) -> Path:
    return config.MODELS_DIR / project / "identification" / "splits.json"


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
