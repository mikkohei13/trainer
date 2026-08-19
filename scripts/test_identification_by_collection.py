"""
Evaluate the identification model on the frozen test split, with metrics
broken out by source collection.

    uv run python scripts/test_identification_by_collection.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT = "auchenorrhyncha"
RUN_ID = "20260819-135326"
SPLITS_NAME = "splits-until-20260819-135326.json"
TOP_K = 3
BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "trainer" / "models" / PROJECT / "identification" / RUN_ID
SPLITS_PATH = (
    PROJECT_ROOT / "trainer" / "models" / PROJECT / "identification" / SPLITS_NAME
)
CHECKPOINT = RUN_DIR / "best.pt"
PROCESSED_DIR = PROJECT_ROOT / "trainer" / "images_processed"
OUT_PATH = RUN_DIR / "test_by_collection.json"

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


def pil_to_tensor(img: Image.Image, img_size: int):
    import torch

    boxed = letterbox(img, img_size, fill=0)
    arr = np.asarray(boxed, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr)


def collection_from_crop_path(crop_path: str) -> str:
    parts = crop_path.split("/")
    if len(parts) < 2:
        raise ValueError(f"unexpected crop_path: {crop_path}")
    return parts[1]


def load_test_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "test" not in data:
        raise ValueError(f"{path} missing 'test'")
    return [
        {"crop_path": r["crop_path"], "genus": r["genus"]}
        for r in data["test"]
    ]


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
    img_size = ckpt.get("img_size_eval", ckpt.get("img_size"))
    return model, class_to_idx, idx_to_class, int(img_size)


def macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if (tp + fn) == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def metrics_for(y_true: list[int], y_pred: list[int], in_top3: list[bool], num_classes: int) -> dict:
    n = len(y_true)
    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    top3 = sum(in_top3) / n if n else 0.0
    return {
        "n": n,
        "num_classes": len(set(y_true)),
        "accuracy": top1,
        "top1": top1,
        "top3": top3,
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
    }


def main() -> None:
    import torch
    import torch.nn.functional as F

    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")
    if not SPLITS_PATH.is_file():
        raise SystemExit(f"Splits not found: {SPLITS_PATH}")

    records = load_test_records(SPLITS_PATH)
    if not records:
        raise SystemExit(f"No test records in {SPLITS_PATH}")

    device = pick_device()
    model, class_to_idx, idx_to_class, img_size = load_model(CHECKPOINT, device)
    num_classes = len(class_to_idx)

    print(f"run={RUN_DIR}")
    print(f"splits={SPLITS_PATH}")
    print(f"device={device} classes={num_classes} img_size={img_size}")
    print(f"n_test={len(records)}")
    print()

    y_true: list[int] = []
    y_pred: list[int] = []
    in_top3: list[bool] = []
    collections: list[str] = []
    n_unreadable = 0
    n_unknown_genus = 0
    n_missing = 0

    pending_tensors = []
    pending_true = []
    pending_coll = []

    def flush_batch() -> None:
        if not pending_tensors:
            return
        x = torch.stack(pending_tensors).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)
        k = min(TOP_K, probs.size(1))
        _, indices = torch.topk(probs, k, dim=1)
        pred = indices[:, 0].tolist()
        topk = indices.tolist()
        for true_idx, pred_idx, top, coll in zip(pending_true, pred, topk, pending_coll):
            y_true.append(true_idx)
            y_pred.append(int(pred_idx))
            in_top3.append(true_idx in top)
            collections.append(coll)
        pending_tensors.clear()
        pending_true.clear()
        pending_coll.clear()

    for i, rec in enumerate(records, start=1):
        genus = rec["genus"]
        if genus not in class_to_idx:
            n_unknown_genus += 1
            continue
        path = PROCESSED_DIR / rec["crop_path"]
        if not path.is_file():
            n_missing += 1
            print(f"missing: {rec['crop_path']}")
            continue
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            n_unreadable += 1
            print(f"{rec['crop_path']}  error: {exc}")
            continue

        pending_tensors.append(pil_to_tensor(rgb, img_size))
        pending_true.append(int(class_to_idx[genus]))
        pending_coll.append(collection_from_crop_path(rec["crop_path"]))
        if len(pending_tensors) >= BATCH_SIZE:
            flush_batch()

        if i % 200 == 0 or i == len(records):
            print(f"  {i}/{len(records)}", flush=True)

    flush_batch()

    overall = metrics_for(y_true, y_pred, in_top3, num_classes)
    by_collection: dict[str, dict] = {}
    for coll in sorted(set(collections)):
        idx = [i for i, c in enumerate(collections) if c == coll]
        by_collection[coll] = metrics_for(
            [y_true[i] for i in idx],
            [y_pred[i] for i in idx],
            [in_top3[i] for i in idx],
            num_classes,
        )

    payload = {
        "run": RUN_ID,
        "splits": SPLITS_NAME,
        "n_test": len(records),
        "n_eval": overall["n"],
        "n_unreadable": n_unreadable,
        "n_missing": n_missing,
        "n_unknown_genus": n_unknown_genus,
        "overall": overall,
        "by_collection": by_collection,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _fmt(row: dict) -> str:
        return (
            f"n={row['n']} classes={row['num_classes']} "
            f"accuracy={row['accuracy']:.4f} top1={row['top1']:.4f} "
            f"top3={row['top3']:.4f} macro_f1={row['macro_f1']:.4f}"
        )

    print()
    print(f"n_eval={overall['n']}")
    print(f"n_unreadable={n_unreadable} n_missing={n_missing} n_unknown_genus={n_unknown_genus}")
    print(f"overall  {_fmt(overall)}")
    for coll, row in by_collection.items():
        print(f"{coll}  {_fmt(row)}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
