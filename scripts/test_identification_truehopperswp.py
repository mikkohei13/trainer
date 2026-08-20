"""
Run the identification model on already-cropped images in
trainer/images_processed/<project>/truehopperswp.

Ground-truth genus is the first `_`-separated part of the species folder
name (e.g. Aphrophora_alni → Aphrophora). Images whose genus is not in
the checkpoint are skipped. Prints top-1, top-3, and macro-F1, and
writes a confusion matrix next to the run.

    uv run python scripts/test_identification_truehopperswp.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from trainer.images import IMAGE_EXTS

PROJECT = "auchenorrhyncha"
RUN_ID = "20260820-000744"
TOP_K = 3
TOP_CONFUSIONS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "trainer" / "models" / PROJECT / "identification" / RUN_ID
TEST_DIR = PROJECT_ROOT / "trainer" / "images_processed" / PROJECT / "truehopperswp"
CHECKPOINT = RUN_DIR / "best.pt"

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


def genus_from_folder(folder_name: str) -> str:
    return folder_name.split("_")[0]


def list_test_images(test_dir: Path) -> list[Path]:
    return sorted(
        p for p in test_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def partition_known_genera(
    images: list[Path],
    class_to_idx: dict[str, int],
) -> tuple[list[Path], dict[str, int]]:
    known: list[Path] = []
    skipped: dict[str, int] = {}
    for path in images:
        genus = genus_from_folder(path.parent.name)
        if genus in class_to_idx:
            known.append(path)
        else:
            skipped[genus] = skipped.get(genus, 0) + 1
    return known, skipped


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


def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    idx_to_class: dict[int, str],
) -> tuple[np.ndarray, list[str]]:
    used = sorted(set(y_true) | set(y_pred), key=lambda i: idx_to_class[i])
    names = [idx_to_class[i] for i in used]
    index = {c: i for i, c in enumerate(used)}
    n_labels = len(used)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm, names


def confused_pairs(cm: np.ndarray, names: list[str], k: int) -> list[dict]:
    n_labels = len(names)
    pairs = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            a_to_b = int(cm[i, j])
            b_to_a = int(cm[j, i])
            total = a_to_b + b_to_a
            if total == 0:
                continue
            support = int(cm[i].sum() + cm[j].sum())
            pairs.append({
                "a": names[i],
                "b": names[j],
                "a_to_b": a_to_b,
                "b_to_a": b_to_a,
                "n": total,
                "rate": total / support if support else 0.0,
            })
    pairs.sort(key=lambda r: (-r["n"], -r["rate"], r["a"], r["b"]))
    return pairs[:k]


def write_confusion_matrix(
    cm: np.ndarray,
    names: list[str],
    out_png: Path,
    out_csv: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_labels = len(names)
    header = ",".join(["true\\pred"] + names)
    rows = [header]
    for i, name in enumerate(names):
        rows.append(",".join([name] + [str(int(v)) for v in cm[i]]))
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    side = max(12.0, n_labels * 0.22)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_labels), names, rotation=90, fontsize=5)
    ax.set_yticks(range(n_labels), names, fontsize=5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("truehopperswp confusion")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    import torch
    import torch.nn.functional as F

    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")
    if not TEST_DIR.is_dir():
        raise SystemExit(f"Test directory not found: {TEST_DIR}")

    images = list_test_images(TEST_DIR)
    if not images:
        raise SystemExit(f"No images in {TEST_DIR}")

    device = pick_device()
    model, class_to_idx, idx_to_class, img_size = load_model(CHECKPOINT, device)
    images, skipped = partition_known_genera(images, class_to_idx)
    if not images:
        raise SystemExit("No images whose genus is in the checkpoint")

    print(f"run={RUN_DIR}")
    print(f"test_dir={TEST_DIR}")
    print(f"device={device} classes={len(idx_to_class)} img_size={img_size}")
    print(f"n_images={len(images)}")
    print()

    y_true: list[int] = []
    y_pred: list[int] = []
    n_top3 = 0
    n_unreadable = 0

    for i, path in enumerate(images, start=1):
        genus = genus_from_folder(path.parent.name)
        true_idx = class_to_idx[genus]

        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            n_unreadable += 1
            print(f"{path.relative_to(TEST_DIR)}  error: {exc}")
            continue

        x = pil_to_tensor(rgb, img_size).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
        k = min(TOP_K, probs.numel())
        _, indices = torch.topk(probs, k)
        pred_idx = int(indices[0].item())
        topk = {int(v) for v in indices.tolist()}

        y_true.append(int(true_idx))
        y_pred.append(pred_idx)
        if true_idx in topk:
            n_top3 += 1

        if i % 100 == 0 or i == len(images):
            print(f"  {i}/{len(images)}", flush=True)

    n = len(y_true)
    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    top3 = n_top3 / n if n else 0.0
    f1 = macro_f1(y_true, y_pred, len(class_to_idx))

    cm, names = confusion_matrix(y_true, y_pred, idx_to_class)
    confusion_png = RUN_DIR / "truehopperswp_confusion.png"
    confusion_csv = RUN_DIR / "truehopperswp_confusion.csv"
    write_confusion_matrix(cm, names, confusion_png, confusion_csv)
    pairs = confused_pairs(cm, names, TOP_CONFUSIONS)

    skipped_parts = [
        f"{genus}={count}" for genus, count in sorted(skipped.items())
    ]

    print()
    print(f"n_eval={n}")
    print(f"skipped_unknown_genera={', '.join(skipped_parts) if skipped_parts else '(none)'}")
    print(f"n_unreadable={n_unreadable}")
    print(f"top1={top1:.4f}")
    print(f"top3={top3:.4f}")
    print(f"macro_f1={f1:.4f}")
    print(f"confusion_png={confusion_png}")
    print(f"confusion_csv={confusion_csv}")
    print()
    print(f"most confused pairs (top {len(pairs)}):")
    if not pairs:
        print("  (none)")
    for row in pairs:
        print(
            f"  {row['a']} ↔ {row['b']}  {row['n']}  "
            f"({row['a']}→{row['b']} {row['a_to_b']}, "
            f"{row['b']}→{row['a']} {row['b_to_a']})"
        )


if __name__ == "__main__":
    main()
