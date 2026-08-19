"""
Correlate crop resolution (megapixels) with quality ratings.

Reads trainer/images_processed/<project>/quality.json and measures each
crop under trainer/images_processed/. Writes a hexbin chart and a summary
JSON next to quality.json. Copies mismatch examples into two folders,
and 100 random crops into a samples folder named by quality score.

    uv run python scripts/analyze_quality_vs_resolution.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Hardcoded parameters
# ---------------------------------------------------------------------------

PROJECT = "auchenorrhyncha"
QUALITY_JSON_NAME = "quality.json"
CHART_NAME = "quality_vs_resolution.png"
SUMMARY_NAME = "quality_vs_resolution.json"
LOW_RES_HIGH_Q_DIR = "low_resolution_high_quality"
HIGH_RES_LOW_Q_DIR = "high_resolution_low_quality"
RANDOM_SAMPLE_DIR = "samples"
HEXBIN_GRIDSIZE = (80, 40)
HEXBIN_X_MAX = 1.5
N_BINS = 20
SAMPLE_N = 50
RANDOM_SAMPLE_N = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "trainer" / "images_processed"


def quality_json_path(project: str) -> Path:
    return PROCESSED_DIR / project / QUALITY_JSON_NAME


def load_quality_ratings(project: str) -> dict[str, float]:
    path = quality_json_path(project)
    if not path.is_file():
        raise SystemExit(f"Missing quality ratings: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def crop_megapixels(path: Path) -> float:
    with Image.open(path) as img:
        width, height = img.size
    return (width * height) / 1_000_000


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx.astype(float), ry.astype(float))[0, 1])


def collect_pairs(
    project: str, ratings: dict[str, float]
) -> tuple[list[str], np.ndarray, np.ndarray, int, int]:
    paths: list[str] = []
    megapixels: list[float] = []
    qualities: list[float] = []
    skipped_missing = 0
    skipped_unreadable = 0
    total = len(ratings)

    for i, (rel_path, quality) in enumerate(ratings.items(), start=1):
        if i % 1000 == 0 or i == 1:
            print(f"reading crops {i}/{total} …")
        abs_path = PROCESSED_DIR / rel_path
        if not abs_path.is_file():
            skipped_missing += 1
            continue
        try:
            mp = crop_megapixels(abs_path)
        except (OSError, UnidentifiedImageError) as exc:
            print(f"skip unreadable {rel_path}: {exc}")
            skipped_unreadable += 1
            continue
        paths.append(rel_path)
        megapixels.append(mp)
        qualities.append(quality)

    return (
        paths,
        np.array(megapixels, dtype=np.float64),
        np.array(qualities, dtype=np.float64),
        skipped_missing,
        skipped_unreadable,
    )


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def binned_mean_quality(megapixels: np.ndarray, qualities: np.ndarray, n_bins: int) -> list[dict]:
    edges = np.histogram_bin_edges(megapixels, bins=n_bins)
    rows: list[dict] = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (megapixels >= lo) & (megapixels <= hi)
        else:
            mask = (megapixels >= lo) & (megapixels < hi)
        n = int(np.count_nonzero(mask))
        if n == 0:
            continue
        rows.append({
            "megapixels_min": lo,
            "megapixels_max": hi,
            "n": n,
            "mean_quality": float(np.mean(qualities[mask])),
        })
    return rows


def _sample_rows(
    indices: np.ndarray,
    paths: list[str],
    megapixels: np.ndarray,
    qualities: np.ndarray,
) -> list[dict]:
    return [
        {
            "path": paths[i],
            "megapixels": float(megapixels[i]),
            "quality": float(qualities[i]),
        }
        for i in indices
    ]


def sample_mismatches(
    paths: list[str],
    megapixels: np.ndarray,
    qualities: np.ndarray,
    n: int,
) -> dict:
    mp_low = float(np.quantile(megapixels, 0.25))
    mp_high = float(np.quantile(megapixels, 0.75))
    q_low = float(np.quantile(qualities, 0.25))
    q_high = float(np.quantile(qualities, 0.75))

    low_res_high_q = np.flatnonzero((megapixels <= mp_low) & (qualities >= q_high))
    high_res_low_q = np.flatnonzero((megapixels >= mp_high) & (qualities <= q_low))
    low_res_high_q = low_res_high_q[np.argsort(megapixels[low_res_high_q])][:n]
    high_res_low_q = high_res_low_q[np.argsort(-megapixels[high_res_low_q])][:n]

    return {
        "thresholds": {
            "megapixels_low": mp_low,
            "megapixels_high": mp_high,
            "quality_low": q_low,
            "quality_high": q_high,
        },
        "low_resolution_high_quality": _sample_rows(
            low_res_high_q, paths, megapixels, qualities
        ),
        "high_resolution_low_quality": _sample_rows(
            high_res_low_q, paths, megapixels, qualities
        ),
    }


def copy_samples(rows: list[dict], dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    used: set[str] = set()
    for row in rows:
        src = PROCESSED_DIR / row["path"]
        name = Path(row["path"]).name
        if name in used:
            name = f"{Path(row['path']).parent.name}_{name}"
        used.add(name)
        shutil.copy2(src, dest_dir / name)


def copy_random_samples(
    paths: list[str],
    qualities: np.ndarray,
    dest_dir: Path,
    n: int,
) -> int:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    count = min(n, len(paths))
    rng = np.random.default_rng()
    indices = rng.choice(len(paths), size=count, replace=False)
    used: set[str] = set()
    for i in indices:
        src = PROCESSED_DIR / paths[int(i)]
        filename = Path(paths[int(i)]).name
        name = f"{qualities[int(i)]:.3f}-{filename}"
        if name in used:
            name = f"{qualities[int(i)]:.3f}-{Path(paths[int(i)]).parent.name}_{filename}"
        used.add(name)
        shutil.copy2(src, dest_dir / name)
    return count


def print_samples(title: str, rows: list[dict]) -> None:
    print(f"{title} ({len(rows)}):")
    for row in rows:
        print(f"  {row['path']}  mp={row['megapixels']:.4f}  quality={row['quality']:.4f}")


def write_chart(megapixels: np.ndarray, qualities: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(
        megapixels,
        qualities,
        gridsize=HEXBIN_GRIDSIZE,
        extent=(0, HEXBIN_X_MAX, 0, 1),
        mincnt=1,
        cmap="viridis",
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("count")
    ax.set_xlabel("Crop resolution (megapixels)")
    ax.set_ylabel("Quality rating")
    ax.set_xlim(0, HEXBIN_X_MAX)
    ax.set_ylim(0, 1)
    ax.set_title("Quality vs crop resolution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze(project: str) -> None:
    ratings = load_quality_ratings(project)
    paths, megapixels, qualities, skipped_missing, skipped_unreadable = collect_pairs(
        project, ratings
    )
    n = int(megapixels.size)
    if n < 2:
        raise SystemExit("Need at least two rated crops to compute correlation")

    out_dir = quality_json_path(project).parent
    chart_path = out_dir / CHART_NAME
    summary_path = out_dir / SUMMARY_NAME

    write_chart(megapixels, qualities, chart_path)
    samples = sample_mismatches(paths, megapixels, qualities, SAMPLE_N)
    low_res_dir = out_dir / LOW_RES_HIGH_Q_DIR
    high_res_dir = out_dir / HIGH_RES_LOW_Q_DIR
    copy_samples(samples["low_resolution_high_quality"], low_res_dir)
    copy_samples(samples["high_resolution_low_quality"], high_res_dir)
    random_dir = out_dir / RANDOM_SAMPLE_DIR
    random_n = copy_random_samples(paths, qualities, random_dir, RANDOM_SAMPLE_N)

    summary = {
        "n": n,
        "skipped_missing": skipped_missing,
        "skipped_unreadable": skipped_unreadable,
        "pearson_r": float(np.corrcoef(megapixels, qualities)[0, 1]),
        "spearman_rho": spearman_rho(megapixels, qualities),
        "megapixels": describe(megapixels),
        "quality": describe(qualities),
        "binned_means": binned_mean_quality(megapixels, qualities, N_BINS),
        "samples": samples,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"n={n} pearson_r={summary['pearson_r']:.4f} spearman_rho={summary['spearman_rho']:.4f}")
    print(f"chart={chart_path}")
    print(f"summary={summary_path}")
    print(f"low_resolution_high_quality={low_res_dir}")
    print(f"high_resolution_low_quality={high_res_dir}")
    print(f"samples={random_dir} n={random_n}")
    print_samples("low resolution, high quality", samples["low_resolution_high_quality"])
    print_samples("high resolution, low quality", samples["high_resolution_low_quality"])


def main() -> None:
    analyze(PROJECT)


if __name__ == "__main__":
    main()
