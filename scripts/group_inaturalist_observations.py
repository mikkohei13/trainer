"""
Group iNaturalist photos that probably belong to the same observation.

Filenames under trainer/images_processed/<project>/inaturalist are iNaturalist
photo IDs. Those IDs are assigned sequentially at upload time, so photos from
one observation usually have nearby IDs. This script clusters numeric filenames
within each taxon folder when consecutive IDs differ by at most MAX_GAP.

Writes JSON that another script can load:

    data = json.loads(path.read_text(encoding="utf-8"))
    for photos in data["observations"]:
        # photos: paths relative to the inaturalist root, same observation

    mapping = data["photo_to_observation"]
    if mapping[path_a] == mapping[path_b]:
        # same observation

    uv run python scripts/group_inaturalist_observations.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from trainer.images import IMAGE_EXTS

PROJECT = "auchenorrhyncha"
COLLECTION = "inaturalist"
# Consecutive photo IDs in the same taxon folder within this distance are
# treated as one observation. iNaturalist IDs are global, so a gap of 20 is
# a short burst of uploads, typical of multi-photo observations.
MAX_GAP = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = (
    PROJECT_ROOT / "trainer" / "images_processed" / PROJECT / COLLECTION
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "trainer"
    / "images_processed"
    / PROJECT
    / "inaturalist_observations.json"
)


def photo_id_from_name(name: str) -> int | None:
    stem = Path(name).stem
    if not stem.isdigit():
        return None
    return int(stem)


def list_photos(root: Path) -> dict[str, list[tuple[int, str]]]:
    """Map each taxon folder to (photo_id, relative_path) pairs."""
    by_folder: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if not root.is_dir():
        return by_folder
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        photo_id = photo_id_from_name(path.name)
        if photo_id is None:
            continue
        rel = path.relative_to(root).as_posix()
        folder = path.parent.relative_to(root).as_posix()
        by_folder[folder].append((photo_id, rel))
    return by_folder


def cluster_sorted_items(
    items: list[tuple[int, str]], max_gap: int
) -> list[list[str]]:
    """Group items sorted by photo ID when consecutive IDs differ by <= max_gap."""
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item[0])
    groups: list[list[str]] = [[ordered[0][1]]]
    prev_id = ordered[0][0]
    for photo_id, rel in ordered[1:]:
        if photo_id - prev_id <= max_gap:
            groups[-1].append(rel)
        else:
            groups.append([rel])
        prev_id = photo_id
    return groups


def group_photos(root: Path, max_gap: int) -> list[list[str]]:
    """Return observation groups as lists of paths relative to root."""
    groups: list[list[str]] = []
    by_folder = list_photos(root)
    for folder in sorted(by_folder):
        groups.extend(cluster_sorted_items(by_folder[folder], max_gap))
    return groups


def build_payload(groups: list[list[str]], max_gap: int, source: str) -> dict:
    photo_to_observation = {
        photo: index
        for index, photos in enumerate(groups)
        for photo in photos
    }
    return {
        "max_gap": max_gap,
        "source": source,
        "photo_count": len(photo_to_observation),
        "observation_count": len(groups),
        "observations": groups,
        "photo_to_observation": photo_to_observation,
    }


def load_observations(path: Path) -> dict:
    """Read JSON written by this script."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    groups = group_photos(IMAGES_ROOT, MAX_GAP)
    source = IMAGES_ROOT.relative_to(PROJECT_ROOT).as_posix()
    payload = build_payload(groups, MAX_GAP, source)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    multi = sum(1 for g in groups if len(g) > 1)
    in_multi = sum(len(g) for g in groups if len(g) > 1)
    print(
        f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}: "
        f"{payload['photo_count']} photos, "
        f"{payload['observation_count']} observations "
        f"({multi} with multiple photos, {in_multi} photos in those), "
        f"max_gap={MAX_GAP}."
    )


if __name__ == "__main__":
    main()
