"""Image filesystem helpers under trainer/images/."""

from pathlib import Path

from trainer import db

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def count_images(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def project_stats(taxon: str) -> dict:
    project_dir = IMAGES_DIR / taxon
    collections = []
    taxa_rows = []

    if project_dir.is_dir():
        for collection_dir in sorted(project_dir.iterdir()):
            if not collection_dir.is_dir():
                continue
            collection_count = count_images(collection_dir)
            collections.append({
                "name": collection_dir.name,
                "count": collection_count,
            })
            for taxon_dir in sorted(collection_dir.iterdir()):
                if not taxon_dir.is_dir():
                    continue
                count = sum(
                    1 for f in taxon_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                )
                name = taxon_dir.name
                existing = next((t for t in taxa_rows if t["taxon"] == name), None)
                if existing:
                    existing["count"] += count
                else:
                    taxa_rows.append({"taxon": name, "count": count})

    anno = _project_annotation_distribution(taxon)
    return {"collections": collections, "taxa": taxa_rows, "annotation": anno}


def project_annotation_buckets(taxon: str) -> dict[str, list[str]]:
    """
    Partition project images by annotation class.
    Keys: 'not_annotated', '0', '1', '2', … (organism count as string).
    """
    paths = list_project_image_paths(taxon)
    if not paths:
        return {"not_annotated": []}

    no_set, box_map = db.project_annotation_state(taxon)
    buckets: dict[str, list[str]] = {"not_annotated": []}

    for p in paths:
        if p in no_set:
            key = "0"
        elif p in box_map:
            key = str(box_map[p])
        else:
            key = "not_annotated"
        buckets.setdefault(key, []).append(p)

    for k in buckets:
        buckets[k].sort()
    return buckets


def normalize_annotation_bucket(bucket: str | None) -> str | None:
    """Return canonical bucket key, or None if invalid."""
    if bucket is None:
        return None
    b = bucket.strip()
    if not b:
        return None
    if b == "not_annotated":
        return "not_annotated"
    if b.isdigit():
        return str(int(b))
    return None


def _project_annotation_distribution(taxon: str) -> dict:
    """
    Count images by annotation status: not annotated, 0 organisms, 1, 2, …
    """
    buckets = project_annotation_buckets(taxon)
    all_paths = list_project_image_paths(taxon)
    if not all_paths:
        return {
            "not_annotated": 0,
            "by_organism_count": [],
            "total_images": 0,
        }

    not_annotated = len(buckets.get("not_annotated", []))
    by_count = []
    for k in sorted(int(x) for x in buckets if x != "not_annotated"):
        sk = str(k)
        c = len(buckets.get(sk, []))
        if c > 0:
            by_count.append({"organisms": k, "count": c})

    return {
        "not_annotated": not_annotated,
        "by_organism_count": by_count,
        "total_images": len(all_paths),
    }


def list_project_image_paths(taxon: str) -> list[str]:
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


def image_path_under_images_root(image_path: str) -> bool:
    try:
        full = (IMAGES_DIR / image_path).resolve()
        root = IMAGES_DIR.resolve()
        full.relative_to(root)
        return True
    except ValueError:
        return False
