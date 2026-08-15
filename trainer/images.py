"""Image filesystem helpers under trainer/images/."""

from pathlib import Path

from trainer import db

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def count_images(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def project_stats(taxon: str, identification_rank: str = "species") -> dict:
    project_dir = IMAGES_DIR / taxon
    collections = []

    if project_dir.is_dir():
        for collection_dir in sorted(project_dir.iterdir()):
            if not collection_dir.is_dir():
                continue
            collections.append({
                "name": collection_dir.name,
                "count": count_images(collection_dir),
            })

    if identification_rank == "genus":
        taxa_rows = _harmonized_taxa_by_genus(taxon)
    else:
        taxa_rows = _harmonized_taxa_by_species(taxon)

    for row in taxa_rows:
        row["images_per_observation"] = _images_per_observation(
            row["count"], row.get("observation_count_finland", "")
        )

    anno = _project_annotation_distribution(taxon)
    return {"collections": collections, "taxa": taxa_rows, "annotation": anno}


def _images_per_observation(count: int, observation_count: str) -> float | None:
    if not observation_count:
        return None
    obs = int(observation_count)
    if obs <= 0:
        return None
    return count / obs


def _load_project_taxa(project: str) -> list[dict]:
    from trainer.harmonize import load_taxa_json, taxa_json_path

    path = taxa_json_path(project)
    if not path.is_file():
        return []
    return load_taxa_json(path)


def _harmonization_rows(project: str) -> list[dict]:
    from trainer.harmonize import harmonization_path, read_harmonization

    rows = read_harmonization(harmonization_path(project))
    if not rows:
        return []
    return [r for r in rows if r.get("authoritative_name")]


def _image_count(row: dict) -> int:
    raw = row.get("image_count") or "0"
    return int(raw)


def _harmonized_taxa_by_species(project: str) -> list[dict]:
    by_name: dict[str, dict] = {}
    for row in _harmonization_rows(project):
        name = row["authoritative_name"]
        existing = by_name.get(name)
        if existing is None:
            by_name[name] = {
                "taxon": name,
                "count": _image_count(row),
                "observation_count_finland": row.get("observation_count_finland", ""),
            }
        else:
            existing["count"] += _image_count(row)
    return [by_name[name] for name in sorted(by_name)]


def _genus_observation_count(genus_name: str, finbif_taxa: list[dict]) -> str:
    from trainer.harmonize import normalize_name

    target = normalize_name(genus_name)
    for taxon in finbif_taxa:
        if taxon.get("taxon_rank") != "MX.genus":
            continue
        if normalize_name(taxon["scientific_name"]) == target:
            return taxon.get("observation_count_finland", "")
    return ""


def _harmonized_taxa_by_genus(project: str) -> list[dict]:
    counts: dict[str, int] = {}
    for row in _harmonization_rows(project):
        genus = row["authoritative_name"].split()[0]
        counts[genus] = counts.get(genus, 0) + _image_count(row)
    finbif_taxa = _load_project_taxa(project)
    return [
        {
            "taxon": name,
            "count": counts[name],
            "observation_count_finland": _genus_observation_count(name, finbif_taxa),
        }
        for name in sorted(counts)
    ]


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


def image_path_under_taxon_project(image_path: str, taxon: str) -> bool:
    """True if path resolves to a file under trainer/images/<taxon>/."""
    if not image_path_under_images_root(image_path):
        return False
    full = (IMAGES_DIR / image_path).resolve()
    project_root = (IMAGES_DIR / taxon).resolve()
    try:
        full.relative_to(project_root)
    except ValueError:
        return False
    return full.is_file()
