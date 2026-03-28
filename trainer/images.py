"""Image filesystem helpers under trainer/images/."""

from pathlib import Path

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

    return {"collections": collections, "taxa": taxa_rows}


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
