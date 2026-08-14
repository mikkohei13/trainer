"""Harmonize image folder names to FinBIF scientific names."""

import csv
import json
from pathlib import Path

from trainer import images

DATA_DIR = Path(__file__).resolve().parent / "data"

TAXA_JSON_NAME = "taxa.json"
HARMONIZATION_NAME = "harmonization.tsv"
FIELDNAMES = ("image_name", "authoritative_name", "status")
ALLOWED_RANKS = {"MX.genus", "MX.species"}


class TaxaJsonMissing(Exception):
    """Raised when taxa.json is not found for a project."""


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def taxa_json_path(project: str) -> Path:
    return DATA_DIR / project / TAXA_JSON_NAME


def harmonization_path(project: str) -> Path:
    return DATA_DIR / project / HARMONIZATION_NAME


def collect_image_names(project: str) -> list[str]:
    """Unique species-folder names under trainer/images/<project>/, sorted."""
    project_dir = images.IMAGES_DIR / project
    names: set[str] = set()
    if not project_dir.is_dir():
        return []
    for collection_dir in project_dir.iterdir():
        if not collection_dir.is_dir():
            continue
        for taxon_dir in collection_dir.iterdir():
            if taxon_dir.is_dir():
                names.add(taxon_dir.name)
    return sorted(names)


def load_taxa_json(path: Path) -> list[dict]:
    """
    Load genus and species from a FinBIF taxa.json file.
    Each item is {scientific_name, synonyms} where synonyms is a list of names.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"] if isinstance(data, dict) else data
    rows = []
    for item in results:
        if item.get("taxonRank") not in ALLOWED_RANKS:
            continue
        scientific = (item.get("scientificName") or "").strip()
        if not scientific:
            continue
        synonyms = []
        for syn in item.get("synonyms") or []:
            name = (syn.get("scientificName") or "").strip()
            if name:
                synonyms.append(name)
        rows.append({
            "scientific_name": scientific,
            "synonyms": synonyms,
        })
    return rows


def match_image_name(image_name: str, taxa: list[dict]) -> dict:
    """
    Return {image_name, authoritative_name, status} for one image folder name.
    status is "" on success, "unknown", or "multiple".
    """
    norm = normalize_name(image_name)
    if not norm:
        return {
            "image_name": image_name,
            "authoritative_name": "",
            "status": "unknown",
        }

    exact = [
        t["scientific_name"]
        for t in taxa
        if normalize_name(t["scientific_name"]) == norm
    ]
    exact_unique = list(dict.fromkeys(exact))
    if len(exact_unique) == 1:
        return {
            "image_name": image_name,
            "authoritative_name": exact_unique[0],
            "status": "",
        }
    if len(exact_unique) > 1:
        return {
            "image_name": image_name,
            "authoritative_name": "",
            "status": "multiple",
        }

    synonym_hits = [
        t["scientific_name"]
        for t in taxa
        if any(normalize_name(s) == norm for s in t["synonyms"])
    ]
    synonym_unique = list(dict.fromkeys(synonym_hits))
    if len(synonym_unique) == 1:
        return {
            "image_name": image_name,
            "authoritative_name": synonym_unique[0],
            "status": "",
        }
    if len(synonym_unique) > 1:
        return {
            "image_name": image_name,
            "authoritative_name": "",
            "status": "multiple",
        }

    return {
        "image_name": image_name,
        "authoritative_name": "",
        "status": "unknown",
    }


def build_harmonization_rows(image_names: list[str], taxa: list[dict]) -> list[dict]:
    return [match_image_name(name, taxa) for name in sorted(image_names)]


def write_harmonization(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image_name": row["image_name"],
                "authoritative_name": row["authoritative_name"],
                "status": row["status"],
            })


def read_harmonization(path: Path) -> list[dict] | None:
    """Return rows from harmonization.tsv, or None if the file does not exist."""
    if not path.is_file():
        return None
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({
                "image_name": row.get("image_name") or "",
                "authoritative_name": row.get("authoritative_name") or "",
                "status": row.get("status") or "",
            })
    return rows


def generate_harmonization(project: str) -> Path:
    """
    Build and write harmonization.tsv for a project.
    Raises TaxaJsonMissing if taxa.json is absent.
    """
    json_path = taxa_json_path(project)
    if not json_path.is_file():
        raise TaxaJsonMissing(f"taxa.json not found for project '{project}'")
    taxa = load_taxa_json(json_path)
    image_names = collect_image_names(project)
    rows = build_harmonization_rows(image_names, taxa)
    out_path = harmonization_path(project)
    write_harmonization(out_path, rows)
    return out_path
