"""Harmonize image folder names to FinBIF scientific names."""

import csv
import json
from pathlib import Path

from trainer import images

DATA_DIR = Path(__file__).resolve().parent / "data"

TAXA_JSON_NAME = "taxa.json"
HARMONIZATION_NAME = "harmonization.tsv"
FIELDNAMES = (
    "image_name",
    "authoritative_name",
    "status",
    "observation_count_finland",
    "image_count",
)
ALLOWED_RANKS = {"MX.genus", "MX.species"}


class TaxaJsonMissing(Exception):
    """Raised when taxa.json is not found for a project."""


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")



def taxa_json_path(project: str) -> Path:
    return DATA_DIR / project / TAXA_JSON_NAME


def harmonization_path(project: str) -> Path:
    return DATA_DIR / project / HARMONIZATION_NAME




def collect_image_name_counts(project: str) -> dict[str, int]:
    """Image counts per species-folder name under trainer/images/<project>/."""
    project_dir = images.IMAGES_DIR / project
    counts: dict[str, int] = {}
    if not project_dir.is_dir():
        return counts
    for collection_dir in project_dir.iterdir():
        if not collection_dir.is_dir():
            continue
        for taxon_dir in collection_dir.iterdir():
            if not taxon_dir.is_dir():
                continue
            count = sum(
                1
                for f in taxon_dir.iterdir()
                if f.is_file() and f.suffix.lower() in images.IMAGE_EXTS
            )
            name = taxon_dir.name
            counts[name] = counts.get(name, 0) + count
    return counts


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
        obs = item.get("observationCountFinland")
        if obs is None or obs == "":
            observation_count = ""
        else:
            observation_count = str(obs)
        rows.append({
            "scientific_name": scientific,
            "synonyms": synonyms,
            "observation_count_finland": observation_count,
            "taxon_rank": item.get("taxonRank") or "",
        })
    return rows


def _unmatched_row(image_name: str, status: str) -> dict:
    return {
        "image_name": image_name,
        "authoritative_name": "",
        "status": status,
        "observation_count_finland": "",
    }


def _matched_row(image_name: str, taxon: dict) -> dict:
    return {
        "image_name": image_name,
        "authoritative_name": taxon["scientific_name"],
        "status": "",
        "observation_count_finland": taxon.get("observation_count_finland", ""),
    }


def _unique_taxa(hits: list[dict]) -> list[dict]:
    unique = []
    seen: set[str] = set()
    for taxon in hits:
        name = taxon["scientific_name"]
        if name in seen:
            continue
        seen.add(name)
        unique.append(taxon)
    return unique


def match_image_name(image_name: str, taxa: list[dict]) -> dict:
    """
    Return a harmonization row for one image folder name.
    status is "" on success, "unknown", or "multiple".
    observation_count_finland is set only when there is a single authoritative name.
    """
    norm = normalize_name(image_name)
    if not norm:
        return _unmatched_row(image_name, "unknown")

    exact = _unique_taxa([
        t for t in taxa if normalize_name(t["scientific_name"]) == norm
    ])
    if len(exact) == 1:
        return _matched_row(image_name, exact[0])
    if len(exact) > 1:
        return _unmatched_row(image_name, "multiple")

    synonym_hits = _unique_taxa([
        t for t in taxa
        if any(normalize_name(s) == norm for s in t["synonyms"])
    ])
    if len(synonym_hits) == 1:
        return _matched_row(image_name, synonym_hits[0])
    if len(synonym_hits) > 1:
        return _unmatched_row(image_name, "multiple")

    return _unmatched_row(image_name, "unknown")


def build_harmonization_rows(
    image_counts: dict[str, int],
    taxa: list[dict],
) -> list[dict]:
    rows = []
    for name in sorted(image_counts):
        row = match_image_name(name, taxa)
        row["image_count"] = str(image_counts[name])
        rows.append(row)
    return rows


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
                "observation_count_finland": row.get("observation_count_finland", ""),
                "image_count": row.get("image_count", ""),
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
                "observation_count_finland": row.get("observation_count_finland") or "",
                "image_count": row.get("image_count") or "",
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
    image_counts = collect_image_name_counts(project)
    rows = build_harmonization_rows(image_counts, taxa)
    out_path = harmonization_path(project)
    write_harmonization(out_path, rows)
    return out_path
