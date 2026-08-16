"""
Fetch observation photos from iNaturalist for FinBIF genera and species listed
in taxa.json. One folder per taxon (Genus_species), files named <photo_id>.jpg.

Looks up each FinBIF scientific name and its synonyms on the iNaturalist taxa
API, then queries observations for the matching taxon ids. Images are always
saved under the FinBIF scientificName, not the synonym used for the query.

API reference: https://api.inaturalist.org/v2/docs/
Taxa lookup: https://api.inaturalist.org/v1/docs/#!/Taxa/get_taxa

Read-only search does not require an API key.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAXA_LOOKUP_URL = "https://api.inaturalist.org/v1/taxa"
OBSERVATIONS_URL = "https://api.inaturalist.org/v2/observations"

taxon_name = "auchenorrhyncha"
target_folder = "inaturalist"
taxa_json_path = PROJECT_ROOT / "trainer" / "data" / taxon_name / "taxa.json"

images_path = PROJECT_ROOT / "trainer" / "images" / taxon_name
target_path = images_path / target_folder
handled_log_path = images_path / "inaturalist_finnishtaxa.log"

ALLOWED_RANKS = {"MX.species", "MX.genus"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_IMAGES_PER_TAXON = 50

# Mirrors https://www.inaturalist.org/observations?hrank=genus&lat=64.893&lng=25.845&
# quality_grade=research&radius=2000&verifiable=any
# The v2 API accepts verifiable only as "true" or "false"; omit it to match "any".
QUERY_PARAMS: dict = {
    "lat": 64.893,
    "lng": 25.845,
    "radius": 2500,
    "quality_grade": "research",
    "hrank": "genus",
    "per_page": 200,
    "order_by": "id",
    "order": "asc",
    "fields": "(photos:(id:!t,url:!t),taxon:(name:!t,rank:!t))",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"


def _headers_json() -> dict:
    return {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def _headers_image() -> dict:
    return {
        "User-Agent": USER_AGENT,
    }


def _observations_url(taxon_id: int, page: int, finbif_rank: str) -> str:
    params = dict(QUERY_PARAMS)
    params["taxon_id"] = taxon_id
    params["page"] = page
    if finbif_rank == "MX.genus":
        params.pop("hrank", None)
        params["rank"] = "genus"
    query = urllib.parse.urlencode(params, doseq=True)
    return f"{OBSERVATIONS_URL}?{query}"


def _http_json(url: str) -> dict:
    print(f"Fetching JSON from {url}")
    req = urllib.request.Request(url, headers=_headers_json())
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_bytes(url: str) -> bytes:
    print(f"Fetching image from {url}")
    req = urllib.request.Request(url, headers=_headers_image())
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _folder_name(scientific_name: str) -> str:
    name = scientific_name.strip().split("(", 1)[0].strip()
    parts = name.split()
    if len(parts) >= 2:
        folder = f"{parts[0]}_{parts[1]}"
    else:
        folder = parts[0]
    return re.sub(r'[<>:"/\\|?*]', "_", folder)


def _square_url_to_large(square_url: str) -> str:
    if "/square." in square_url:
        return square_url.replace("/square.", "/large.")
    return re.sub(r"/[^/]+\.(jpg|jpeg|png)(\?.*)?$", "/large.jpg", square_url, flags=re.I)


def load_existing_photo_ids(root: Path) -> set[int]:
    """Collect numeric photo IDs from image filenames under root (any subfolder)."""
    ids: set[int] = set()
    if not root.is_dir():
        return ids
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = path.stem
        if stem.isdigit():
            ids.add(int(stem))
    return ids


def load_handled_taxa(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            names.add(name)
    return names


def mark_taxon_handled(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")


def query_names_for_taxon(scientific_name: str, synonyms: list[str]) -> list[str]:
    names: list[str] = []
    for name in [scientific_name, *synonyms]:
        name = name.strip()
        if name and name not in names:
            names.append(name)
    return names


def pending_query_names(query_names: list[str], handled: set[str]) -> list[str]:
    return [name for name in query_names if name not in handled]


def load_finbif_taxa() -> list[dict]:
    print(f"Reading FinBIF taxa from {taxa_json_path}")
    with taxa_json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    taxa = []
    for item in data["results"]:
        if item.get("taxonRank") not in ALLOWED_RANKS:
            continue
        name = (item.get("scientificName") or "").strip()
        if not name:
            continue
        synonyms = []
        for syn in item.get("synonyms") or []:
            syn_name = (syn.get("scientificName") or "").strip()
            if syn_name:
                synonyms.append(syn_name)
        taxa.append({
            "scientific_name": name,
            "taxon_rank": item["taxonRank"],
            "synonyms": synonyms,
        })
    print(f"Found {len(taxa)} genera and species")
    return taxa


def lookup_inaturalist_taxon_id(scientific_name: str) -> int | None:
    params = {
        "q": scientific_name,
        "rank": "genus,species",
        "order": "desc",
        "order_by": "observations_count",
    }
    url = f"{TAXA_LOOKUP_URL}?{urllib.parse.urlencode(params)}"
    print(f"Looking up iNaturalist taxon id for '{scientific_name}'")
    data = _http_json(url)

    sleep_sec = random.uniform(0.5, 1)
    time.sleep(sleep_sec)
    print(f"Sleeping for {sleep_sec} seconds")

    for result in data.get("results") or []:
        if result.get("name") == scientific_name:
            taxon_id = result.get("id")
            print(f"Matched '{scientific_name}' to iNaturalist id {taxon_id}")
            return taxon_id
    print(f"No iNaturalist taxon with name '{scientific_name}'")
    return None


def fetch_observations_for_taxon(taxon_id: int, finbif_rank: str) -> list:
    rows = []
    page = 1
    while True:
        url = _observations_url(taxon_id, page, finbif_rank)
        data = _http_json(url)

        sleep_sec = random.uniform(0.5, 1)
        time.sleep(sleep_sec)
        print(f"Sleeping for {sleep_sec} seconds")

        batch = data.get("results") or []
        if not batch:
            break
        rows.extend(batch)
        total = data.get("total_results")
        per_page = data.get("per_page") or len(batch)
        print(f"Page {page}: {len(batch)} observations (total_results={total})")
        if total is not None and page * per_page >= total:
            break
        page += 1
    return rows


def download_observation_photos(
    observations: list,
    existing_ids: set[int],
    seen_ids: set[int],
    scientific_name: str,
) -> tuple[int, int, int]:
    downloaded = 0
    skipped = 0
    already_existed = 0
    photo_index = 0
    folder = _folder_name(scientific_name)
    out_dir = target_path / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    while downloaded < MAX_IMAGES_PER_TAXON:
        had_photo_at_index = False
        for obs in observations:
            photos = obs.get("photos") or []
            if photo_index >= len(photos):
                continue
            had_photo_at_index = True
            photo = photos[photo_index]

            if photo.get("url") is None:
                skipped += 1
                continue
            photo_id = photo.get("id")
            if photo_id is None:
                skipped += 1
                continue
            out_file = out_dir / f"{photo_id}.jpg"
            if photo_id in existing_ids or out_file.is_file():
                already_existed += 1
                continue
            if photo_id in seen_ids:
                continue
            seen_ids.add(photo_id)

            large_url = _square_url_to_large(photo["url"])
            print(f"Scientific name: {scientific_name}, photo {photo_id}")
            print(f"Downloading image from {large_url}, saving to {out_file}")
            out_file.write_bytes(_http_bytes(large_url))
            existing_ids.add(photo_id)
            downloaded += 1

            sleep_sec = random.uniform(1, 3)
            time.sleep(sleep_sec)
            print(f"Sleeping for {sleep_sec} seconds")

            if downloaded >= MAX_IMAGES_PER_TAXON:
                print(f"Reached {MAX_IMAGES_PER_TAXON} images for this taxon")
                return downloaded, skipped, already_existed

        if not had_photo_at_index:
            break
        photo_index += 1

    return downloaded, skipped, already_existed


def main() -> None:
    target_path.mkdir(parents=True, exist_ok=True)
    taxa = load_finbif_taxa()
    existing_ids = load_existing_photo_ids(target_path)
    print(f"Found {len(existing_ids)} existing images under {target_path}")
    handled = load_handled_taxa(handled_log_path)
    print(f"Found {len(handled)} already handled taxa in {handled_log_path}")
    seen_ids: set[int] = set()
    downloaded = 0
    skipped = 0
    already_existed = 0
    unmatched = 0

    for index, taxon in enumerate(taxa, start=1):
        name = taxon["scientific_name"]
        rank = taxon["taxon_rank"]
        query_names = query_names_for_taxon(name, taxon.get("synonyms") or [])
        pending_names = pending_query_names(query_names, handled)
        print(f"\n[{index}/{len(taxa)}] {name} ({rank})")
        print(f"Query names: {', '.join(query_names)}")
        if not pending_names:
            print(f"Already handled, skipping")
            continue
        print(f"Pending query names: {', '.join(pending_names)}")

        taxon_ids: list[int] = []
        seen_taxon_ids: set[int] = set()
        for query_name in pending_names:
            taxon_id = lookup_inaturalist_taxon_id(query_name)
            if taxon_id is None or taxon_id in seen_taxon_ids:
                continue
            seen_taxon_ids.add(taxon_id)
            taxon_ids.append(taxon_id)

        if not taxon_ids:
            unmatched += 1
            mark_taxon_handled(handled_log_path, pending_names)
            handled.update(pending_names)
            continue

        observations = []
        for taxon_id in taxon_ids:
            print(f"Fetching observations for taxon_id={taxon_id}, radius={QUERY_PARAMS['radius']}")
            batch = fetch_observations_for_taxon(taxon_id, rank)
            print(f"Got {len(batch)} observations for taxon_id={taxon_id}")
            observations.extend(batch)
        print(f"Got {len(observations)} observations for {name}")
        if observations:
            new, skip, existed = download_observation_photos(
                observations, existing_ids, seen_ids, name
            )
            downloaded += new
            skipped += skip
            already_existed += existed

        mark_taxon_handled(handled_log_path, pending_names)
        handled.update(pending_names)

    print(
        f"\nDone. Downloaded {downloaded} new images to {target_path}. "
        f"Already existed: {already_existed}. "
        f"No iNaturalist match: {unmatched}. "
        f"Skipped (no photo): {skipped}."
    )


if __name__ == "__main__":
    main()
