"""
Fetch observation photos from iNaturalist API v2 (api.inaturalist.org) into
target_path: one folder per taxon (Genus_species), files named <photo_id>.jpg.

API reference: https://api.inaturalist.org/v2/docs/

Read-only search does not require an API key.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OBSERVATIONS_URL = "https://api.inaturalist.org/v2/observations"

taxon_name = "auchenorrhyncha"
target_folder = "inaturalist"

target_path = PROJECT_ROOT / "trainer" / "images" / taxon_name / target_folder

# Mirrors https://www.inaturalist.org/observations?hrank=genus&lat=64.893&lng=25.845&
# quality_grade=research&radius=700&taxon_id=125816&verifiable=any
# The v2 API accepts verifiable only as "true" or "false"; omit it to match "any".
QUERY_PARAMS: dict = {
    "taxon_id": 125816,
    "lat": 64.893,
    "lng": 25.845,
    "radius": 700,
    "quality_grade": "research",
    "hrank": "genus",
    "per_page": 200,
    "order_by": "id",
    "order": "asc",
    "fields": "(photos:(id:!t,url:!t),taxon:(name:!t,rank:!t))",
}

REQUEST_DELAY_SEC = 1.0

USER_AGENT = "trainer-fetch-inaturalist/1.0 (+local dev)"


def _headers_json() -> dict:
    return {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def _headers_image() -> dict:
    return {
        "User-Agent": USER_AGENT,
    }


def _observations_url(page: int) -> str:
    params = dict(QUERY_PARAMS)
    params["page"] = page
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


def fetch_all_observations() -> list:
    rows = []
    page = 1
    while True:
        url = _observations_url(page)
        data = _http_json(url)
        time.sleep(REQUEST_DELAY_SEC)
        batch = data.get("results") or []
        if not batch:
            break
        rows.extend(batch)
        total = data.get("total_results")
        per_page = data.get("per_page") or len(batch)
        if total is not None and page * per_page >= total:
            break
        page += 1
    return rows


def main() -> None:
    target_path.mkdir(parents=True, exist_ok=True)
    observations = fetch_all_observations()
    seen_ids: set[int] = set()
    downloaded = 0
    skipped = 0

    for obs in observations:
        photos = obs.get("photos") or []
        taxon = obs.get("taxon") or {}
        scientific = taxon.get("name")
        if not scientific:
            skipped += 1
            continue

        folder = _folder_name(scientific)
        out_dir = target_path / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        for photo in photos:
            if photo.get("url") is None:
                skipped += 1
                continue
            photo_id = photo.get("id")
            if photo_id is None:
                skipped += 1
                continue
            if photo_id in seen_ids:
                continue
            seen_ids.add(photo_id)

            out_file = out_dir / f"{photo_id}.jpg"
            if out_file.is_file():
                continue

            large_url = _square_url_to_large(photo["url"])
            print(f"Scientific name: {scientific}, photo {photo_id}")
            print(f"Downloading image from {large_url}, saving to {out_file}")
            out_file.write_bytes(_http_bytes(large_url))
            downloaded += 1
            time.sleep(REQUEST_DELAY_SEC)

    print(
        f"Done. Downloaded {downloaded} new images to {target_path}. "
        f"Skipped (no taxon name / no photo / duplicates in feed): {skipped}."
    )


if __name__ == "__main__":
    main()
