"""
Fetch occurrence images from FinBIF (api.laji.fi) into target_path: one folder per
taxon (Genus_species), files named MM.xxxxx.jpg.

API key: FINBIF_API_TOKEN in secrets.py at project root (see example.secrets.py).
"""

import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("trainer_secrets", PROJECT_ROOT / "secrets.py")
_secrets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_secrets)
TOKEN = _secrets.FINBIF_API_TOKEN.strip()
if not TOKEN:
    raise SystemExit("FINBIF_API_TOKEN is missing or empty in secrets.py")

API_BASE = "https://api.laji.fi"
UNIT_MEDIA_LIST = f"{API_BASE}/warehouse/query/unitMedia/list"

taxon_name = "auchenorrhyncha"
target_folder = "vihko"
taxon_id = "MX.289596"
collection_id = "HR.1747"

target_path = PROJECT_ROOT / "trainer" / "images" / taxon_name / target_folder

QUERY_PARAMS: dict = {
    "page": 1,
    "pageSize": 1000,
    "cache": "true",
    "target": taxon_id,
    "countryId": "ML.206",
    "collectionId": collection_id,
    "recordQuality": "EXPERT_VERIFIED,COMMUNITY_VERIFIED,NEUTRAL",
    "taxonRankId": "MX.genus,MX.species",
    "hasUnitImages": "true",
    "needsCheck": "false",
    "selected": "media.fullURL,media.mediaType,unit.linkings.taxon.scientificName",
}

IMAGE_ID_RE = re.compile(r"\b(MM\.\d+)\b")
REQUEST_DELAY_SEC = 1.0


def _headers_json() -> dict:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "API-Version": "1",
        "Accept": "application/json",
        "Accept-Language": "en",
    }


def _headers_image() -> dict:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "API-Version": "1",
    }


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


def fetch_all_rows() -> list:
    rows = []
    page = 1
    while True:
        params = dict(QUERY_PARAMS)
        params["page"] = page
        url = f"{UNIT_MEDIA_LIST}?{urllib.parse.urlencode(params, doseq=True)}"
        data = _http_json(url)
        time.sleep(REQUEST_DELAY_SEC)
        rows.extend(data["results"])
        next_page = data.get("nextPage")
        if next_page is None:
            break
        page = int(next_page)
    return rows


def main() -> None:
    target_path.mkdir(parents=True, exist_ok=True)
    rows = fetch_all_rows()
    seen_ids: set[str] = set()
    downloaded = 0
    skipped = 0

    for row in rows:
        print(f"Processing row: {row}")

        media = row["media"]
        if media["mediaType"] != "IMAGE":
            skipped += 1
            continue
        match = IMAGE_ID_RE.search(media["fullURL"])
        if match is None:
            skipped += 1
            continue
        image_id = match.group(1)
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        scientific = row["unit"]["linkings"]["taxon"]["scientificName"]
        print(f"Scientific name: {scientific}")

        folder = _folder_name(scientific)
        out_dir = target_path / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{image_id}.jpg"
        if out_file.is_file():
            continue

        img_url = f"{API_BASE}/images/{image_id}/large.jpg"

        print(f"Downloading image from {img_url}, saving to {out_file}")
        out_file.write_bytes(_http_bytes(img_url))
        downloaded += 1
        time.sleep(REQUEST_DELAY_SEC)

    print(
        f"Done. Downloaded {downloaded} new images to {target_path}. "
        f"Skipped (non-image / no id in URL / duplicates in feed): {skipped}."
    )


if __name__ == "__main__":
    main()
