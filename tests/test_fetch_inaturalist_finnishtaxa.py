import importlib.util
import shutil
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "fetch_inaturalist_finnishtaxa.py"
    spec = importlib.util.spec_from_file_location("fetch_inaturalist_finnishtaxa", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fetch = _load_script()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


class TestLoadExistingPhotoIds(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_collects_ids_recursively_across_subfolders(self):
        _touch(self.tmp / "Cixius" / "111.jpg")
        _touch(self.tmp / "Cixius_nervosus" / "222.jpeg")
        _touch(self.tmp / "nested" / "deeper" / "333.PNG")
        _touch(self.tmp / "Cixius" / "notes.txt")
        _touch(self.tmp / "Cixius" / "not-a-number.jpg")

        ids = fetch.load_existing_photo_ids(self.tmp)
        self.assertEqual(ids, {111, 222, 333})

    def test_missing_root_returns_empty(self):
        self.assertEqual(fetch.load_existing_photo_ids(self.tmp / "missing"), set())


class TestObservationsUrlByRank(unittest.TestCase):
    def test_genus_uses_exact_rank_filter(self):
        url = fetch._observations_url(342233, 1, "MX.genus")
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(params["taxon_id"], ["342233"])
        self.assertEqual(params["rank"], ["genus"])
        self.assertEqual(params["quality_grade"], ["research"])
        self.assertNotIn("hrank", params)

    def test_species_keeps_hrank_genus(self):
        url = fetch._observations_url(512498, 2, "MX.species")
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(params["taxon_id"], ["512498"])
        self.assertEqual(params["page"], ["2"])
        self.assertEqual(params["hrank"], ["genus"])
        self.assertEqual(params["quality_grade"], ["research"])
        self.assertNotIn("rank", params)


class TestDownloadSkipsExistingIds(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_target = fetch.target_path
        fetch.target_path = self.tmp

    def tearDown(self):
        fetch.target_path = self._orig_target
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skips_photo_already_present_in_another_folder(self):
        _touch(self.tmp / "Other_taxon" / "999.jpg")
        existing_ids = {999}
        seen_ids: set[int] = set()
        observations = [
            {
                "taxon": {"name": "Cixius nervosus"},
                "photos": [{"id": 999, "url": "https://example.com/photos/999/square.jpg"}],
            }
        ]

        with patch.object(fetch, "_http_bytes") as http_bytes:
            downloaded, skipped, already_existed = fetch.download_observation_photos(
                observations, existing_ids, seen_ids
            )

        http_bytes.assert_not_called()
        self.assertEqual(downloaded, 0)
        self.assertEqual(skipped, 0)
        self.assertEqual(already_existed, 1)
        self.assertFalse((self.tmp / "Cixius_nervosus" / "999.jpg").exists())

    def test_downloads_new_photo_and_records_id(self):
        existing_ids: set[int] = set()
        seen_ids: set[int] = set()
        observations = [
            {
                "taxon": {"name": "Cixius"},
                "photos": [{"id": 42, "url": "https://example.com/photos/42/square.jpg"}],
            }
        ]

        with patch.object(fetch, "_http_bytes", return_value=b"img") as http_bytes:
            with patch.object(fetch.time, "sleep"):
                downloaded, skipped, already_existed = fetch.download_observation_photos(
                    observations, existing_ids, seen_ids
                )

        http_bytes.assert_called_once()
        self.assertEqual(downloaded, 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(already_existed, 0)
        self.assertEqual(existing_ids, {42})
        self.assertEqual(seen_ids, {42})
        self.assertEqual((self.tmp / "Cixius" / "42.jpg").read_bytes(), b"img")

    def test_stops_after_max_new_saves_ignoring_existing_on_disk(self):
        existing_ids = {1}
        seen_ids: set[int] = set()
        observations = [
            {
                "taxon": {"name": "Cixius"},
                "photos": [
                    {"id": 1, "url": "https://example.com/photos/1/square.jpg"},
                    {"id": 2, "url": "https://example.com/photos/2/square.jpg"},
                    {"id": 3, "url": "https://example.com/photos/3/square.jpg"},
                    {"id": 4, "url": "https://example.com/photos/4/square.jpg"},
                ],
            }
        ]

        with patch.object(fetch, "MAX_IMAGES_PER_TAXON", 2):
            with patch.object(fetch, "_http_bytes", return_value=b"img") as http_bytes:
                with patch.object(fetch.time, "sleep"):
                    downloaded, skipped, already_existed = fetch.download_observation_photos(
                        observations, existing_ids, seen_ids
                    )

        self.assertEqual(http_bytes.call_count, 2)
        self.assertEqual(downloaded, 2)
        self.assertEqual(already_existed, 1)
        self.assertTrue((self.tmp / "Cixius" / "2.jpg").is_file())
        self.assertTrue((self.tmp / "Cixius" / "3.jpg").is_file())
        self.assertFalse((self.tmp / "Cixius" / "4.jpg").exists())


if __name__ == "__main__":
    unittest.main()
