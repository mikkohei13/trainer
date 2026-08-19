import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "group_inaturalist_observations.py"
    spec = importlib.util.spec_from_file_location("group_inaturalist_observations", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


group = _load_script()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


class TestPhotoIdFromName(unittest.TestCase):
    def test_numeric_stem(self):
        self.assertEqual(group.photo_id_from_name("123.jpg"), 123)

    def test_rejects_non_numeric(self):
        self.assertIsNone(group.photo_id_from_name("notes.jpg"))
        self.assertIsNone(group.photo_id_from_name("12a.jpg"))


class TestClusterSortedItems(unittest.TestCase):
    def test_groups_nearby_ids(self):
        items = [(10, "a/10.jpg"), (12, "a/12.jpg"), (100, "a/100.jpg")]
        self.assertEqual(
            group.cluster_sorted_items(items, max_gap=5),
            [["a/10.jpg", "a/12.jpg"], ["a/100.jpg"]],
        )

    def test_gap_equal_to_max_is_grouped(self):
        items = [(10, "a/10.jpg"), (30, "a/30.jpg")]
        self.assertEqual(
            group.cluster_sorted_items(items, max_gap=20),
            [["a/10.jpg", "a/30.jpg"]],
        )

    def test_gap_above_max_splits(self):
        items = [(10, "a/10.jpg"), (31, "a/31.jpg")]
        self.assertEqual(
            group.cluster_sorted_items(items, max_gap=20),
            [["a/10.jpg"], ["a/31.jpg"]],
        )

    def test_chains_consecutive_small_gaps(self):
        items = [(1, "a/1.jpg"), (10, "a/10.jpg"), (20, "a/20.jpg")]
        self.assertEqual(
            group.cluster_sorted_items(items, max_gap=10),
            [["a/1.jpg", "a/10.jpg", "a/20.jpg"]],
        )


class TestGroupPhotos(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clusters_within_folder_not_across_folders(self):
        _touch(self.tmp / "Alpha" / "10.jpg")
        _touch(self.tmp / "Alpha" / "12.jpg")
        _touch(self.tmp / "Beta" / "11.jpg")
        _touch(self.tmp / "Beta" / "200.jpg")
        _touch(self.tmp / "Alpha" / "notes.txt")
        _touch(self.tmp / "Alpha" / "not-a-number.jpg")

        groups = group.group_photos(self.tmp, max_gap=5)
        self.assertEqual(
            groups,
            [
                ["Alpha/10.jpg", "Alpha/12.jpg"],
                ["Beta/11.jpg"],
                ["Beta/200.jpg"],
            ],
        )

    def test_reads_nested_subfolders(self):
        _touch(self.tmp / "Alpha" / "nested" / "10.jpeg")
        _touch(self.tmp / "Alpha" / "nested" / "11.PNG")
        groups = group.group_photos(self.tmp, max_gap=5)
        self.assertEqual(groups, [["Alpha/nested/10.jpeg", "Alpha/nested/11.PNG"]])

    def test_payload_roundtrip(self):
        groups = [["Alpha/10.jpg", "Alpha/12.jpg"], ["Beta/11.jpg"]]
        payload = group.build_payload(groups, max_gap=20, source="inaturalist")
        self.assertEqual(payload["observation_count"], 2)
        self.assertEqual(payload["photo_count"], 3)
        self.assertEqual(payload["photo_to_observation"]["Alpha/10.jpg"], 0)
        self.assertEqual(payload["photo_to_observation"]["Alpha/12.jpg"], 0)
        self.assertEqual(payload["photo_to_observation"]["Beta/11.jpg"], 1)

        out = self.tmp / "obs.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        loaded = group.load_observations(out)
        self.assertEqual(loaded["observations"], groups)
        self.assertEqual(loaded["photo_to_observation"]["Alpha/12.jpg"], 0)
