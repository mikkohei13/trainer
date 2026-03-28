import os
import shutil
import tempfile
import unittest
from pathlib import Path

import trainer.db as db
import trainer.images as images


class TestAnnotations(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._db_file = path
        self._orig_path = db.DB_PATH
        db.DB_PATH = Path(path)
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self._orig_path
        os.unlink(self._db_file)

    def test_round_trip_boxes(self):
        image_path = "taxon/collection/sp/foo.jpg"
        db.save_annotations(
            image_path,
            [
                {
                    "anno_id": "a1",
                    "x": 1.0,
                    "y": 2.0,
                    "w": 10.0,
                    "h": 20.0,
                },
            ],
            False,
        )
        data = db.get_annotations(image_path)
        self.assertFalse(data["no_organism"])
        self.assertEqual(len(data["boxes"]), 1)
        b = data["boxes"][0]
        self.assertEqual(b["anno_id"], "a1")
        self.assertEqual(b["x"], 1.0)
        self.assertEqual(b["y"], 2.0)
        self.assertEqual(b["w"], 10.0)
        self.assertEqual(b["h"], 20.0)

    def test_no_organism_flag(self):
        image_path = "taxon/a.jpg"
        db.save_annotations(image_path, [], True)
        data = db.get_annotations(image_path)
        self.assertTrue(data["no_organism"])
        self.assertEqual(data["boxes"], [])

    def test_replace_clears_previous_state(self):
        image_path = "taxon/b.jpg"
        db.save_annotations(
            image_path,
            [{"anno_id": "x", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}],
            False,
        )
        db.save_annotations(image_path, [], True)
        data = db.get_annotations(image_path)
        self.assertTrue(data["no_organism"])
        self.assertEqual(data["boxes"], [])

        db.save_annotations(
            image_path,
            [{"anno_id": "y", "x": 5.0, "y": 5.0, "w": 2.0, "h": 3.0}],
            False,
        )
        data2 = db.get_annotations(image_path)
        self.assertFalse(data2["no_organism"])
        self.assertEqual(len(data2["boxes"]), 1)
        self.assertEqual(data2["boxes"][0]["anno_id"], "y")

    def test_project_annotation_state_prefix(self):
        db.save_annotations("mytaxon/a.jpg", [], True)
        db.save_annotations(
            "mytaxon/b.jpg",
            [{"anno_id": "1", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}],
            False,
        )
        db.save_annotations(
            "mytaxon/c.jpg",
            [
                {"anno_id": "1", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
                {"anno_id": "2", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            ],
            False,
        )
        no_set, box_map = db.project_annotation_state("mytaxon")
        self.assertEqual(no_set, {"mytaxon/a.jpg"})
        self.assertEqual(box_map["mytaxon/b.jpg"], 1)
        self.assertEqual(box_map["mytaxon/c.jpg"], 2)

    def test_project_annotation_distribution_buckets(self):
        _orig_images = images.IMAGES_DIR
        tmp = Path(tempfile.mkdtemp())
        try:
            images.IMAGES_DIR = tmp
            (tmp / "t" / "c").mkdir(parents=True)
            for name in ("u.jpg", "z.jpg", "one.jpg", "two.jpg"):
                (tmp / "t" / "c" / name).write_bytes(b"x")

            db.save_annotations("t/c/u.jpg", [], True)
            db.save_annotations(
                "t/c/one.jpg",
                [{"anno_id": "a", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}],
                False,
            )
            db.save_annotations(
                "t/c/two.jpg",
                [
                    {"anno_id": "a", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
                    {"anno_id": "b", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
                ],
                False,
            )
            stats = images.project_stats("t")
            ann = stats["annotation"]
            self.assertEqual(ann["total_images"], 4)
            self.assertEqual(ann["not_annotated"], 1)
            by = {r["organisms"]: r["count"] for r in ann["by_organism_count"]}
            self.assertEqual(by[0], 1)
            self.assertEqual(by[1], 1)
            self.assertEqual(by[2], 1)
            buckets = images.project_annotation_buckets("t")
            self.assertEqual(
                buckets["not_annotated"],
                ["t/c/z.jpg"],
            )
            self.assertEqual(buckets["0"], ["t/c/u.jpg"])
            self.assertEqual(buckets["1"], ["t/c/one.jpg"])
            self.assertEqual(buckets["2"], ["t/c/two.jpg"])
        finally:
            images.IMAGES_DIR = _orig_images
            shutil.rmtree(tmp, ignore_errors=True)

    def test_normalize_annotation_bucket(self):
        self.assertEqual(images.normalize_annotation_bucket("not_annotated"), "not_annotated")
        self.assertEqual(images.normalize_annotation_bucket("03"), "3")
        self.assertEqual(images.normalize_annotation_bucket("0"), "0")
        self.assertIsNone(images.normalize_annotation_bucket("bad"))
        self.assertIsNone(images.normalize_annotation_bucket(None))


if __name__ == "__main__":
    unittest.main()
