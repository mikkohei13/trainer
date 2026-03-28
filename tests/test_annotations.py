import os
import tempfile
import unittest
from pathlib import Path

import trainer.db as db


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


if __name__ == "__main__":
    unittest.main()
