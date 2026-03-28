import os
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import trainer.db as db
import trainer.images as images
import trainer.training as training


def _make_image(path: Path, width: int, height: int) -> None:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path)


class TestExportYoloDataset(unittest.TestCase):
    def setUp(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._db_file = db_path
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = Path(db_path)
        db.init_db()

        self._tmp_images = Path(tempfile.mkdtemp())
        self._orig_images_dir = images.IMAGES_DIR
        images.IMAGES_DIR = self._tmp_images
        training.IMAGES_DIR = self._tmp_images

        taxon_dir = self._tmp_images / "bugs" / "col"
        taxon_dir.mkdir(parents=True)
        for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"):
            _make_image(taxon_dir / name, width=200, height=100)

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        images.IMAGES_DIR = self._orig_images_dir
        training.IMAGES_DIR = self._orig_images_dir
        os.unlink(self._db_file)
        shutil.rmtree(self._tmp_images, ignore_errors=True)

    def test_label_format_with_boxes(self):
        db.save_annotations(
            "bugs/col/a.jpg",
            [{"anno_id": "1", "x": 20.0, "y": 10.0, "w": 40.0, "h": 30.0}],
            False,
        )
        output_dir = Path(tempfile.mkdtemp())
        try:
            training.export_yolo_dataset("bugs", output_dir)
            label_files = list((output_dir / "labels").rglob("*.txt"))
            annotated = [f for f in label_files if f.read_text().strip()]
            self.assertEqual(len(annotated), 1)
            content = annotated[0].read_text().strip()
            parts = content.split()
            self.assertEqual(len(parts), 5)
            self.assertEqual(parts[0], "0")
            # cx = (20 + 40/2) / 200 = 0.2
            self.assertAlmostEqual(float(parts[1]), 0.2, places=4)
            # cy = (10 + 30/2) / 100 = 0.25
            self.assertAlmostEqual(float(parts[2]), 0.25, places=4)
            # bw = 40 / 200 = 0.2
            self.assertAlmostEqual(float(parts[3]), 0.2, places=4)
            # bh = 30 / 100 = 0.3
            self.assertAlmostEqual(float(parts[4]), 0.3, places=4)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_empty_label_for_no_organism(self):
        db.save_annotations("bugs/col/b.jpg", [], True)
        output_dir = Path(tempfile.mkdtemp())
        try:
            training.export_yolo_dataset("bugs", output_dir)
            label_files = list((output_dir / "labels").rglob("*.txt"))
            self.assertEqual(len(label_files), 1)
            self.assertEqual(label_files[0].read_text(), "")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_train_val_split(self):
        for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"):
            db.save_annotations(
                f"bugs/col/{name}",
                [{"anno_id": "1", "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}],
                False,
            )
        output_dir = Path(tempfile.mkdtemp())
        try:
            train_count, val_count = training.export_yolo_dataset("bugs", output_dir)
            self.assertEqual(train_count + val_count, 5)
            self.assertGreaterEqual(train_count, 1)
            train_images = list((output_dir / "images" / "train").iterdir())
            self.assertEqual(len(train_images), train_count)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_dataset_yaml(self):
        db.save_annotations("bugs/col/c.jpg", [], True)
        output_dir = Path(tempfile.mkdtemp())
        try:
            training.export_yolo_dataset("bugs", output_dir)
            yaml_text = (output_dir / "dataset.yaml").read_text()
            self.assertIn("nc: 1", yaml_text)
            self.assertIn("names: [organism]", yaml_text)
            self.assertIn("train: images/train", yaml_text)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_unannotated_images_excluded(self):
        # Only annotate one; the rest should be excluded
        db.save_annotations("bugs/col/a.jpg", [], True)
        output_dir = Path(tempfile.mkdtemp())
        try:
            train_count, val_count = training.export_yolo_dataset("bugs", output_dir)
            self.assertEqual(train_count + val_count, 1)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_no_annotated_images_returns_zero(self):
        output_dir = Path(tempfile.mkdtemp())
        try:
            train_count, val_count = training.export_yolo_dataset("bugs", output_dir)
            self.assertEqual(train_count, 0)
            self.assertEqual(val_count, 0)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_flat_filename_avoids_collision(self):
        """Images from different subdirs with the same basename become unique filenames."""
        taxon_dir2 = self._tmp_images / "bugs" / "col2"
        taxon_dir2.mkdir(parents=True)
        _make_image(taxon_dir2 / "a.jpg", width=100, height=50)

        db.save_annotations("bugs/col/a.jpg", [], True)
        db.save_annotations("bugs/col2/a.jpg", [], True)

        output_dir = Path(tempfile.mkdtemp())
        try:
            train_count, val_count = training.export_yolo_dataset("bugs", output_dir)
            all_images = list((output_dir / "images").rglob("*.jpg"))
            names = [f.name for f in all_images]
            self.assertEqual(len(names), len(set(names)), "Filenames should be unique")
            self.assertEqual(train_count + val_count, 2)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
