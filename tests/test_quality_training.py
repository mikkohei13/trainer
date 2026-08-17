import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import trainer.db as db
import trainer.images as images
import trainer.inference as inference
import trainer.quality_training as quality_training


def _make_image(path: Path, width: int, height: int) -> None:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path)


class TestQualityCropBox(unittest.TestCase):
    def test_single_box(self):
        box = inference.quality_crop_box([
            {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0, "conf": 0.9},
        ])
        self.assertEqual(box, {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0})

    def test_multiple_boxes(self):
        box = inference.quality_crop_box([
            {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0, "conf": 0.9},
            {"x": 50.0, "y": 5.0, "w": 20.0, "h": 15.0, "conf": 0.5},
        ])
        self.assertEqual(box["x"], 10.0)
        self.assertEqual(box["y"], 5.0)
        self.assertEqual(box["w"], 60.0)
        self.assertEqual(box["h"], 55.0)

    def test_filters_boxes_below_threshold(self):
        box = inference.quality_crop_box([
            {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0, "conf": 0.9},
            {"x": 50.0, "y": 5.0, "w": 20.0, "h": 15.0, "conf": 0.2},
        ])
        self.assertEqual(box, {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0})

    def test_none_when_all_below_threshold(self):
        box = inference.quality_crop_box([
            {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0, "conf": 0.2},
        ])
        self.assertIsNone(box)


class TestCropBoxWithPadding(unittest.TestCase):
    def test_pads_union_box_by_ten_percent(self):
        img = Image.new("RGB", (200, 100), color=(0, 0, 0))
        crop = quality_training._crop_box_with_padding(
            img,
            {"x": 20.0, "y": 10.0, "w": 40.0, "h": 30.0},
            0.10,
        )
        self.assertEqual(crop.size, (48, 36))


class TestCollectQualityRecords(unittest.TestCase):
    def setUp(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._db_file = db_path
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = Path(db_path)
        db.init_db()

        self._tmp_images = Path(tempfile.mkdtemp())
        self._orig_images_dir = images.IMAGES_DIR
        self._orig_qt_images = quality_training.IMAGES_DIR
        images.IMAGES_DIR = self._tmp_images
        quality_training.IMAGES_DIR = self._tmp_images

        db.create_project("bugs")
        project = db.get_project("bugs")
        det_run = db.create_training_run(project["id"])
        fd_pt, pt_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd_pt)
        Path(pt_path).write_bytes(b"x")
        self._det_weights = pt_path
        db.finish_training_run(det_run, pt_path, None, None, "/tmp/log")
        db.set_active_training_run("bugs", det_run)

        taxon_dir = self._tmp_images / "bugs" / "col"
        taxon_dir.mkdir(parents=True)
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            _make_image(taxon_dir / name, width=200, height=100)

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        images.IMAGES_DIR = self._orig_images_dir
        quality_training.IMAGES_DIR = self._orig_qt_images
        os.unlink(self._db_file)
        os.unlink(self._det_weights)
        shutil.rmtree(self._tmp_images, ignore_errors=True)

    def test_requires_active_detection_model(self):
        db.set_active_training_run("bugs", None)
        db.save_image_quality("bugs/col/a.jpg", 1.0)
        with self.assertRaises(ValueError) as ctx:
            quality_training._collect_quality_records("bugs")
        self.assertIn("No active object-detection model", str(ctx.exception))

    @patch("trainer.quality_training.predict_boxes")
    def test_skips_images_without_detection(self, mock_predict):
        db.save_image_quality("bugs/col/a.jpg", 1.0)
        db.save_image_quality("bugs/col/b.jpg", 0.0)

        def _boxes(model_path, image_abs_path, conf=0.1):
            self.assertEqual(conf, quality_training.DETECTION_CONF_THRESHOLD)
            if image_abs_path.name == "a.jpg":
                return [{"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0, "conf": 0.8}]
            return []

        mock_predict.side_effect = _boxes
        records = quality_training._collect_quality_records("bugs")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["image_path"], "bugs/col/a.jpg")
        self.assertEqual(records[0]["quality"], 1.0)

    @patch("trainer.quality_training.predict_boxes")
    def test_uses_union_of_detected_boxes(self, mock_predict):
        db.save_image_quality("bugs/col/a.jpg", 0.666)
        mock_predict.return_value = [
            {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0, "conf": 0.9},
            {"x": 50.0, "y": 5.0, "w": 20.0, "h": 15.0, "conf": 0.5},
        ]
        records = quality_training._collect_quality_records("bugs")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["box"]["x"], 10.0)
        self.assertEqual(records[0]["box"]["y"], 5.0)
        self.assertEqual(records[0]["box"]["w"], 60.0)
        self.assertEqual(records[0]["box"]["h"], 55.0)
        mock_predict.assert_called_once()
        self.assertEqual(
            mock_predict.call_args.kwargs["conf"],
            quality_training.DETECTION_CONF_THRESHOLD,
        )


class TestValDiagnostics(unittest.TestCase):
    def test_nearest_rating(self):
        self.assertEqual(quality_training._nearest_rating(0.0), 0.0)
        self.assertEqual(quality_training._nearest_rating(0.31), 0.333)
        self.assertEqual(quality_training._nearest_rating(0.8), 0.666)
        self.assertEqual(quality_training._nearest_rating(0.95), 1.0)

    def test_image_source(self):
        self.assertEqual(
            quality_training._image_source("auchenorrhyncha/inaturalist/foo.jpg"),
            "inaturalist",
        )
        self.assertEqual(quality_training._image_source("solo.jpg"), "unknown")

    def test_logs_mae_and_confusion(self):
        messages: list[str] = []
        records = [
            {"image_path": "bugs/inaturalist/a.jpg"},
            {"image_path": "bugs/vihko/b.jpg"},
        ]
        quality_training._log_val_diagnostics(
            messages.append,
            1,
            records,
            [1.0, 0.3330000042915344],
            [0.9, 0.4],
        )
        joined = "\n".join(messages)
        self.assertIn("val mae=", joined)
        self.assertIn("0.333=", joined)
        self.assertNotIn("0.3330000042915344", joined)
        self.assertIn("val mae by rating:", joined)
        self.assertIn("inaturalist=", joined)
        self.assertIn("vihko=", joined)
        self.assertIn("val confusion", joined)


if __name__ == "__main__":
    unittest.main()
