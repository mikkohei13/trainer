import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import trainer.db as db
import trainer.images as images
from trainer.app import app
from trainer.routes import api as api_routes


class TestActiveTrainingRun(unittest.TestCase):
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

    def test_migration_adds_active_training_run_id(self):
        with sqlite3.connect(self._db_file) as con:
            cur = con.execute("PRAGMA table_info(project)")
            names = {row[1] for row in cur.fetchall()}
        self.assertIn("active_training_run_id", names)

    def test_set_active_and_clear(self):
        db.create_project("bugs")
        project = db.get_project("bugs")
        run_id = db.create_training_run(project["id"])
        fd, pt_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        Path(pt_path).write_bytes(b"x")
        try:
            db.finish_training_run(run_id, pt_path, None, None, "/tmp/log")
            db.set_active_training_run("bugs", run_id)
            p2 = db.get_project("bugs")
            self.assertEqual(p2["active_training_run_id"], run_id)
            resolved = db.get_active_model_path_for_taxon("bugs")
            self.assertEqual(
                os.path.realpath(resolved),
                os.path.realpath(pt_path),
            )
            db.set_active_training_run("bugs", None)
            p3 = db.get_project("bugs")
            self.assertIsNone(p3["active_training_run_id"])
        finally:
            os.unlink(pt_path)

    def test_set_active_rejects_wrong_project(self):
        db.create_project("a")
        db.create_project("b")
        project_a = db.get_project("a")
        run_id = db.create_training_run(project_a["id"])
        fd, pt_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        Path(pt_path).write_bytes(b"x")
        try:
            db.finish_training_run(run_id, pt_path, None, None, "/tmp/log")
            with self.assertRaises(ValueError):
                db.set_active_training_run("b", run_id)
        finally:
            os.unlink(pt_path)

    def test_set_active_rejects_non_done_run(self):
        db.create_project("bugs")
        project = db.get_project("bugs")
        run_id = db.create_training_run(project["id"])
        with self.assertRaises(ValueError):
            db.set_active_training_run("bugs", run_id)


class TestDetectApi(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._db_file = path
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = Path(path)
        db.init_db()

        self._tmp_images = Path(tempfile.mkdtemp())
        self._orig_images_dir = images.IMAGES_DIR
        self._orig_api_images = api_routes.IMAGES_DIR
        images.IMAGES_DIR = self._tmp_images
        api_routes.IMAGES_DIR = self._tmp_images

        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

        db.create_project("bugs")
        project = db.get_project("bugs")
        run_id = db.create_training_run(project["id"])
        fd_pt, pt_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd_pt)
        Path(pt_path).write_bytes(b"x")
        self._weights_path = pt_path
        db.finish_training_run(run_id, pt_path, None, None, "/tmp/log")
        db.set_active_training_run("bugs", run_id)

        col = self._tmp_images / "bugs" / "col"
        col.mkdir(parents=True)
        self._img_rel = "bugs/col/a.jpg"
        (col / "a.jpg").write_bytes(b"\xff\xd8\xff")

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        images.IMAGES_DIR = self._orig_images_dir
        api_routes.IMAGES_DIR = self._orig_api_images
        os.unlink(self._db_file)
        shutil.rmtree(self._tmp_images, ignore_errors=True)
        try:
            os.unlink(self._weights_path)
        except OSError:
            pass

    def test_detect_no_active_model(self):
        db.set_active_training_run("bugs", None)
        res = self.client.post(
            "/api/projects/bugs/detect",
            json={"image_path": self._img_rel},
        )
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    @patch("trainer.routes.api.inference.predict_top_box")
    def test_detect_returns_boxes(self, mock_predict):
        mock_predict.return_value = [{"x": 1.0, "y": 2.0, "w": 10.0, "h": 20.0}]
        res = self.client.post(
            "/api/projects/bugs/detect",
            json={"image_path": self._img_rel},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data["boxes"]), 1)
        b = data["boxes"][0]
        self.assertEqual(b["x"], 1.0)
        self.assertEqual(b["y"], 2.0)
        self.assertEqual(b["w"], 10.0)
        self.assertEqual(b["h"], 20.0)
        mock_predict.assert_called_once()
        call_args = mock_predict.call_args[0]
        self.assertEqual(
            os.path.realpath(call_args[0]),
            os.path.realpath(self._weights_path),
        )
        self.assertEqual(call_args[1], self._tmp_images / self._img_rel)

    def test_detect_unknown_project(self):
        res = self.client.post(
            "/api/projects/missing/detect",
            json={"image_path": self._img_rel},
        )
        self.assertEqual(res.status_code, 404)


class TestEvaluateImageApi(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._db_file = path
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = Path(path)
        db.init_db()

        self._tmp_images = Path(tempfile.mkdtemp())
        self._orig_images_dir = images.IMAGES_DIR
        self._orig_api_images = api_routes.IMAGES_DIR
        images.IMAGES_DIR = self._tmp_images
        api_routes.IMAGES_DIR = self._tmp_images

        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

        db.create_project("bugs")
        project = db.get_project("bugs")

        det_run = db.create_training_run(project["id"])
        fd_pt, pt_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd_pt)
        Path(pt_path).write_bytes(b"x")
        self._det_weights = pt_path
        db.finish_training_run(det_run, pt_path, None, None, "/tmp/log")
        db.set_active_training_run("bugs", det_run)

        q_run = db.create_quality_training_run(project["id"])
        fd_q, q_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd_q)
        Path(q_path).write_bytes(b"x")
        self._quality_weights = q_path
        db.finish_quality_training_run(q_run, q_path, 0.1, "/tmp/qlog")
        db.set_active_quality_run("bugs", q_run)

        col = self._tmp_images / "bugs" / "col"
        col.mkdir(parents=True)
        self._img_rel = "bugs/col/a.jpg"
        Image.new("RGB", (20, 10), "black").save(col / "a.jpg")

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        images.IMAGES_DIR = self._orig_images_dir
        api_routes.IMAGES_DIR = self._orig_api_images
        os.unlink(self._db_file)
        shutil.rmtree(self._tmp_images, ignore_errors=True)
        for p in (self._det_weights, self._quality_weights):
            try:
                os.unlink(p)
            except OSError:
                pass

    @patch("trainer.routes.api.inference.predict_quality_score")
    @patch("trainer.routes.api.inference.predict_boxes")
    def test_evaluate_returns_all_boxes_with_confidence(self, mock_boxes, mock_quality):
        top = {"x": 1.0, "y": 2.0, "w": 10.0, "h": 8.0, "conf": 0.91}
        other = {"x": 5.0, "y": 1.0, "w": 4.0, "h": 4.0, "conf": 0.22}
        mock_boxes.return_value = [top, other]
        mock_quality.return_value = 0.75

        res = self.client.post(
            "/api/projects/bugs/evaluate-image",
            json={"image_path": self._img_rel},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["boxes"], [top, other])
        self.assertEqual(data["quality_score"], 0.75)
        self.assertEqual(data["img_w"], 20)
        self.assertEqual(data["img_h"], 10)

        mock_boxes.assert_called_once()
        self.assertEqual(mock_boxes.call_args.kwargs.get("conf"), 0.1)
        mock_quality.assert_called_once()
        self.assertEqual(
            mock_quality.call_args[0][2],
            {"x": 1.0, "y": 2.0, "w": 10.0, "h": 8.0},
        )

    @patch("trainer.routes.api.inference.predict_quality_score")
    @patch("trainer.routes.api.inference.predict_boxes")
    def test_evaluate_scores_union_of_confident_boxes(self, mock_boxes, mock_quality):
        a = {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0, "conf": 0.9}
        b = {"x": 50.0, "y": 5.0, "w": 20.0, "h": 15.0, "conf": 0.5}
        mock_boxes.return_value = [a, b]
        mock_quality.return_value = 0.4

        res = self.client.post(
            "/api/projects/bugs/evaluate-image",
            json={"image_path": self._img_rel},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["boxes"], [a, b])
        self.assertEqual(data["quality_score"], 0.4)
        mock_quality.assert_called_once()
        self.assertEqual(
            mock_quality.call_args[0][2],
            {"x": 10.0, "y": 5.0, "w": 60.0, "h": 55.0},
        )

    @patch("trainer.routes.api.inference.predict_quality_score")
    @patch("trainer.routes.api.inference.predict_boxes")
    def test_evaluate_quality_none_when_only_low_confidence_boxes(self, mock_boxes, mock_quality):
        low = {"x": 1.0, "y": 2.0, "w": 10.0, "h": 8.0, "conf": 0.22}
        mock_boxes.return_value = [low]

        res = self.client.post(
            "/api/projects/bugs/evaluate-image",
            json={"image_path": self._img_rel},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["boxes"], [low])
        self.assertIsNone(data["quality_score"])
        mock_quality.assert_not_called()

    @patch("trainer.routes.api.inference.predict_quality_score")
    @patch("trainer.routes.api.inference.predict_boxes")
    def test_evaluate_no_detection(self, mock_boxes, mock_quality):
        mock_boxes.return_value = []
        res = self.client.post(
            "/api/projects/bugs/evaluate-image",
            json={"image_path": self._img_rel},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["boxes"], [])
        self.assertIsNone(data["quality_score"])
        mock_quality.assert_not_called()


class TestPredictBoxes(unittest.TestCase):
    @patch("trainer.inference._get_yolo")
    def test_returns_all_boxes_sorted_by_confidence(self, mock_get_yolo):
        import torch
        from trainer.inference import predict_boxes

        class FakeBoxes:
            def __init__(self):
                self.xyxy = torch.tensor([
                    [1.0, 2.0, 11.0, 10.0],
                    [20.0, 3.0, 24.0, 7.0],
                ])
                self.conf = torch.tensor([0.22, 0.91])

            def __len__(self):
                return 2

        class FakeResult:
            boxes = FakeBoxes()

        model = mock_get_yolo.return_value
        model.predict.return_value = [FakeResult()]

        boxes = predict_boxes(Path("/tmp/fake.pt"), Path("/tmp/img.jpg"), conf=0.1)
        self.assertEqual(len(boxes), 2)
        self.assertAlmostEqual(boxes[0]["conf"], 0.91, places=5)
        self.assertAlmostEqual(boxes[0]["x"], 20.0)
        self.assertAlmostEqual(boxes[0]["y"], 3.0)
        self.assertAlmostEqual(boxes[0]["w"], 4.0)
        self.assertAlmostEqual(boxes[0]["h"], 4.0)
        self.assertAlmostEqual(boxes[1]["conf"], 0.22, places=5)
        model.predict.assert_called_once()
        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.1)

