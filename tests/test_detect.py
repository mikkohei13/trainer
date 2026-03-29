import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
