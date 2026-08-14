import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import trainer.db as db
import trainer.images as images


class TestIdentificationRank(unittest.TestCase):
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

    def test_migration_adds_identification_rank(self):
        with sqlite3.connect(self._db_file) as con:
            cur = con.execute("PRAGMA table_info(project)")
            names = {row[1] for row in cur.fetchall()}
        self.assertIn("identification_rank", names)

    def test_migration_sets_species_on_existing_rows(self):
        os.unlink(self._db_file)
        with sqlite3.connect(self._db_file) as con:
            con.execute("""
                CREATE TABLE project (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    taxon TEXT NOT NULL UNIQUE,
                    created TEXT NOT NULL
                )
            """)
            con.execute(
                "INSERT INTO project (taxon, created) VALUES ('bugs', '2020-01-01')"
            )
        db.init_db()
        project = db.get_project("bugs")
        self.assertEqual(project["identification_rank"], "species")

    def test_create_defaults_to_species(self):
        db.create_project("bugs")
        project = db.get_project("bugs")
        self.assertEqual(project["identification_rank"], "species")

    def test_create_with_rank(self):
        db.create_project("bugs", "genus")
        project = db.get_project("bugs")
        self.assertEqual(project["identification_rank"], "genus")

    def test_create_rejects_invalid_rank(self):
        with self.assertRaises(ValueError):
            db.create_project("bugs", "order")

    def test_set_identification_rank(self):
        db.create_project("bugs")
        db.set_identification_rank("bugs", "family")
        project = db.get_project("bugs")
        self.assertEqual(project["identification_rank"], "family")

    def test_set_identification_rank_rejects_invalid(self):
        db.create_project("bugs")
        with self.assertRaises(ValueError):
            db.set_identification_rank("bugs", "order")
        project = db.get_project("bugs")
        self.assertEqual(project["identification_rank"], "species")


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


class TestImagesPerTaxon(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._db_file = path
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = Path(path)
        db.init_db()

        self._tmp_images = Path(tempfile.mkdtemp())
        self._orig_images_dir = images.IMAGES_DIR
        images.IMAGES_DIR = self._tmp_images

        _touch_jpg(self._tmp_images / "bugs" / "col1" / "Cicadella_viridis" / "a.jpg")
        _touch_jpg(self._tmp_images / "bugs" / "col1" / "Cicadella_viridis" / "b.jpg")
        _touch_jpg(self._tmp_images / "bugs" / "col1" / "Aphrodes_bicincta" / "c.jpg")
        _touch_jpg(self._tmp_images / "bugs" / "col2" / "Cicadella_viridis" / "d.jpg")
        _touch_jpg(self._tmp_images / "bugs" / "col2" / "Aphrodes_makarovi" / "e.jpg")

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        images.IMAGES_DIR = self._orig_images_dir
        os.unlink(self._db_file)
        shutil.rmtree(self._tmp_images, ignore_errors=True)

    def test_species_keeps_folder_names(self):
        stats = images.project_stats("bugs", "species")
        by_name = {t["taxon"]: t["count"] for t in stats["taxa"]}
        self.assertEqual(by_name["Cicadella_viridis"], 3)
        self.assertEqual(by_name["Aphrodes_bicincta"], 1)
        self.assertEqual(by_name["Aphrodes_makarovi"], 1)

    def test_genus_aggregates_on_first_underscore(self):
        stats = images.project_stats("bugs", "genus")
        rows = [(t["taxon"], t["count"]) for t in stats["taxa"]]
        self.assertEqual(rows, [("Aphrodes", 2), ("Cicadella", 3)])
