import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import trainer.db as db
import trainer.harmonize as harmonize
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

    def test_species_empty_without_harmonization(self):
        stats = images.project_stats("bugs", "species")
        self.assertEqual(stats["taxa"], [])

    def test_genus_empty_without_harmonization(self):
        stats = images.project_stats("bugs", "genus")
        self.assertEqual(stats["taxa"], [])


def _write_taxa_json(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


class TestImagesPerTaxonObservationCounts(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._db_file = path
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = Path(path)
        db.init_db()

        self._tmp = Path(tempfile.mkdtemp())
        self._orig_images_dir = images.IMAGES_DIR
        self._orig_data_dir = harmonize.DATA_DIR
        images.IMAGES_DIR = self._tmp / "images"
        harmonize.DATA_DIR = self._tmp / "data"

        _touch_jpg(images.IMAGES_DIR / "bugs" / "col1" / "Cicadella_viridis" / "a.jpg")
        _touch_jpg(images.IMAGES_DIR / "bugs" / "col1" / "Cicadella_viridis" / "b.jpg")
        _touch_jpg(images.IMAGES_DIR / "bugs" / "col1" / "Tettigella_viridis" / "c.jpg")
        _touch_jpg(images.IMAGES_DIR / "bugs" / "col1" / "Cicadella" / "d.jpg")
        _touch_jpg(images.IMAGES_DIR / "bugs" / "col1" / "Cicadella" / "e.jpg")
        _touch_jpg(images.IMAGES_DIR / "bugs" / "col1" / "Aphrodes_makarovi" / "f.jpg")

        _write_taxa_json(
            harmonize.DATA_DIR / "bugs" / "taxa.json",
            [
                {
                    "scientificName": "Cicadella",
                    "taxonRank": "MX.genus",
                    "observationCountFinland": 2000,
                },
                {
                    "scientificName": "Cicadella viridis",
                    "taxonRank": "MX.species",
                    "observationCountFinland": 1472,
                    "synonyms": [{"scientificName": "Tettigella viridis"}],
                },
                {
                    "scientificName": "Aphrodes",
                    "taxonRank": "MX.genus",
                    "observationCountFinland": 500,
                },
                {
                    "scientificName": "Aphrodes makarovi",
                    "taxonRank": "MX.species",
                    "observationCountFinland": 80,
                },
            ],
        )

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        images.IMAGES_DIR = self._orig_images_dir
        harmonize.DATA_DIR = self._orig_data_dir
        os.unlink(self._db_file)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_species_uses_harmonized_names(self):
        harmonize.generate_harmonization("bugs")
        stats = images.project_stats("bugs", "species")
        by_name = {t["taxon"]: t for t in stats["taxa"]}
        self.assertEqual(set(by_name), {
            "Cicadella viridis",
            "Cicadella",
            "Aphrodes makarovi",
        })
        self.assertEqual(by_name["Cicadella viridis"]["count"], 3)
        self.assertEqual(by_name["Cicadella"]["count"], 2)
        self.assertEqual(by_name["Aphrodes makarovi"]["count"], 1)
        self.assertEqual(by_name["Cicadella viridis"]["observation_count_finland"], "1472")
        self.assertEqual(by_name["Cicadella"]["observation_count_finland"], "2000")
        self.assertEqual(by_name["Aphrodes makarovi"]["observation_count_finland"], "80")
        self.assertAlmostEqual(by_name["Cicadella viridis"]["images_per_observation"], 3 / 1472)
        self.assertAlmostEqual(by_name["Aphrodes makarovi"]["images_per_observation"], 1 / 80)

    def test_genus_sums_harmonized_genus_and_species(self):
        harmonize.generate_harmonization("bugs")
        stats = images.project_stats("bugs", "genus")
        by_name = {t["taxon"]: t for t in stats["taxa"]}
        self.assertEqual(by_name["Cicadella"]["count"], 5)
        self.assertEqual(by_name["Aphrodes"]["count"], 1)
        self.assertNotIn("Tettigella", by_name)
        self.assertEqual(by_name["Cicadella"]["observation_count_finland"], "2000")
        self.assertEqual(by_name["Aphrodes"]["observation_count_finland"], "500")
        self.assertAlmostEqual(by_name["Cicadella"]["images_per_observation"], 5 / 2000)
        self.assertAlmostEqual(by_name["Aphrodes"]["images_per_observation"], 1 / 500)
