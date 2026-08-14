import json
import shutil
import tempfile
import unittest
from pathlib import Path

import trainer.harmonize as harmonize
import trainer.images as images


def _write_taxa_json(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


def _taxon(
    scientific_name: str,
    rank: str = "MX.species",
    synonym_names: list[str] | None = None,
) -> dict:
    item = {
        "scientificName": scientific_name,
        "taxonRank": rank,
        "id": f"MX.{scientific_name}",
    }
    if synonym_names:
        item["synonyms"] = [
            {"scientificName": name, "taxonRank": "MX.species"}
            for name in synonym_names
        ]
    return item


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


class TestNormalizeName(unittest.TestCase):
    def test_underscore_and_case(self):
        self.assertEqual(
            harmonize.normalize_name("Cicadella_Viridis"),
            "cicadella viridis",
        )


class TestLoadTaxaJson(unittest.TestCase):
    def test_filters_to_genus_and_species(self):
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "taxa.json"
        _write_taxa_json(path, [
            _taxon("Cixiidae", "MX.family"),
            _taxon("Cixius", "MX.genus"),
            _taxon("Cixius nervosus", "MX.species"),
            _taxon("Cixius (Cixius)", "MX.subgenus"),
        ])
        taxa = harmonize.load_taxa_json(path)
        names = [t["scientific_name"] for t in taxa]
        self.assertEqual(names, ["Cixius", "Cixius nervosus"])
        shutil.rmtree(tmp, ignore_errors=True)

    def test_loads_synonym_scientific_names(self):
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "taxa.json"
        _write_taxa_json(path, [
            _taxon("Eurysula lurida", synonym_names=["Eurysula laevifrons"]),
        ])
        taxa = harmonize.load_taxa_json(path)
        self.assertEqual(taxa[0]["synonyms"], ["Eurysula laevifrons"])
        shutil.rmtree(tmp, ignore_errors=True)


class TestMatchImageName(unittest.TestCase):
    def setUp(self):
        self.taxa = [
            {
                "scientific_name": "Cicadella viridis",
                "synonyms": ["Tettigella viridis"],
            },
            {
                "scientific_name": "Javesella pellucida",
                "synonyms": ["Javesella marginata"],
            },
            {
                "scientific_name": "Jassargus flori",
                "synonyms": ["Jassargus pseudocellaris", "Jassargus falleni"],
            },
            {
                "scientific_name": "Jassargus allobrogicus",
                "synonyms": ["Jassargus pseudocellaris", "Jassargus falleni"],
            },
            {
                "scientific_name": "Eurysula lurida",
                "synonyms": ["Eurysula laevifrons"],
            },
        ]

    def test_exact_match_underscore_and_case(self):
        row = harmonize.match_image_name("cicadella_VIRIDIS", self.taxa)
        self.assertEqual(row["authoritative_name"], "Cicadella viridis")
        self.assertEqual(row["status"], "")

    def test_synonym_exact_match(self):
        row = harmonize.match_image_name("Eurysula_laevifrons", self.taxa)
        self.assertEqual(row["authoritative_name"], "Eurysula lurida")
        self.assertEqual(row["status"], "")

    def test_synonym_is_not_substring(self):
        row = harmonize.match_image_name("Tettigella", self.taxa)
        self.assertEqual(row["status"], "unknown")

    def test_exact_preferred_over_synonym(self):
        taxa = [
            {
                "scientific_name": "Alpha beta",
                "synonyms": [],
            },
            {
                "scientific_name": "Other name",
                "synonyms": ["Alpha beta"],
            },
        ]
        row = harmonize.match_image_name("Alpha_beta", taxa)
        self.assertEqual(row["authoritative_name"], "Alpha beta")
        self.assertEqual(row["status"], "")

    def test_unknown(self):
        row = harmonize.match_image_name("Unknown_species", self.taxa)
        self.assertEqual(row["authoritative_name"], "")
        self.assertEqual(row["status"], "unknown")

    def test_multiple_synonym_hits(self):
        row = harmonize.match_image_name("Jassargus_pseudocellaris", self.taxa)
        self.assertEqual(row["authoritative_name"], "")
        self.assertEqual(row["status"], "multiple")


class TestGenerateHarmonization(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig_images = images.IMAGES_DIR
        self._orig_data = harmonize.DATA_DIR
        images.IMAGES_DIR = self._tmp / "images"
        harmonize.DATA_DIR = self._tmp / "data"

        _touch_jpg(
            images.IMAGES_DIR / "bugs" / "col1" / "Cicadella_viridis" / "a.jpg"
        )
        _touch_jpg(
            images.IMAGES_DIR / "bugs" / "col1" / "Eurysula_laevifrons" / "b.jpg"
        )
        _touch_jpg(
            images.IMAGES_DIR / "bugs" / "col2" / "Cicadella_viridis" / "c.jpg"
        )
        _touch_jpg(
            images.IMAGES_DIR / "bugs" / "col1" / "Unknown_bug" / "d.jpg"
        )
        _touch_jpg(
            images.IMAGES_DIR
            / "bugs"
            / "col1"
            / "Jassargus_pseudocellaris"
            / "e.jpg"
        )

        _write_taxa_json(
            harmonize.DATA_DIR / "bugs" / "taxa.json",
            [
                _taxon("Cixiidae", "MX.family", ["Unknown_bug"]),
                _taxon(
                    "Cicadella viridis",
                    synonym_names=["Tettigella viridis"],
                ),
                _taxon(
                    "Eurysula lurida",
                    synonym_names=["Eurysula laevifrons"],
                ),
                _taxon(
                    "Jassargus flori",
                    synonym_names=["Jassargus pseudocellaris", "Jassargus falleni"],
                ),
                _taxon(
                    "Jassargus allobrogicus",
                    synonym_names=["Jassargus pseudocellaris", "Jassargus falleni"],
                ),
            ],
        )

    def tearDown(self):
        images.IMAGES_DIR = self._orig_images
        harmonize.DATA_DIR = self._orig_data
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_generate_writes_expected_rows(self):
        out = harmonize.generate_harmonization("bugs")
        self.assertTrue(out.is_file())
        rows = harmonize.read_harmonization(out)
        by_name = {r["image_name"]: r for r in rows}
        self.assertEqual(
            set(by_name),
            {
                "Cicadella_viridis",
                "Eurysula_laevifrons",
                "Unknown_bug",
                "Jassargus_pseudocellaris",
            },
        )
        self.assertEqual(
            by_name["Cicadella_viridis"]["authoritative_name"],
            "Cicadella viridis",
        )
        self.assertEqual(by_name["Cicadella_viridis"]["status"], "")
        self.assertEqual(
            by_name["Eurysula_laevifrons"]["authoritative_name"],
            "Eurysula lurida",
        )
        self.assertEqual(by_name["Unknown_bug"]["status"], "unknown")
        self.assertEqual(by_name["Unknown_bug"]["authoritative_name"], "")
        self.assertEqual(
            by_name["Jassargus_pseudocellaris"]["status"], "multiple"
        )

    def test_overwrite_on_regenerate(self):
        harmonize.generate_harmonization("bugs")
        shutil.rmtree(images.IMAGES_DIR / "bugs" / "col1" / "Unknown_bug")
        harmonize.generate_harmonization("bugs")
        rows = harmonize.read_harmonization(harmonize.harmonization_path("bugs"))
        names = {r["image_name"] for r in rows}
        self.assertNotIn("Unknown_bug", names)
        self.assertIn("Cicadella_viridis", names)

    def test_missing_json_raises(self):
        with self.assertRaises(harmonize.TaxaJsonMissing):
            harmonize.generate_harmonization("nope")

    def test_tsv_header(self):
        out = harmonize.generate_harmonization("bugs")
        header = out.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "image_name\tauthoritative_name\tstatus")


class TestReadHarmonization(unittest.TestCase):
    def test_missing_file_returns_none(self):
        path = Path(tempfile.mkdtemp()) / "missing.tsv"
        self.assertIsNone(harmonize.read_harmonization(path))
        shutil.rmtree(path.parent, ignore_errors=True)
