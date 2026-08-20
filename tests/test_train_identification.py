"""Unit tests for identification crop / train helpers (no OD / no full train)."""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from identification import config as tid_config
from identification import data as tid_data
from identification import dataset as tid_dataset
from identification import train as tid_train


def _load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


crop = _load_script("crop_identification_images.py")


class TestGenusFromAuthoritative(unittest.TestCase):
    def test_first_word(self):
        self.assertEqual(tid_data.genus_from_authoritative("Acericerus ribauti"), "Acericerus")

    def test_single_word(self):
        self.assertEqual(tid_data.genus_from_authoritative("Cixius"), "Cixius")

    def test_empty(self):
        self.assertIsNone(tid_data.genus_from_authoritative(""))
        self.assertIsNone(tid_data.genus_from_authoritative("   "))


class TestFilterByMinCount(unittest.TestCase):
    def test_drops_small_classes(self):
        labeled = [
            ("a/1.jpg", "Alpha"),
            ("a/2.jpg", "Alpha"),
            ("b/1.jpg", "Beta"),
        ]
        labeled += [(f"a/{i}.jpg", "Alpha") for i in range(3, 11)]
        out = tid_data.filter_by_min_count(labeled, min_count=10)
        genera = {g for _, g in out}
        self.assertEqual(genera, {"Alpha"})
        self.assertEqual(len(out), 10)


class TestFilterByCollections(unittest.TestCase):
    def test_collection_from_crop_path(self):
        self.assertEqual(
            tid_data.collection_from_crop_path("bugs/truehopperswp/Taxon/a.jpg"),
            "truehopperswp",
        )

    def test_drops_excluded_collections(self):
        labeled = [
            ("bugs/inaturalist/A/1.jpg", "Alpha"),
            ("bugs/truehopperswp/A/2.jpg", "Alpha"),
            ("bugs/vihko/B/3.jpg", "Beta"),
        ]
        out = tid_data.filter_labeled_by_collections(
            labeled, excluded=("truehopperswp",)
        )
        self.assertEqual(
            out,
            [
                ("bugs/inaturalist/A/1.jpg", "Alpha"),
                ("bugs/vihko/B/3.jpg", "Beta"),
            ],
        )

    def test_filters_all_splits(self):
        splits = {
            "train": [
                {"crop_path": "bugs/inaturalist/A/1.jpg", "genus": "Alpha"},
                {"crop_path": "bugs/truehopperswp/A/2.jpg", "genus": "Alpha"},
            ],
            "val": [{"crop_path": "bugs/truehopperswp/B/3.jpg", "genus": "Beta"}],
            "test": [{"crop_path": "bugs/vihko/C/4.jpg", "genus": "Gamma"}],
        }
        out = tid_data.filter_splits_by_collections(
            splits, excluded=("truehopperswp",)
        )
        self.assertEqual(
            out["train"],
            [{"crop_path": "bugs/inaturalist/A/1.jpg", "genus": "Alpha"}],
        )
        self.assertEqual(out["val"], [])
        self.assertEqual(
            out["test"],
            [{"crop_path": "bugs/vihko/C/4.jpg", "genus": "Gamma"}],
        )


class TestFilterByQuality(unittest.TestCase):
    def test_drops_scores_at_or_below_threshold(self):
        ratings = {
            "a/high.jpg": 0.9,
            "a/edge.jpg": 0.25,
            "a/low.jpg": 0.24,
            "b/just_above.jpg": 0.2500001,
        }
        labeled = [(rel, "Alpha") for rel in ratings]
        out = tid_data.filter_labeled_by_quality(labeled, ratings)
        self.assertEqual([rel for rel, _ in out], ["a/high.jpg", "b/just_above.jpg"])

    def test_drops_when_rating_missing(self):
        labeled = [("a/ok.jpg", "Alpha"), ("a/missing.jpg", "Alpha")]
        out = tid_data.filter_labeled_by_quality(labeled, {"a/ok.jpg": 0.9})
        self.assertEqual(out, [("a/ok.jpg", "Alpha")])

    def test_filters_all_splits(self):
        ratings = {
            "a/train_ok.jpg": 0.8,
            "a/train_low.jpg": 0.1,
            "a/val_ok.jpg": 0.6,
            "a/test_low.jpg": 0.0,
        }
        splits = {
            "train": [
                {"crop_path": "a/train_ok.jpg", "genus": "Alpha"},
                {"crop_path": "a/train_low.jpg", "genus": "Alpha"},
            ],
            "val": [{"crop_path": "a/val_ok.jpg", "genus": "Alpha"}],
            "test": [{"crop_path": "a/test_low.jpg", "genus": "Alpha"}],
        }
        out = tid_data.filter_splits_by_quality(splits, ratings)
        self.assertEqual(out["train"], [{"crop_path": "a/train_ok.jpg", "genus": "Alpha"}])
        self.assertEqual(out["val"], [{"crop_path": "a/val_ok.jpg", "genus": "Alpha"}])
        self.assertEqual(out["test"], [])

    def test_quality_json_path(self):
        self.assertEqual(
            tid_data.quality_ratings_path("bugs"),
            tid_config.PROCESSED_DIR / "bugs" / "quality.json",
        )


class TestLetterbox(unittest.TestCase):
    def test_square_output(self):
        img = Image.new("RGB", (100, 50), (255, 0, 0))
        out = tid_dataset.letterbox(img, 64, fill=0)
        self.assertEqual(out.size, (64, 64))

    def test_tall_image(self):
        img = Image.new("RGB", (40, 80), (0, 255, 0))
        out = tid_dataset.letterbox(img, 32, fill=0)
        self.assertEqual(out.size, (32, 32))


class TestProcessedPath(unittest.TestCase):
    def test_mirrors_layout(self):
        rel = "auchenorrhyncha/britishbugs/Acericerus_ribauti/Acericerus_ribauti_0.jpg"
        out = crop.processed_path_for(rel)
        self.assertEqual(out, crop.PROCESSED_DIR / rel)


class TestCropBoxPadding(unittest.TestCase):
    def test_pads_and_clamps(self):
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        box = {"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0}
        cropped = crop.crop_box_with_padding(img, box, 0.10)
        # 10% of 20 = 2 → 8..32
        self.assertEqual(cropped.size, (24, 24))


class TestCropProjectImagesSkipsFailures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.images = self.tmp / "images"
        self.processed = self.tmp / "processed"
        taxon_dir = self.images / "bugs" / "col" / "Taxon"
        taxon_dir.mkdir(parents=True)
        Image.new("RGB", (20, 20), (1, 2, 3)).save(taxon_dir / "ok.jpg")
        Image.new("RGB", (20, 20), (4, 5, 6)).save(taxon_dir / "bad.jpg")

        self._orig_images = crop.IMAGES_DIR
        self._orig_processed = crop.PROCESSED_DIR
        crop.IMAGES_DIR = self.images
        crop.PROCESSED_DIR = self.processed

    def tearDown(self):
        crop.IMAGES_DIR = self._orig_images
        crop.PROCESSED_DIR = self._orig_processed
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch.object(crop, "predict_quality_score")
    @patch.object(crop.db, "get_active_quality_model_path_for_taxon")
    @patch.object(crop, "predict_top_box")
    @patch.object(crop.db, "get_active_model_path_for_taxon")
    @patch.object(crop, "list_project_image_paths")
    def test_continues_when_predict_fails(
        self, mock_paths, mock_model, mock_predict, mock_quality_model, mock_quality
    ):
        mock_model.return_value = Path("/tmp/fake.pt")
        mock_quality_model.return_value = Path("/tmp/quality.pt")
        mock_quality.return_value = 0.8
        mock_paths.return_value = [
            "bugs/col/Taxon/bad.jpg",
            "bugs/col/Taxon/ok.jpg",
        ]

        def predict(_model_path, image_abs_path):
            if image_abs_path.name == "bad.jpg":
                raise ValueError("need at least one array to stack")
            return [{"x": 2.0, "y": 2.0, "w": 10.0, "h": 10.0}]

        mock_predict.side_effect = predict
        crop.crop_project_images("bugs")

        self.assertTrue((self.processed / "bugs/col/Taxon/ok.jpg").is_file())
        self.assertFalse((self.processed / "bugs/col/Taxon/bad.jpg").is_file())
        ratings = crop.load_quality_ratings("bugs")
        self.assertEqual(ratings, {"bugs/col/Taxon/ok.jpg": 0.8})


class TestCropSkipsExistingCropAndRating(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.images = self.tmp / "images"
        self.processed = self.tmp / "processed"
        taxon_dir = self.images / "bugs" / "col" / "Taxon"
        taxon_dir.mkdir(parents=True)
        Image.new("RGB", (20, 20), (1, 2, 3)).save(taxon_dir / "a.jpg")
        Image.new("RGB", (20, 20), (4, 5, 6)).save(taxon_dir / "b.jpg")

        self._orig_images = crop.IMAGES_DIR
        self._orig_processed = crop.PROCESSED_DIR
        crop.IMAGES_DIR = self.images
        crop.PROCESSED_DIR = self.processed

    def tearDown(self):
        crop.IMAGES_DIR = self._orig_images
        crop.PROCESSED_DIR = self._orig_processed
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_crop(self, name: str) -> None:
        out = self.processed / "bugs" / "col" / "Taxon"
        out.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10), (9, 9, 9)).save(out / name)

    @patch.object(crop, "predict_quality_score")
    @patch.object(crop.db, "get_active_quality_model_path_for_taxon")
    @patch.object(crop, "predict_top_box")
    @patch.object(crop.db, "get_active_model_path_for_taxon")
    @patch.object(crop, "list_project_image_paths")
    def test_rates_existing_crop_without_recropping(
        self, mock_paths, mock_model, mock_predict, mock_quality_model, mock_quality
    ):
        mock_model.return_value = Path("/tmp/fake.pt")
        mock_quality_model.return_value = Path("/tmp/quality.pt")
        mock_quality.return_value = 0.42
        mock_paths.return_value = ["bugs/col/Taxon/a.jpg"]
        self._write_crop("a.jpg")

        crop.crop_project_images("bugs")

        mock_predict.assert_not_called()
        mock_quality.assert_called_once()
        self.assertEqual(crop.load_quality_ratings("bugs")["bugs/col/Taxon/a.jpg"], 0.42)

    @patch.object(crop, "predict_quality_score")
    @patch.object(crop.db, "get_active_quality_model_path_for_taxon")
    @patch.object(crop, "predict_top_box")
    @patch.object(crop.db, "get_active_model_path_for_taxon")
    @patch.object(crop, "list_project_image_paths")
    def test_skips_existing_rating(
        self, mock_paths, mock_model, mock_predict, mock_quality_model, mock_quality
    ):
        mock_model.return_value = Path("/tmp/fake.pt")
        mock_quality_model.return_value = Path("/tmp/quality.pt")
        mock_paths.return_value = ["bugs/col/Taxon/a.jpg"]
        self._write_crop("a.jpg")
        crop.save_quality_ratings("bugs", {"bugs/col/Taxon/a.jpg": 0.11})

        crop.crop_project_images("bugs")

        mock_predict.assert_not_called()
        mock_quality.assert_not_called()
        self.assertEqual(crop.load_quality_ratings("bugs")["bugs/col/Taxon/a.jpg"], 0.11)


class TestListProcessed(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig = tid_config.PROCESSED_DIR
        tid_config.PROCESSED_DIR = self.tmp

    def tearDown(self):
        tid_config.PROCESSED_DIR = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lists_images(self):
        root = self.tmp / "bugs" / "col" / "Genus_sp"
        root.mkdir(parents=True)
        (root / "a.jpg").write_bytes(b"x")
        (root / "notes.txt").write_text("nope")
        paths = tid_data.list_processed_image_paths("bugs")
        self.assertEqual(paths, ["bugs/col/Genus_sp/a.jpg"])


class TestFrozenSplits(unittest.TestCase):
    def test_round_trip(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            splits = {
                "train": [{"crop_path": "a/1.jpg", "genus": "Alpha"}],
                "val": [{"crop_path": "a/2.jpg", "genus": "Alpha"}],
                "test": [{"crop_path": "b/1.jpg", "genus": "Beta"}],
            }
            path = tmp / "splits.json"
            tid_data.write_splits(path, splits)
            loaded = tid_data.load_splits(path)
            self.assertEqual(loaded, splits)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_frozen_path_is_outside_run_dir(self):
        path = tid_data.frozen_splits_path("bugs")
        self.assertEqual(
            path, tid_config.MODELS_DIR / "bugs" / "identification" / "splits.json"
        )


class TestObservationAwareSplit(unittest.TestCase):
    def test_groups_same_observation(self):
        records = [
            {"crop_path": "bugs/inaturalist/A/1.jpg", "genus": "Alpha"},
            {"crop_path": "bugs/inaturalist/A/2.jpg", "genus": "Alpha"},
            {"crop_path": "bugs/britishbugs/A/x.jpg", "genus": "Alpha"},
        ]
        photo_to_obs = {"A/1.jpg": 10, "A/2.jpg": 10}
        units = tid_data.group_records_by_observation(records, photo_to_obs, "bugs")
        self.assertEqual(len(units), 2)
        multi = next(u for u in units if len(u) == 2)
        paths = {r["crop_path"] for r in multi}
        self.assertEqual(
            paths,
            {"bugs/inaturalist/A/1.jpg", "bugs/inaturalist/A/2.jpg"},
        )

    def test_split_keeps_observation_together(self):
        # Enough distinct units per genus for stratified 3-way split.
        records = []
        photo_to_obs = {}
        for i in range(12):
            records.append(
                {
                    "crop_path": f"bugs/inaturalist/A/{i}a.jpg",
                    "genus": "Alpha",
                }
            )
            records.append(
                {
                    "crop_path": f"bugs/inaturalist/A/{i}b.jpg",
                    "genus": "Alpha",
                }
            )
            photo_to_obs[f"A/{i}a.jpg"] = 100 + i
            photo_to_obs[f"A/{i}b.jpg"] = 100 + i
        for i in range(12):
            records.append(
                {
                    "crop_path": f"bugs/inaturalist/B/{i}.jpg",
                    "genus": "Beta",
                }
            )
            photo_to_obs[f"B/{i}.jpg"] = 200 + i

        splits = tid_data.stratified_split(
            records,
            seed=0,
            photo_to_observation=photo_to_obs,
            project="bugs",
        )
        path_to_split = {}
        for name, recs in splits.items():
            for r in recs:
                path_to_split[r["crop_path"]] = name

        for i in range(12):
            a = path_to_split[f"bugs/inaturalist/A/{i}a.jpg"]
            b = path_to_split[f"bugs/inaturalist/A/{i}b.jpg"]
            self.assertEqual(a, b, f"observation {100 + i} split across {a} and {b}")

        all_paths = {r["crop_path"] for recs in splits.values() for r in recs}
        self.assertEqual(all_paths, {r["crop_path"] for r in records})

    def test_load_photo_to_observation_missing_file(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            with patch.object(tid_config, "PROCESSED_DIR", tmp):
                self.assertEqual(tid_data.load_photo_to_observation("bugs"), {})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_photo_to_observation(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            project = tmp / "bugs"
            project.mkdir()
            payload = {
                "photo_to_observation": {"Taxon/1.jpg": 0, "Taxon/2.jpg": 0},
            }
            (project / "inaturalist_observations.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with patch.object(tid_config, "PROCESSED_DIR", tmp):
                mapping = tid_data.load_photo_to_observation("bugs")
            self.assertEqual(mapping, {"Taxon/1.jpg": 0, "Taxon/2.jpg": 0})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestInverseSqrtWeights(unittest.TestCase):
    def test_rarer_class_gets_higher_weight(self):
        records = [{"genus": "A"}] * 100 + [{"genus": "B"}] * 4
        weights = tid_data.inverse_sqrt_sample_weights(records)
        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(weights[0], 1.0 / (100 ** 0.5))
        self.assertAlmostEqual(weights[-1], 1.0 / (4 ** 0.5))


class TestWorstClassRecalls(unittest.TestCase):
    def test_orders_by_recall(self):
        idx_to_class = {0: "Alpha", 1: "Beta"}
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 0]
        rows = tid_train.worst_class_recalls(y_true, y_pred, idx_to_class, k=2, min_support=1)
        self.assertEqual(rows[0]["genus"], "Beta")
        self.assertAlmostEqual(rows[0]["recall"], 0.5)
        self.assertEqual(rows[1]["genus"], "Alpha")
        self.assertAlmostEqual(rows[1]["recall"], 1.0)

    def test_skips_classes_below_min_support(self):
        idx_to_class = {0: "Alpha", 1: "Beta"}
        y_true = [0] * 8 + [1, 1]
        y_pred = [0] * 8 + [1, 0]
        rows = tid_train.worst_class_recalls(y_true, y_pred, idx_to_class, k=2, min_support=8)
        self.assertEqual([r["genus"] for r in rows], ["Alpha"])

    def test_ties_prefer_larger_support(self):
        idx_to_class = {0: "Alpha", 1: "Beta"}
        y_true = [0] * 8 + [1] * 12
        y_pred = [1] * 8 + [0] * 12
        rows = tid_train.worst_class_recalls(y_true, y_pred, idx_to_class, k=2, min_support=8)
        self.assertEqual([r["genus"] for r in rows], ["Beta", "Alpha"])


class TestBestClassRecalls(unittest.TestCase):
    def test_orders_by_recall_desc(self):
        idx_to_class = {0: "Alpha", 1: "Beta"}
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 0]
        rows = tid_train.best_class_recalls(y_true, y_pred, idx_to_class, k=2, min_support=1)
        self.assertEqual(rows[0]["genus"], "Alpha")
        self.assertAlmostEqual(rows[0]["recall"], 1.0)
        self.assertEqual(rows[1]["genus"], "Beta")
        self.assertAlmostEqual(rows[1]["recall"], 0.5)

    def test_ties_prefer_larger_support(self):
        idx_to_class = {0: "Alpha", 1: "Beta"}
        y_true = [0] * 8 + [1] * 12
        y_pred = [0] * 8 + [1] * 12
        rows = tid_train.best_class_recalls(y_true, y_pred, idx_to_class, k=2, min_support=8)
        self.assertEqual([r["genus"] for r in rows], ["Beta", "Alpha"])


class TestFilterEvalByQuality(unittest.TestCase):
    def test_keeps_scores_at_or_above_threshold(self):
        records = [
            {"crop_path": "a/high.jpg"},
            {"crop_path": "a/edge.jpg"},
            {"crop_path": "a/low.jpg"},
            {"crop_path": "a/missing.jpg"},
        ]
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 0, 3]
        ratings = {"a/high.jpg": 0.9, "a/edge.jpg": 0.7, "a/low.jpg": 0.69}
        yt, yp = tid_train.filter_eval_by_quality(records, y_true, y_pred, ratings, 0.7)
        self.assertEqual(yt, [0, 1])
        self.assertEqual(yp, [0, 1])


class TestClassificationMetrics(unittest.TestCase):
    def test_skip_empty_averages_only_classes_with_support(self):
        # class 0: 2/2 correct; class 1 absent from y_true
        y_true = [0, 0]
        y_pred = [0, 0]
        all_classes = tid_train.classification_metrics(y_true, y_pred, num_classes=2)
        present = tid_train.classification_metrics(
            y_true, y_pred, num_classes=2, skip_empty=True
        )
        self.assertAlmostEqual(all_classes["macro_f1"], 0.5)
        self.assertAlmostEqual(present["macro_f1"], 1.0)
        self.assertEqual(present["n"], 2)
        self.assertEqual(present["num_classes"], 1)


if __name__ == "__main__":
    unittest.main()
