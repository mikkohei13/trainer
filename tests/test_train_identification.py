"""Unit tests for identification crop / train helpers (no OD / no full train)."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def _load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tid = _load_script("train_identification.py")
crop = _load_script("crop_identification_images.py")


class TestGenusFromAuthoritative(unittest.TestCase):
    def test_first_word(self):
        self.assertEqual(tid.genus_from_authoritative("Acericerus ribauti"), "Acericerus")

    def test_single_word(self):
        self.assertEqual(tid.genus_from_authoritative("Cixius"), "Cixius")

    def test_empty(self):
        self.assertIsNone(tid.genus_from_authoritative(""))
        self.assertIsNone(tid.genus_from_authoritative("   "))


class TestFilterByMinCount(unittest.TestCase):
    def test_drops_small_classes(self):
        labeled = [
            ("a/1.jpg", "Alpha"),
            ("a/2.jpg", "Alpha"),
            ("b/1.jpg", "Beta"),
        ]
        labeled += [(f"a/{i}.jpg", "Alpha") for i in range(3, 11)]
        out = tid.filter_by_min_count(labeled, min_count=10)
        genera = {g for _, g in out}
        self.assertEqual(genera, {"Alpha"})
        self.assertEqual(len(out), 10)


class TestLetterbox(unittest.TestCase):
    def test_square_output(self):
        img = Image.new("RGB", (100, 50), (255, 0, 0))
        out = tid.letterbox(img, 64, fill=0)
        self.assertEqual(out.size, (64, 64))

    def test_tall_image(self):
        img = Image.new("RGB", (40, 80), (0, 255, 0))
        out = tid.letterbox(img, 32, fill=0)
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

    @patch.object(crop, "predict_top_box")
    @patch.object(crop.db, "get_active_model_path_for_taxon")
    @patch.object(crop, "list_project_image_paths")
    def test_continues_when_predict_fails(self, mock_paths, mock_model, mock_predict):
        mock_model.return_value = Path("/tmp/fake.pt")
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


class TestListProcessed(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig = tid.PROCESSED_DIR
        tid.PROCESSED_DIR = self.tmp

    def tearDown(self):
        tid.PROCESSED_DIR = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lists_images(self):
        root = self.tmp / "bugs" / "col" / "Genus_sp"
        root.mkdir(parents=True)
        (root / "a.jpg").write_bytes(b"x")
        (root / "notes.txt").write_text("nope")
        paths = tid.list_processed_image_paths("bugs")
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
            tid.write_splits(path, splits)
            loaded = tid.load_splits(path)
            self.assertEqual(loaded, splits)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_round_trip_preserves_quality(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            splits = {
                "train": [{"crop_path": "a/1.jpg", "genus": "Alpha", "quality": 0.81}],
                "val": [{"crop_path": "a/2.jpg", "genus": "Alpha", "quality": 0.4}],
                "test": [{"crop_path": "b/1.jpg", "genus": "Beta", "quality": 0.95}],
            }
            path = tmp / "splits.json"
            tid.write_splits(path, splits)
            loaded = tid.load_splits(path)
            self.assertEqual(loaded, splits)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_frozen_path_is_outside_run_dir(self):
        path = tid.frozen_splits_path("bugs")
        self.assertEqual(path, tid.MODELS_DIR / "bugs" / "identification" / "splits.json")


class TestInverseSqrtWeights(unittest.TestCase):
    def test_rarer_class_gets_higher_weight(self):
        records = [{"genus": "A"}] * 100 + [{"genus": "B"}] * 4
        weights = tid.inverse_sqrt_sample_weights(records)
        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(weights[0], 1.0 / (100 ** 0.5))
        self.assertAlmostEqual(weights[-1], 1.0 / (4 ** 0.5))


class TestWorstClassRecalls(unittest.TestCase):
    def test_orders_by_recall(self):
        idx_to_class = {0: "Alpha", 1: "Beta"}
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 0]
        rows = tid.worst_class_recalls(y_true, y_pred, idx_to_class, k=2)
        self.assertEqual(rows[0]["genus"], "Beta")
        self.assertAlmostEqual(rows[0]["recall"], 0.5)
        self.assertEqual(rows[1]["genus"], "Alpha")
        self.assertAlmostEqual(rows[1]["recall"], 1.0)

    def test_skips_zero_support(self):
        idx_to_class = {0: "Alpha", 1: "Beta"}
        y_true = [0, 0]
        y_pred = [0, 1]
        rows = tid.worst_class_recalls(y_true, y_pred, idx_to_class, k=2)
        self.assertEqual([r["genus"] for r in rows], ["Alpha"])


class TestKeepFracForCount(unittest.TestCase):
    def test_rarest_mid_common(self):
        self.assertAlmostEqual(tid.keep_frac_for_count(10, 10, 1000), 1.0)
        self.assertAlmostEqual(tid.keep_frac_for_count(1000, 10, 1000), 0.30)
        self.assertAlmostEqual(tid.keep_frac_for_count(100, 10, 1000), 0.65)


class TestSelectTrainByQuality(unittest.TestCase):
    def test_drops_floor_and_keeps_highest_quality(self):
        rare = [
            {"crop_path": f"rare/{i}.jpg", "genus": "Rare", "quality": 0.40 + i * 0.01}
            for i in range(10)
        ]
        rare.append({"crop_path": "rare/bad.jpg", "genus": "Rare", "quality": 0.1})
        common = [
            {"crop_path": f"common/{i:03d}.jpg", "genus": "Common", "quality": 0.35 + i * 0.005}
            for i in range(100)
        ]
        common.append({"crop_path": "common/bad.jpg", "genus": "Common", "quality": 0.05})

        kept, stats = tid.select_train_by_quality(rare + common)
        self.assertTrue(all(r["quality"] >= 0.3 for r in kept))
        self.assertFalse(any(r["crop_path"].endswith("bad.jpg") for r in kept))

        by_genus = {row["genus"]: row for row in stats}
        self.assertEqual(by_genus["Rare"]["n_raw"], 11)
        self.assertEqual(by_genus["Rare"]["n_after_floor"], 10)
        self.assertEqual(by_genus["Rare"]["n_kept"], 10)
        self.assertAlmostEqual(by_genus["Rare"]["keep_frac"], 1.0)

        self.assertEqual(by_genus["Common"]["n_raw"], 101)
        self.assertEqual(by_genus["Common"]["n_after_floor"], 100)
        self.assertAlmostEqual(by_genus["Common"]["keep_frac"], 0.30)
        self.assertEqual(by_genus["Common"]["n_kept"], 30)

        common_kept = [r for r in kept if r["genus"] == "Common"]
        self.assertEqual(len(common_kept), 30)
        worst_kept = min(r["quality"] for r in common_kept)
        dropped_ok = [
            r for r in common if r["quality"] >= 0.3 and r not in common_kept
        ]
        self.assertTrue(all(r["quality"] <= worst_kept for r in dropped_ok))


class TestDropGeneraMissingFromTrain(unittest.TestCase):
    def test_drops_class_with_no_train_images(self):
        splits = {
            "train": [{"crop_path": "a.jpg", "genus": "Alpha", "quality": 0.8}],
            "val": [
                {"crop_path": "b.jpg", "genus": "Alpha", "quality": 0.8},
                {"crop_path": "c.jpg", "genus": "Beta", "quality": 0.9},
            ],
            "test": [{"crop_path": "d.jpg", "genus": "Beta", "quality": 0.9}],
        }
        out, dropped = tid.drop_genera_missing_from_train(splits)
        self.assertEqual(dropped, ["Beta"])
        self.assertEqual([r["genus"] for r in out["val"]], ["Alpha"])
        self.assertEqual(out["test"], [])


class TestDualMetricsHqSupport(unittest.TestCase):
    def test_hq_macro_f1_ignores_classes_with_zero_hq_support(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 1]
        qualities = [0.9, 0.8, 0.5, 0.4]
        metrics = tid.dual_metrics(y_true, y_pred, qualities, num_classes=2)
        self.assertEqual(metrics["hq"]["n"], 2)
        self.assertEqual(metrics["hq"]["n_classes_scored"], 1)
        self.assertAlmostEqual(metrics["hq"]["macro_f1"], 1.0)
        self.assertEqual(metrics["all"]["n"], 4)
        self.assertEqual(metrics["all"]["n_classes_scored"], 2)
        # class 0: tp=2 fp=1 fn=0 → prec=2/3 rec=1 F1=0.8
        # class 1: tp=1 fp=0 fn=1 → prec=1 rec=0.5 F1=2/3
        self.assertAlmostEqual(metrics["all"]["macro_f1"], (0.8 + 2.0 / 3.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
