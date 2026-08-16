"""Unit tests for identification crop / train helpers (no OD / no full train)."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
