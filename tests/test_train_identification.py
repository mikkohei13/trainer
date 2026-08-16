"""Unit tests for identification training helpers (no OD / no full train)."""

import importlib.util
import unittest
from pathlib import Path

from PIL import Image


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "train_identification.py"
    spec = importlib.util.spec_from_file_location("train_identification", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tid = _load_script()


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
        # pad Alpha to 10
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


class TestCropFilename(unittest.TestCase):
    def test_encodes_path(self):
        name = tid.crop_filename_for_source(
            "auchenorrhyncha/britishbugs/Acericerus_ribauti/Acericerus_ribauti_0.jpg"
        )
        self.assertNotIn("/", name)
        self.assertTrue(name.endswith(".jpg"))
        self.assertIn("britishbugs", name)


if __name__ == "__main__":
    unittest.main()
