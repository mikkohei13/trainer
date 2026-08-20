"""Image transforms and CropDataset."""

from __future__ import annotations

import logging
import random

import numpy as np
from PIL import Image, ImageEnhance

from . import config
from .data import log


def letterbox(img: Image.Image, size: int, fill: int = 0) -> Image.Image:
    """Resize preserving aspect ratio and pad to a square of side `size`."""
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), (fill, fill, fill))
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def train_augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle90 = random.choice([0, 90, 180, 270])
    if angle90:
        img = img.rotate(angle90, expand=True)
    small = random.uniform(-30.0, 30.0)
    if abs(small) > 0.5:
        img = img.rotate(small, expand=True, fillcolor=(0, 0, 0))
    scale = random.uniform(0.85, 1.15)
    if abs(scale - 1.0) > 0.01:
        w, h = img.size
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.8:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    return img


def pil_to_normalized_tensor(img: Image.Image, img_size: int):
    import torch

    boxed = letterbox(img, img_size, fill=0)
    arr = np.asarray(boxed, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - config.IMAGENET_MEAN) / config.IMAGENET_STD
    return torch.from_numpy(arr)


def pil_to_uint8_tensor(img: Image.Image, img_size: int):
    import torch

    boxed = letterbox(img, img_size, fill=0)
    # Keep as uint8 to shrink RAM cache; normalize later on-device.
    arr = np.array(boxed, dtype=np.uint8).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr))


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return img
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.BILINEAR)


def maybe_normalize_on_device(x):
    """If x is a cached uint8 image tensor (BCHW), normalize it for ImageNet."""
    import torch

    if x.dtype != torch.uint8:
        return x
    x = x.to(dtype=torch.float32).div(255.0)
    mean = torch.as_tensor(config.IMAGENET_MEAN, device=x.device, dtype=torch.float32)
    std = torch.as_tensor(config.IMAGENET_STD, device=x.device, dtype=torch.float32)
    return (x - mean) / std


class CropDataset:
    def __init__(
        self,
        records: list[dict],
        class_to_idx: dict[str, int],
        train: bool,
        img_size: int | None = None,
        cache_images: bool | None = None,
        cache_train_resize_factor: float | None = None,
        cache_eval_as_uint8: bool | None = None,
        logger: logging.Logger | None = None,
    ):
        if img_size is None:
            img_size = config.IMG_SIZE
        if cache_images is None:
            cache_images = config.CACHE_IMAGES
        if cache_train_resize_factor is None:
            cache_train_resize_factor = config.CACHE_TRAIN_RESIZE_FACTOR
        if cache_eval_as_uint8 is None:
            cache_eval_as_uint8 = config.CACHE_EVAL_AS_UINT8
        self.records = records
        self.class_to_idx = class_to_idx
        self.train = train
        self.img_size = img_size
        self.cache_train_resize_factor = cache_train_resize_factor
        self.cache_eval_as_uint8 = cache_eval_as_uint8
        self.cache = None
        if cache_images:
            self.cache = self._build_cache(logger)

    def _build_cache(self, logger: logging.Logger | None):
        cached = []
        n = len(self.records)
        for i, rec in enumerate(self.records, start=1):
            if logger and (i == 1 or i % 2000 == 0 or i == n):
                split = "train" if self.train else "eval"
                log(logger, f"caching {split} images {i}/{n}")
            path = config.PROCESSED_DIR / rec["crop_path"]
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                if self.train:
                    # Pre-resize so augmentation runs on a smaller bitmap.
                    max_side = max(1, int(round(self.img_size * self.cache_train_resize_factor)))
                    rgb = resize_max_side(rgb, max_side)
                    cached.append(rgb.copy())
                else:
                    if self.cache_eval_as_uint8:
                        cached.append(pil_to_uint8_tensor(rgb, self.img_size))
                    else:
                        cached.append(pil_to_normalized_tensor(rgb, self.img_size))
        if logger:
            split = "train" if self.train else "eval"
            if self.train:
                pixels = sum(im.width * im.height * 3 for im in cached)
                log(logger, f"cached {n} {split} RGB crops (~{pixels / 1e9:.2f} GB uncompressed)")
            else:
                kind = "uint8" if self.cache_eval_as_uint8 else "float32 normalized"
                log(logger, f"cached {n} {split} tensors ({kind})")
        return cached

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, index: int) -> Image.Image:
        if self.cache is not None:
            return self.cache[index]
        rec = self.records[index]
        with Image.open(config.PROCESSED_DIR / rec["crop_path"]) as img:
            return img.convert("RGB")

    def __getitem__(self, index: int):
        rec = self.records[index]
        if self.train:
            rgb = self._load_rgb(index)
            if self.cache is not None:
                rgb = rgb.copy()
            rgb = train_augment(rgb)
            arr = pil_to_normalized_tensor(rgb, self.img_size)
        elif self.cache is not None:
            arr = self.cache[index]
        else:
            arr = pil_to_normalized_tensor(self._load_rgb(index), self.img_size)
        label = self.class_to_idx[rec["genus"]]
        return arr, label
