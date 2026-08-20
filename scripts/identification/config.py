"""Hardcoded training parameters — edit here, then re-run the train script."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT = "auchenorrhyncha"
MIN_IMAGES_PER_CLASS = 10
# Skip these source folders under images_processed/<project>/ (train/val/test).
EXCLUDE_COLLECTIONS = ("truehopperswp",)
# Drop processed crops whose predicted quality is at or below this.
MIN_QUALITY = 0.25
# Final-model report: metrics on photos at or above this quality.
HIGH_QUALITY = 0.7
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
TEST_FRAC = 0.10
SEED = 13

# Evaluation image size (used for val/test).
IMG_SIZE = 384
# Training image size (used for train, to speed up CPU augmentation + training).
TRAIN_IMG_SIZE = 384
BATCH_SIZE = 128  # drop to 64 if MPS runs out of memory at 384
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 60
UNFREEZE_LEAF_MODULES = 60  # later MBConv blocks; keep head unfrozen too
LR_HEAD = 1e-3
LR_FINETUNE_HEAD = 1e-4
LR_FINETUNE_BACKBONE = 1e-5
LR_FINETUNE_MIN = 1e-6
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
PATIENCE = 8
WORST_CLASSES_TO_LOG = 20
# Ignore tiny eval classes when ranking worst/best recall (n=1–4 is mostly noise).
MIN_RECALL_SUPPORT = 8
# Keep 0 when images are cached in RAM. DataLoader workers on MPS would copy
# the cache per process and fight the GPU for unified memory.
NUM_WORKERS = 0
CACHE_IMAGES = True
# For RAM cache only: pre-resize training crops so augmentation runs faster
# and cache uses less memory. Validation/test already get letterboxed to
# IMG_SIZE at cache time, so only train benefits from this.
CACHE_TRAIN_RESIZE_FACTOR = 1.2
CACHE_EVAL_AS_UINT8 = True
BASE_MODEL = "tf_efficientnetv2_s.in21k"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# scripts/identification/config.py → project root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "trainer" / "models"
PROCESSED_DIR = PROJECT_ROOT / "trainer" / "images_processed"
