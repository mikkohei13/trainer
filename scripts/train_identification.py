"""
Train a genus identification model for a project (v2, decoupled CLI).

Expects OD-cropped images from scripts/crop_identification_images.py under
trainer/images_processed/<project>/ (same layout as trainer/images/).

Hardcoded parameters — edit scripts/identification/config.py, then:

    uv run python scripts/crop_identification_images.py   # once / incremental
    uv run python scripts/train_identification.py

Never modifies originals under trainer/images/. Training artifacts go to
trainer/models/<project>/identification/<run_id>/.
"""

from __future__ import annotations

from identification.train import main

if __name__ == "__main__":
    main()
