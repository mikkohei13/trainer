# Trainer

This is an application to manage the process of training ML image classification models to identify insect species from images. It's under development and will have following features once it's ready:
- Fetching updated images from different sources, e.g. api.laji.fi (done)
- Training object detection model to crop the images to the insect species (done)
- Classify the images to remove images that are not insects (done)
- Classify the images based on their quality, e.g. blurry, low resolution, etc. (done)
- Classify the images based on their life stage, e.g. larva, pupa, adult, etc.
- Harmonize taxon names to FinBIF accepted names
- Augment image data
- Train a model to identify the insect species using a selection of the images
- Evaluate the model performance

## Run

```bash
uv sync
uv run flask --app trainer.app run --reload --port 5001
```

Open http://127.0.0.1:5001/ in a browser.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

## Read more

See ARCHITECTURE.md for the system architecture details.

See AGENTS.md for the development principles and product constraints.
