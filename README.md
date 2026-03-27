# Trainer

This is an application to manage the process of training ML image classification models to identify insect species from images. It's under development and will have following features once it's ready:
- Fetching updated images from different sources, e.g. api.laji.fi
- Training object detection model to crop the images to the insect species
- Classify the images to remove images that are not insects
- Classify the images based on their quality, e.g. blurry, low resolution, etc.
- Classify the images based on their life stage, e.g. larva, pupa, adult, etc.
- Train a model to identify the insect species using a selection of the images
- Evaluate the model performance

## Run

```bash
uv sync
uv run flask --app trainer.app run --reload
```

Open http://127.0.0.1:5000/ in a browser.

## Project structure

- `trainer`: Flask web app for UI and API
- `scripts`: Helper scripts for data fetching and processing, to be run from command line

## Development principles

- Keep it simple and clear.
- This is one-person app, so avoid premature optimization.
- You can expect necessary data exists, so no need for complex fallback or error messaging logic.
- Add unit tests to ./tests for the most important features.
- Avoid ternary operators and other shorthands, instead use clear if/else statements.
- Avoid unnecessary complexity:
  - No accessibility features
  - No authentication
  - Desktop-optimized UI, no mobile support
  - No JavaScript frameworks
