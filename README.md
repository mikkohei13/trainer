# Trainer

This is an application to manage the process of training ML image classification models to identify insect species from images. It's under development and will have following features once it's ready:
- Fetching updated images from different sources, e.g. api.laji.fi (done)
- Training object detection model to crop the images to the insect species (done)
- Classify the images to remove images that are not insects (done)
- Classify the images based on their quality, e.g. blurry, low resolution, etc. (done)
- Classify the images based on their life stage, e.g. larva, pupa, adult, etc.
- Harmonize taxon names to FinBIF accepted names (done)
- Augment image data
- Train a model to identify the insect species using a selection of the images
- Evaluate the model performance

## Setup

Download taxon names from FinBIF with MX-code of the taxon, and place the taxon.json file into ./trainer/data/{project_name}/taxon.json:

```bash
curl -X 'GET' \
'https://api.laji.fi/taxa/MX.289596/children?checklist=MR.1&selectedFields=id%2CscientificName%2CtypeOfOccurrenceInFinland%2CtaxonRank%2CobservationCountFinland%2Csynonyms%2CsynonymNames&checklistVersion=current&includeMedia=false&includeDescriptions=false&includeRedListEvaluations=false&includeHidden=false&sortOrder=taxonomic' \
-H 'accept: application/json' \
-H 'Authorization: Bearer TOKEN-HERE' \
-H 'Accept-Language: fi' \
-H 'API-Version: 1' > taxa.json
```

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

## Model training

### Image quality model

The app trains a quality regression model from manually rated images:

1. It uses the active object-detection model to locate insects in each rated image. Missing or unreadable images and images with no detection are skipped.
2. If several insects or insect parts are detected, their boxes are combined. The resulting area is cropped with padding and resized for training.
3. The crops are split reproducibly into 80% training and 20% validation data. At least five usable images are required.
4. A pretrained ResNet-18 is fine-tuned to predict a quality score between 0 and 1. Training keeps the model with the best validation RMSE and stops early when validation performance no longer improves.

### Object detection model

- todo

### Taxon identification model

Decoupled CLI (not in the web app).

First harmonize taxonomy on the UI.

```bash
uv run python scripts/crop_identification_images.py
uv run python scripts/train_identification.py
```

1. **Crop.** The project's active object-detection model finds the highest-confidence insect in each image under `trainer/images/<project>/`. The box is padded 10% and saved under `trainer/images_processed/<project>/` with the same folder layout. Originals are never modified. Existing crops are skipped; images with no detection are skipped.
2. **Labels.** Folder names are mapped through `harmonization.tsv`. The class is the first word of the authoritative name (genus). Unmatched folders (e.g. `Non_insects`) are dropped. Genera with fewer than 10 crops are dropped.
3. **Split.** Remaining crops are split 80/10/10 train/val/test, stratified by genus.
4. **Train.** EfficientNetV2-S pretrained on ImageNet-21k. Inputs are letterboxed to 384×384. Train-only augmentation: flip, 90° multiples plus ±30° rotation, mild scale, brightness/contrast/saturation. Focal Loss (γ=2) plus inverse-sqrt class sampling. Phase A trains the head only; phase B unfreezes the last 60 leaf modules with a lower backbone LR than the head and a cosine LR decay. Early stop on val macro-F1; `best.pt` is written whenever it improves. Logs the 20 genera with worst recall.
5. **Output.** `trainer/models/<project>/identification/<run_id>/` contains `best.pt`, `metrics.json` (val/test top-1, macro-F1, worst-class recalls), `splits.json`, `label_map.json`, and `train.log`.

Does not yet split by observation, so near-duplicate iNat photos can leak across splits.

## Read more

See ARCHITECTURE.md for the system architecture details.

See AGENTS.md for the development principles and product constraints.

## Notes

- iNat observations often contain multiple nearly similar images of the same individual. These can cause leaking/overfitting. Todo: for first taxa, remove images and download again, since now the script takes images accross observations.
- Quality model has trained on images that were first cropped using object detection model.

- Check which taxa have many observations and few images, and why.
  - Verdanus:


- Object detection model notes:
  - No insect or non-Hemiptera: 0.36 and below
  - If no insect found, run taxon identification on the whole image. This way failing OD won't hinder taxon identification.
  - Quality model might be biased, giving higher values for brightly colored species than brown ones. So shoud not use it to categorically filter out images accross taxa.