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

## Read more

See ARCHITECTURE.md for the system architecture details.

See AGENTS.md for the development principles and product constraints.

## Notes

- iNat observations often contain multiple nearly similar images of the same individual. These can cause leaking/overfitting. 
- Quality model has trained on images that were first cropped using object detection model.