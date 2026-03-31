# Architecture

- **Stack:** Python 3.11+, Flask, Jinja2 (`trainer/templates/`), SQLite via `sqlite3` (`trainer/db.py` → `trainer.db` at repo root).

- **Web app (`trainer/app.py`):** Server-rendered pages. `GET /` lists projects; `POST /projects` creates a project (slug = taxon name); `GET /projects/<taxon>` shows per-collection and per-species-folder image counts from scanning `trainer/images/`.

- **Data layout:** Project metadata is rows in `project(taxon, created)`. Training images live on disk under `trainer/images/<taxon>/<collection>/<species_folder>/` (e.g. `Genus_species/*.jpg`). The app counts files with image extensions; it does not embed pixels in the DB.

- **Annotation UI (`GET /annotate/<taxon>`):** Server renders `annotate.html` and injects the starting image index (from `?i=` query parameter) as a Jinja variable. The browser then drives everything with vanilla JS:
  - Fetches the full sorted image list from `GET /api/images/<taxon>`.
  - Loads images via `GET /images/<path>` (served from `trainer/images/`).
  - Fetches and saves bounding boxes via `GET|POST /api/annotations/<path>`. Each POST is a full replace for that image.
  - Uses [Annotorious](https://new.annotorious.com/) (CDN, no build step) to draw and display rectangles. Annotorious stores coordinates in the image's natural pixel space, so browser zoom does not affect saved values.
  - Updates `?i=` in the URL via `history.replaceState` on every navigation, so the page can be bookmarked or reloaded at the current image.
  - Annotation state (bounding boxes and "no organisms" flag) is stored in SQLite tables `bounding_box` and `image_no_organism`.

- **Scripts (`scripts/`):** CLI utilities (e.g. FinBIF fetch) run outside Flask; repo-root `secrets.py` holds API keys. Not part of the web process.

