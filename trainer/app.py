from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

from trainer import db

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

app = Flask(__name__)

db.init_db()




def _count_images(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def _project_stats(taxon: str) -> dict:
    project_dir = IMAGES_DIR / taxon
    collections = []
    taxa_rows = []

    if project_dir.is_dir():
        for collection_dir in sorted(project_dir.iterdir()):
            if not collection_dir.is_dir():
                continue
            collection_count = _count_images(collection_dir)
            collections.append({
                "name": collection_dir.name,
                "count": collection_count,
            })
            for taxon_dir in sorted(collection_dir.iterdir()):
                if not taxon_dir.is_dir():
                    continue
                count = sum(
                    1 for f in taxon_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                )
                name = taxon_dir.name
                existing = next((t for t in taxa_rows if t["taxon"] == name), None)
                if existing:
                    existing["count"] += count
                else:
                    taxa_rows.append({"taxon": name, "count": count})

    return {"collections": collections, "taxa": taxa_rows}


@app.get("/")
def index():
    projects = db.get_projects()
    return render_template("index.html", projects=projects)


@app.post("/projects")
def create_project():
    taxon = request.form.get("taxon", "").strip().lower()
    if not taxon:
        return redirect(url_for("index"))
    project_dir = IMAGES_DIR / taxon
    project_dir.mkdir(parents=True, exist_ok=True)
    try:
        db.create_project(taxon)
    except Exception:
        pass  # already exists — silently ignore
    return redirect(url_for("project_detail", taxon=taxon))


@app.get("/projects/<taxon>")
def project_detail(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("index"))
    stats = _project_stats(taxon)
    return render_template("project.html", project=project, stats=stats)
