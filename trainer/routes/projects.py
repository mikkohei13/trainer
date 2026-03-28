from flask import Blueprint, abort, redirect, render_template, request, url_for

from trainer import db
from trainer.images import (
    IMAGES_DIR,
    normalize_annotation_bucket,
    project_annotation_buckets,
    project_stats,
)

bp = Blueprint("projects", __name__)

GALLERY_PER_PAGE = 100


def _annotation_bucket_label(bucket_key: str) -> str:
    if bucket_key == "not_annotated":
        return "Not annotated"
    n = int(bucket_key)
    if n == 0:
        return "No organisms"
    if n == 1:
        return "One organism"
    return f"{n} organisms"


@bp.get("/")
def index():
    projects = db.get_projects()
    return render_template("index.html", projects=projects)


@bp.post("/projects")
def create_project():
    taxon = request.form.get("taxon", "").strip().lower()
    if not taxon:
        return redirect(url_for("projects.index"))
    project_dir = IMAGES_DIR / taxon
    project_dir.mkdir(parents=True, exist_ok=True)
    try:
        db.create_project(taxon)
    except Exception:
        pass  # already exists — silently ignore
    return redirect(url_for("projects.project_detail", taxon=taxon))


@bp.get("/projects/<taxon>")
def project_detail(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("projects.index"))
    stats = project_stats(taxon)
    training_runs = db.get_training_runs(project["id"])
    return render_template("project.html", project=project, stats=stats, training_runs=training_runs)


@bp.get("/projects/<taxon>/annotation-gallery")
def annotation_gallery(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("projects.index"))

    raw = request.args.get("bucket", "")
    bucket_key = normalize_annotation_bucket(raw)
    if bucket_key is None:
        abort(404)

    try:
        page = int(request.args.get("page", "1"))
    except (ValueError, TypeError):
        page = 1
    page = max(1, page)

    buckets = project_annotation_buckets(taxon)
    paths = buckets.get(bucket_key, [])
    total = len(paths)
    total_pages = max(1, (total + GALLERY_PER_PAGE - 1) // GALLERY_PER_PAGE) if total else 1
    if page > total_pages:
        page = total_pages

    start = (page - 1) * GALLERY_PER_PAGE
    page_paths = paths[start : start + GALLERY_PER_PAGE]

    return render_template(
        "annotation_gallery.html",
        project=project,
        bucket_key=bucket_key,
        bucket_label=_annotation_bucket_label(bucket_key),
        image_paths=page_paths,
        page=page,
        total_pages=total_pages,
        total_count=total,
        per_page=GALLERY_PER_PAGE,
    )
