from flask import Blueprint, abort, redirect, render_template, request, url_for
import random

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


def _quality_bucket_label(bucket_key: str) -> str:
    if bucket_key == "unrated":
        return "Unrated"
    if bucket_key == "rated":
        return "Rated"
    raise ValueError("invalid quality bucket")


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
    _, box_map = db.project_annotation_state(taxon)
    quality_map = db.get_image_quality_map(taxon)
    with_boxes = {p for p, n in box_map.items() if n >= 1}
    quality_stats = {
        "rated": len([p for p in with_boxes if p in quality_map]),
        "unrated": len([p for p in with_boxes if p not in quality_map]),
    }
    training_runs = db.get_training_runs(project["id"])
    quality_training_runs = db.get_quality_training_runs(project["id"])
    return render_template(
        "project.html",
        project=project,
        stats=stats,
        quality_stats=quality_stats,
        training_runs=training_runs,
        quality_training_runs=quality_training_runs,
    )


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


@bp.get("/projects/<taxon>/quality-gallery")
def quality_gallery(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("projects.index"))

    bucket_key = request.args.get("bucket", "unrated").strip().lower()
    if bucket_key not in {"unrated", "rated"}:
        abort(404)

    try:
        page = int(request.args.get("page", "1"))
    except (ValueError, TypeError):
        page = 1
    page = max(1, page)

    _, box_map = db.project_annotation_state(taxon)
    quality_map = db.get_image_quality_map(taxon)
    with_boxes = sorted(p for p, n in box_map.items() if n >= 1)
    buckets = {
        "unrated": [p for p in with_boxes if p not in quality_map],
        "rated": [p for p in with_boxes if p in quality_map],
    }

    paths = buckets[bucket_key]
    total = len(paths)
    total_pages = max(1, (total + GALLERY_PER_PAGE - 1) // GALLERY_PER_PAGE) if total else 1
    if page > total_pages:
        page = total_pages

    start = (page - 1) * GALLERY_PER_PAGE
    page_paths = paths[start : start + GALLERY_PER_PAGE]
    page_quality_map = {p: quality_map[p] for p in page_paths if p in quality_map}

    return render_template(
        "quality_gallery.html",
        project=project,
        bucket_key=bucket_key,
        bucket_label=_quality_bucket_label(bucket_key),
        image_paths=page_paths,
        image_quality_map=page_quality_map,
        page=page,
        total_pages=total_pages,
        total_count=total,
        per_page=GALLERY_PER_PAGE,
    )


@bp.get("/projects/<taxon>/evaluate")
def evaluate_models(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("projects.index"))

    try:
        n = int(request.args.get("n", "20"))
    except (TypeError, ValueError):
        n = 20
    n = max(1, min(200, n))

    buckets = project_annotation_buckets(taxon)
    candidates = buckets.get("not_annotated", [])
    sampled = candidates[:]
    random.shuffle(sampled)
    image_paths = sampled[: min(n, len(sampled))]

    has_detection_model = db.get_active_model_path_for_taxon(taxon) is not None
    has_quality_model = db.get_active_quality_model_path_for_taxon(taxon) is not None

    return render_template(
        "evaluate.html",
        project=project,
        requested_n=n,
        available_count=len(candidates),
        image_paths=image_paths,
        has_detection_model=has_detection_model,
        has_quality_model=has_quality_model,
    )
