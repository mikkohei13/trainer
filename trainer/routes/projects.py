from flask import Blueprint, redirect, render_template, request, url_for

from trainer import db
from trainer.images import IMAGES_DIR, project_stats

bp = Blueprint("projects", __name__)


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
    return render_template("project.html", project=project, stats=stats)
