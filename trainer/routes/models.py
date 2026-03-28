"""Blueprint for training run management."""

from flask import Blueprint, redirect, url_for

from trainer import db, training

bp = Blueprint("models", __name__)


@bp.post("/projects/<taxon>/train")
def start_training(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("projects.index"))

    run_id = db.create_training_run(project["id"])
    training.start_training_process(run_id, taxon)

    return redirect(url_for("projects.project_detail", taxon=taxon))
