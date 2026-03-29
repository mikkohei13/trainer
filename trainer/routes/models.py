"""Blueprint for training run management."""

from flask import Blueprint, redirect, request, url_for

from trainer import db, training

bp = Blueprint("models", __name__)


@bp.post("/projects/<taxon>/active-model")
def set_active_model(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("projects.index"))

    raw = request.form.get("run_id", "").strip()
    if raw == "":
        try:
            db.set_active_training_run(taxon, None)
        except ValueError:
            pass
        return redirect(url_for("projects.project_detail", taxon=taxon))

    try:
        run_id = int(raw)
    except ValueError:
        return redirect(url_for("projects.project_detail", taxon=taxon))

    try:
        db.set_active_training_run(taxon, run_id)
    except ValueError:
        pass

    return redirect(url_for("projects.project_detail", taxon=taxon))


@bp.post("/projects/<taxon>/train")
def start_training(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("projects.index"))

    run_id = db.create_training_run(project["id"])
    training.start_training_process(run_id, taxon)

    return redirect(url_for("projects.project_detail", taxon=taxon))
