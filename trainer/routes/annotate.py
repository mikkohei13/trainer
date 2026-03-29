from flask import (
    Blueprint,
    abort,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from trainer import db
from trainer.images import (
    IMAGES_DIR,
    image_path_under_images_root,
    list_project_image_paths,
)

bp = Blueprint("annotate", __name__)


@bp.get("/images/<path:filename>")
def serve_image(filename: str):
    if not image_path_under_images_root(filename):
        abort(404)
    path = IMAGES_DIR / filename
    if not path.is_file():
        abort(404)
    return send_from_directory(IMAGES_DIR, filename)


@bp.get("/annotate/<taxon>")
def annotate(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        return redirect(url_for("projects.index"))

    paths = list_project_image_paths(taxon)

    if not paths:
        return render_template(
            "annotate.html",
            project=project,
            image_path=None,
            annotations=None,
            prev_url=None,
            next_url=None,
            current_num=0,
            total=0,
            show_detect=False,
            detect_api_url="",
        )

    image_path = request.args.get("path", "")
    if image_path not in paths:
        try:
            i = int(request.args.get("i", 1))
        except (ValueError, TypeError):
            i = 1
        i = max(1, min(i, len(paths)))
        image_path = paths[i - 1]

    idx = paths.index(image_path)
    prev_path = paths[idx - 1] if idx > 0 else None
    next_path = paths[idx + 1] if idx < len(paths) - 1 else None

    prev_url = (
        url_for("annotate.annotate", taxon=taxon, path=prev_path) if prev_path else None
    )
    next_url = (
        url_for("annotate.annotate", taxon=taxon, path=next_path) if next_path else None
    )

    annotations = db.get_annotations(image_path)

    is_unannotated = (
        not annotations["no_organism"] and len(annotations["boxes"]) == 0
    )
    has_active_model = db.get_active_model_path_for_taxon(taxon) is not None
    show_detect = is_unannotated and has_active_model
    detect_api_url = url_for("api.detect", taxon=taxon)

    return render_template(
        "annotate.html",
        project=project,
        image_path=image_path,
        annotations=annotations,
        prev_url=prev_url,
        next_url=next_url,
        current_num=idx + 1,
        total=len(paths),
        show_detect=show_detect,
        detect_api_url=detect_api_url,
    )
