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
    annotation_paths,
    image_path_under_images_root,
    normalize_annotation_bucket,
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

    bucket_key = normalize_annotation_bucket(request.args.get("bucket"))
    requested_path = request.args.get("path", "")
    paths = annotation_paths(taxon, bucket_key, include=requested_path or None)

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
            bucket_key=bucket_key,
        )

    image_path = requested_path
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

    def _annotate_url(path: str) -> str:
        if bucket_key:
            return url_for("annotate.annotate", taxon=taxon, path=path, bucket=bucket_key)
        return url_for("annotate.annotate", taxon=taxon, path=path)

    prev_url = _annotate_url(prev_path) if prev_path else None
    next_url = _annotate_url(next_path) if next_path else None

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
        bucket_key=bucket_key,
    )
