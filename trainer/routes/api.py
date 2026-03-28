from flask import Blueprint, abort, jsonify, request

from trainer import db
from trainer.images import image_path_under_images_root, list_project_image_paths

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.get("/images/<taxon>")
def list_images(taxon: str):
    project = db.get_project(taxon)
    if project is None:
        abort(404)
    paths = list_project_image_paths(taxon)
    return jsonify(paths)


@bp.get("/annotations/<path:image_path>")
def get_annotations(image_path: str):
    if not image_path_under_images_root(image_path):
        abort(404)
    data = db.get_annotations(image_path)
    return jsonify(data)


@bp.post("/annotations/<path:image_path>")
def post_annotations(image_path: str):
    if not image_path_under_images_root(image_path):
        abort(404)
    payload = request.get_json(silent=True)
    if payload is None:
        abort(400)
    if "boxes" not in payload or "no_organism" not in payload:
        abort(400)
    boxes = payload["boxes"]
    no_organism = payload["no_organism"]
    if not isinstance(boxes, list):
        abort(400)
    if not isinstance(no_organism, bool):
        abort(400)
    normalized_boxes = []
    for item in boxes:
        if not isinstance(item, dict):
            abort(400)
        required = ("anno_id", "x", "y", "w", "h")
        for key in required:
            if key not in item:
                abort(400)
        normalized_boxes.append({
            "anno_id": str(item["anno_id"]),
            "x": float(item["x"]),
            "y": float(item["y"]),
            "w": float(item["w"]),
            "h": float(item["h"]),
        })
    if no_organism and len(normalized_boxes) > 0:
        abort(400)
    db.save_annotations(image_path, normalized_boxes, no_organism)
    return jsonify({"ok": True})
