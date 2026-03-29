from flask import Blueprint, abort, jsonify, request

from trainer import db
from trainer import inference
from trainer.images import (
    IMAGES_DIR,
    image_path_under_images_root,
    image_path_under_taxon_project,
    list_project_image_paths,
)

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


@bp.post("/projects/<taxon>/detect")
def detect(taxon: str):
    if db.get_project(taxon) is None:
        return jsonify({"error": "unknown project"}), 404

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "invalid JSON"}), 400
    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        return jsonify({"error": "missing image_path"}), 400

    if not image_path_under_taxon_project(image_path, taxon):
        return jsonify({"error": "invalid image path"}), 400

    model_path = db.get_active_model_path_for_taxon(taxon)
    if model_path is None:
        return jsonify({"error": "no active model"}), 400

    abs_image = IMAGES_DIR / image_path
    boxes = inference.predict_top_box(model_path, abs_image)
    return jsonify({"boxes": boxes})
