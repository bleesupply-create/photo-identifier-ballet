"""Phase 4: Flask web UI for reviewing, labeling, and organizing ballet photos."""

import io
import logging
import os

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from PIL import Image

import config
import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(
    __name__,
    instance_path=os.path.join(config.PROJECT_DIR, "instance"),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cluster_stats(conn):
    """Return cluster rows augmented with sample face IDs, sorted by size."""
    clusters = conn.execute("""
        SELECT c.id, c.dancer_name, c.face_count, c.representative_face_id
        FROM clusters c
        ORDER BY c.face_count DESC
    """).fetchall()

    result = []
    for c in clusters:
        sample_faces = conn.execute("""
            SELECT id FROM faces
            WHERE cluster_id = ?
            ORDER BY detection_score DESC
            LIMIT ?
        """, (c["id"], config.CLUSTER_PREVIEW_COUNT)).fetchall()

        result.append({
            "id": c["id"],
            "dancer_name": c["dancer_name"],
            "face_count": c["face_count"],
            "representative_face_id": c["representative_face_id"],
            "sample_face_ids": [f["id"] for f in sample_faces],
        })
    return result


def _crop_face(image_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
    """Crop a face region from an image and resize to thumbnail dimensions."""
    img = Image.open(image_path)
    img = img.convert("RGB")

    w, h = img.size
    # Bounding box coordinates are stored normalized to [0, 1].
    x1 = max(0, int(bbox_x1 * w))
    y1 = max(0, int(bbox_y1 * h))
    x2 = min(w, int(bbox_x2 * w))
    y2 = min(h, int(bbox_y2 * h))

    # Add a small margin around the face (15% of face size)
    face_w = x2 - x1
    face_h = y2 - y1
    margin_x = int(face_w * 0.15)
    margin_y = int(face_h * 0.15)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    cropped = img.crop((x1, y1, x2, y2))
    cropped.thumbnail(config.THUMBNAIL_SIZE, Image.LANCZOS)
    return cropped


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Dashboard: grid of clusters sorted by face count."""
    conn = db.get_connection()
    try:
        search_query = request.args.get("q", "").strip()
        clusters = _get_cluster_stats(conn)

        if search_query:
            clusters = [
                c for c in clusters
                if c["dancer_name"]
                and search_query.lower() in c["dancer_name"].lower()
            ]

        total_faces = conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"]
        total_images = conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
        labeled_clusters = sum(1 for c in clusters if c["dancer_name"])

        return render_template(
            "index.html",
            clusters=clusters,
            search_query=search_query,
            total_faces=total_faces,
            total_images=total_images,
            total_clusters=len(clusters),
            labeled_clusters=labeled_clusters,
        )
    finally:
        conn.close()


@app.route("/cluster/<int:cluster_id>")
def cluster_detail(cluster_id):
    """Detail view for a single cluster: all faces, naming, merge controls."""
    conn = db.get_connection()
    try:
        cluster = conn.execute(
            "SELECT * FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
        if not cluster:
            abort(404)

        faces = conn.execute("""
            SELECT f.id, f.image_id, f.detection_score, f.dancer_name,
                   i.file_path
            FROM faces f
            JOIN images i ON f.image_id = i.id
            WHERE f.cluster_id = ?
            ORDER BY f.detection_score DESC
        """, (cluster_id,)).fetchall()

        # Other clusters for the merge dropdown
        other_clusters = conn.execute("""
            SELECT id, dancer_name, face_count
            FROM clusters
            WHERE id != ?
            ORDER BY face_count DESC
        """, (cluster_id,)).fetchall()

        return render_template(
            "cluster.html",
            cluster=cluster,
            faces=faces,
            other_clusters=other_clusters,
        )
    finally:
        conn.close()


@app.route("/cluster/<int:cluster_id>/name", methods=["POST"])
def set_cluster_name(cluster_id):
    """Set or update the dancer name for a cluster."""
    conn = db.get_connection()
    try:
        name = request.form.get("dancer_name", "").strip() or None
        conn.execute(
            "UPDATE clusters SET dancer_name = ? WHERE id = ?",
            (name, cluster_id),
        )
        # Also update all faces in this cluster
        conn.execute(
            "UPDATE faces SET dancer_name = ? WHERE cluster_id = ?",
            (name, cluster_id),
        )
        conn.commit()

        # Return JSON for AJAX calls, redirect for form submissions
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": True, "dancer_name": name})
        return redirect(url_for("cluster_detail", cluster_id=cluster_id))
    finally:
        conn.close()


@app.route("/cluster/<int:cluster_id>/merge", methods=["POST"])
def merge_clusters(cluster_id):
    """Merge another cluster into this one."""
    conn = db.get_connection()
    try:
        target_id = request.form.get("target_cluster_id", type=int)
        if target_id is None or target_id == cluster_id:
            abort(400)

        target = conn.execute(
            "SELECT * FROM clusters WHERE id = ?", (target_id,)
        ).fetchone()
        if not target:
            abort(404)

        # Move all faces from target cluster into this cluster
        conn.execute(
            "UPDATE faces SET cluster_id = ? WHERE cluster_id = ?",
            (cluster_id, target_id),
        )

        # Update face count
        new_count = conn.execute(
            "SELECT COUNT(*) AS n FROM faces WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()["n"]
        conn.execute(
            "UPDATE clusters SET face_count = ? WHERE id = ?",
            (new_count, cluster_id),
        )

        # Propagate dancer name if the current cluster has one
        current = conn.execute(
            "SELECT dancer_name FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
        if current["dancer_name"]:
            conn.execute(
                "UPDATE faces SET dancer_name = ? WHERE cluster_id = ?",
                (current["dancer_name"], cluster_id),
            )

        # Delete the now-empty target cluster
        conn.execute("DELETE FROM clusters WHERE id = ?", (target_id,))
        conn.commit()

        return redirect(url_for("cluster_detail", cluster_id=cluster_id))
    finally:
        conn.close()


@app.route("/face/<int:face_id>/remove", methods=["POST"])
def remove_face_from_cluster(face_id):
    """Remove a face from its cluster (mark as 'not this person')."""
    conn = db.get_connection()
    try:
        face = conn.execute(
            "SELECT * FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        if not face:
            abort(404)

        old_cluster_id = face["cluster_id"]

        # Set cluster_id to NULL (unclustered) and clear dancer name
        conn.execute(
            "UPDATE faces SET cluster_id = NULL, dancer_name = NULL WHERE id = ?",
            (face_id,),
        )

        # Update the old cluster's face count
        if old_cluster_id is not None:
            new_count = conn.execute(
                "SELECT COUNT(*) AS n FROM faces WHERE cluster_id = ?",
                (old_cluster_id,),
            ).fetchone()["n"]
            conn.execute(
                "UPDATE clusters SET face_count = ? WHERE id = ?",
                (new_count, old_cluster_id),
            )
            # If the cluster is now empty, delete it
            if new_count == 0:
                conn.execute(
                    "DELETE FROM clusters WHERE id = ?", (old_cluster_id,)
                )

        conn.commit()

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": True})
        if old_cluster_id is not None:
            return redirect(url_for("cluster_detail", cluster_id=old_cluster_id))
        return redirect(url_for("index"))
    finally:
        conn.close()


@app.route("/image/<int:image_id>")
def image_view(image_id):
    """Full image view with face bounding box overlays."""
    conn = db.get_connection()
    try:
        image = conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if not image:
            abort(404)

        faces = conn.execute("""
            SELECT f.id, f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
                   f.cluster_id, f.dancer_name, f.detection_score
            FROM faces f
            WHERE f.image_id = ?
        """, (image_id,)).fetchall()

        return render_template(
            "image.html",
            image=image,
            faces=faces,
        )
    finally:
        conn.close()


@app.route("/image/<int:image_id>/file")
def image_file(image_id):
    """Serve the original image file."""
    conn = db.get_connection()
    try:
        image = conn.execute(
            "SELECT file_path FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if not image:
            abort(404)

        file_path = image["file_path"]
        if not os.path.isfile(file_path):
            abort(404)

        return send_file(file_path)
    finally:
        conn.close()


@app.route("/thumbnail/<int:face_id>")
def thumbnail(face_id):
    """Serve a face thumbnail, cropping and caching as needed."""
    conn = db.get_connection()
    try:
        face = conn.execute("""
            SELECT f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
                   i.file_path
            FROM faces f
            JOIN images i ON f.image_id = i.id
            WHERE f.id = ?
        """, (face_id,)).fetchall()

        if not face:
            abort(404)
        face = face[0]

        # Check cache first
        os.makedirs(config.THUMBNAIL_DIR, exist_ok=True)
        cache_path = os.path.join(config.THUMBNAIL_DIR, f"{face_id}.jpg")

        if not os.path.isfile(cache_path):
            if not os.path.isfile(face["file_path"]):
                abort(404)
            cropped = _crop_face(
                face["file_path"],
                face["bbox_x1"],
                face["bbox_y1"],
                face["bbox_x2"],
                face["bbox_y2"],
            )
            cropped.save(cache_path, "JPEG", quality=85)

        return send_file(cache_path, mimetype="image/jpeg")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db.init_db()
    port = int(os.environ.get("PORT", config.FLASK_PORT))
    log.info(
        "Starting Ballet Photo Organizer UI at http://%s:%s",
        config.FLASK_HOST,
        port,
    )
    app.run(
        host=config.FLASK_HOST,
        port=port,
        debug=True,
    )
