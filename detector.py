"""Phase 2: Face detection and embedding extraction.

Scans all unprocessed images in the database, detects faces using InsightFace's
buffalo_l model, extracts 512-dim embeddings, and stores everything back to the DB.

Usage:
    python3 detector.py
"""

import logging
import multiprocessing
import signal
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from tqdm import tqdm

import config
import db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-worker globals (populated by _worker_init)
# ---------------------------------------------------------------------------
_model = None


def _worker_init():
    """Initializer for each Pool worker.

    Creates a dedicated InsightFace FaceAnalysis instance so models are never
    pickled across process boundaries.  Also ignores SIGINT in workers so the
    parent process handles Ctrl-C gracefully.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    global _model
    from insightface.app import FaceAnalysis

    _model = FaceAnalysis(
        name=config.MODEL_NAME,
        allowed_modules=["detection", "recognition"],
    )
    _model.prepare(ctx_id=0, det_size=config.DET_SIZE)


# ---------------------------------------------------------------------------
# Single-image processing (runs inside a worker)
# ---------------------------------------------------------------------------

def _process_image(row):
    """Detect faces and extract embeddings for one image.

    Parameters
    ----------
    row : tuple
        (image_id, file_path) from the images table.

    Returns
    -------
    dict with keys:
        image_id   : int
        file_path  : str
        width      : int
        height     : int
        faces      : list[dict]   -- each dict has embedding, bbox_*, detection_score
        error      : str | None
    """
    image_id, file_path = row

    result = {
        "image_id": image_id,
        "file_path": file_path,
        "width": None,
        "height": None,
        "faces": [],
        "error": None,
    }

    try:
        img = cv2.imread(file_path)
        if img is None:
            result["error"] = f"Could not read image: {file_path}"
            return result

        h, w = img.shape[:2]
        result["width"] = w
        result["height"] = h

        detected = _model.get(img)

        for face in detected:
            bbox = face.bbox.astype(float)  # [x1, y1, x2, y2] in pixels
            embedding = face.normed_embedding.astype(np.float32)

            result["faces"].append({
                "embedding": embedding.tobytes(),
                "bbox_x1": bbox[0] / w,
                "bbox_y1": bbox[1] / h,
                "bbox_x2": bbox[2] / w,
                "bbox_y2": bbox[3] / h,
                "detection_score": float(face.det_score),
            })

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# Batch DB writer
# ---------------------------------------------------------------------------

def _write_batch(conn, batch):
    """Write a batch of processed results to the database in one transaction.

    Parameters
    ----------
    conn  : sqlite3.Connection
    batch : list[dict]  -- results from _process_image
    """
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.cursor()

    for result in batch:
        image_id = result["image_id"]
        face_count = len(result["faces"])

        if result["error"]:
            log.warning("Skipping image %d (%s): %s",
                        image_id, result["file_path"], result["error"])

        # Update image metadata (mark as processed even on error)
        cursor.execute(
            """UPDATE images
               SET processed_at = ?,
                   face_count   = ?,
                   width        = COALESCE(?, width),
                   height       = COALESCE(?, height)
             WHERE id = ?""",
            (now, face_count, result["width"], result["height"], image_id),
        )

        # Insert face rows
        for face in result["faces"]:
            cursor.execute(
                """INSERT INTO faces
                       (image_id, embedding, bbox_x1, bbox_y1,
                        bbox_x2, bbox_y2, detection_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_id,
                    face["embedding"],
                    face["bbox_x1"],
                    face["bbox_y1"],
                    face["bbox_x2"],
                    face["bbox_y2"],
                    face["detection_score"],
                ),
            )

    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_unprocessed(conn):
    """Return list of (id, file_path) for images not yet processed."""
    rows = conn.execute(
        "SELECT id, file_path FROM images WHERE processed_at IS NULL"
    ).fetchall()
    return [(r["id"], r["file_path"]) for r in rows]


def run():
    """Entry point: detect faces across all unprocessed images."""
    db.init_db()

    conn = db.get_connection()
    work_items = _get_unprocessed(conn)

    if not work_items:
        log.info("No unprocessed images found. Nothing to do.")
        conn.close()
        return

    total = len(work_items)
    log.info("Found %d unprocessed images. Using %d workers.", total, config.NUM_WORKERS)

    # --- Graceful shutdown flag ---
    shutdown_requested = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            # Second Ctrl-C: hard exit
            log.warning("Forced shutdown.")
            sys.exit(1)
        shutdown_requested = True
        log.info("Shutdown requested. Finishing current batch...")

    signal.signal(signal.SIGINT, _handle_sigint)

    # --- Statistics ---
    total_faces = 0
    processed_count = 0
    error_count = 0

    batch = []
    pool = multiprocessing.Pool(
        processes=config.NUM_WORKERS,
        initializer=_worker_init,
    )

    try:
        pbar = tqdm(
            total=total,
            desc="Detecting faces",
            unit="img",
            dynamic_ncols=True,
        )

        for result in pool.imap_unordered(_process_image, work_items):
            if shutdown_requested:
                break

            batch.append(result)
            processed_count += 1
            total_faces += len(result["faces"])
            if result["error"]:
                error_count += 1

            pbar.update(1)
            pbar.set_postfix(
                faces=total_faces,
                errors=error_count,
                refresh=False,
            )

            # Flush batch when it reaches BATCH_SIZE
            if len(batch) >= config.BATCH_SIZE:
                _write_batch(conn, batch)
                batch.clear()

        # Write any remaining results
        if batch:
            _write_batch(conn, batch)
            batch.clear()

        pbar.close()

    except Exception:
        # Write whatever we have so far so progress is not lost
        if batch:
            log.info("Writing %d buffered results before exit...", len(batch))
            _write_batch(conn, batch)
            batch.clear()
        raise

    finally:
        pool.terminate()
        pool.join()
        conn.close()
        signal.signal(signal.SIGINT, original_sigint)

    # --- Summary ---
    log.info(
        "Done. Processed %d images, found %d faces (%d errors).",
        processed_count, total_faces, error_count,
    )


if __name__ == "__main__":
    run()
