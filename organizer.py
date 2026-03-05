"""Phase 5: Organize detected and labeled ballet photos into dancer folders.

Modes:
  --mode symlink  (default) Create organized/<dancer_name>/ with symlinks to originals.
  --mode copy     Duplicate files into dancer folders.
  --mode csv      Export a CSV: filename, file_path, dancer_name(s), cluster_id.
"""

import argparse
import csv
import logging
import os
import re
import shutil
import sys
from collections import defaultdict

from tqdm import tqdm

import config
import db
import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

UNLABELED_FOLDER = "unlabeled"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Convert a dancer name into a filesystem-safe folder name.

    Replaces runs of non-alphanumeric / non-space characters with underscores,
    collapses whitespace, strips leading/trailing junk, and lowercases.
    """
    # Replace characters that are problematic on common filesystems.
    safe = re.sub(r"[^\w\s-]", "_", name)
    # Collapse internal whitespace to a single space.
    safe = re.sub(r"\s+", " ", safe).strip()
    # Avoid empty result.
    return safe if safe else "unknown"


def _ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _unique_dest(dest_path: str) -> str:
    """Return *dest_path* with a numeric suffix if the name already exists.

    Prevents overwriting when two source images have the same filename.
    """
    if not os.path.exists(dest_path):
        return dest_path

    base, ext = os.path.splitext(dest_path)
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_image_face_data(conn) -> dict:
    """Query the database and return a mapping of image data to face info.

    Returns a dict keyed by image id::

        {
            image_id: {
                "file_path": str,
                "faces": [
                    {"dancer_name": str | None, "cluster_id": int | None},
                    ...
                ],
            },
            ...
        }

    Only images that have at least one detected face are included.
    """
    rows = conn.execute(
        """
        SELECT i.id       AS image_id,
               i.file_path,
               f.dancer_name,
               f.cluster_id
          FROM images i
          JOIN faces f ON f.image_id = i.id
         ORDER BY i.id
        """
    ).fetchall()

    images: dict = {}
    for row in rows:
        img_id = row["image_id"]
        if img_id not in images:
            images[img_id] = {
                "file_path": row["file_path"],
                "faces": [],
            }
        images[img_id]["faces"].append({
            "dancer_name": row["dancer_name"],
            "cluster_id": row["cluster_id"],
        })

    return images


def classify_images(images: dict) -> tuple[
    dict[str, list[str]],  # dancer_name -> [file_path, ...]
    list[str],              # unlabeled file paths
    int,                    # count of no-face images (for reporting only)
]:
    """Split images into labeled (per dancer), unlabeled, and no-face buckets.

    A photo can appear under multiple dancers if different faces are identified
    in the same image.  Photos where faces exist but none have a dancer_name
    go into the unlabeled list.
    """
    dancer_photos: dict[str, list[str]] = defaultdict(list)
    unlabeled: list[str] = []

    for img in images.values():
        file_path = img["file_path"]
        names_in_image: set[str] = set()
        has_unnamed = False

        for face in img["faces"]:
            name = face["dancer_name"]
            if name:
                names_in_image.add(name)
            else:
                has_unnamed = True

        if names_in_image:
            for name in names_in_image:
                dancer_photos[name].append(file_path)
        else:
            # All faces in the image are unnamed.
            unlabeled.append(file_path)

    return dict(dancer_photos), unlabeled, 0  # no-face count filled later


# ---------------------------------------------------------------------------
# Organizer actions
# ---------------------------------------------------------------------------

def _place_file(src: str, dest_dir: str, mode: str) -> None:
    """Symlink or copy *src* into *dest_dir*, handling name collisions."""
    filename = os.path.basename(src)
    dest = _unique_dest(os.path.join(dest_dir, filename))

    if mode == "symlink":
        os.symlink(os.path.abspath(src), dest)
    elif mode == "copy":
        shutil.copy2(src, dest)


def organize_files(
    dancer_photos: dict[str, list[str]],
    unlabeled: list[str],
    mode: str,
    output_dir: str,
) -> dict[str, int]:
    """Create the folder structure and place files.

    Returns summary counts.
    """
    # Wipe previous output so we don't accumulate stale links/copies.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    _ensure_dir(output_dir)

    photos_organized = 0
    unique_paths: set[str] = set()

    # --- Named dancer folders ---
    all_items: list[tuple[str, str]] = []
    for dancer_name, paths in dancer_photos.items():
        folder = sanitize_name(dancer_name)
        dest_dir = os.path.join(output_dir, folder)
        _ensure_dir(dest_dir)
        for p in paths:
            all_items.append((p, dest_dir))
            unique_paths.add(p)

    # --- Unlabeled folder ---
    if unlabeled:
        dest_dir = os.path.join(output_dir, UNLABELED_FOLDER)
        _ensure_dir(dest_dir)
        for p in unlabeled:
            all_items.append((p, dest_dir))
            unique_paths.add(p)

    for src, dest_dir in tqdm(all_items, desc="Organizing photos", unit="file"):
        _place_file(src, dest_dir, mode)
        photos_organized += 1

    return {
        "dancers": len(dancer_photos),
        "organized": photos_organized,
        "unlabeled": len(unlabeled),
    }


def export_csv(
    images: dict,
    output_dir: str,
) -> str:
    """Write a CSV report of all images with detected faces.

    Returns the path to the written CSV file.
    """
    _ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "photo_report.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "file_path", "dancer_names", "cluster_ids"])

        for img in sorted(images.values(), key=lambda x: x["file_path"]):
            file_path = img["file_path"]
            filename = os.path.basename(file_path)
            names = sorted({
                f["dancer_name"] for f in img["faces"] if f["dancer_name"]
            })
            cluster_ids = sorted({
                str(f["cluster_id"]) for f in img["faces"]
                if f["cluster_id"] is not None
            })
            writer.writerow([
                filename,
                file_path,
                "; ".join(names) if names else "",
                "; ".join(cluster_ids) if cluster_ids else "",
            ])

    return csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Organize ballet photos into per-dancer folders.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy", "csv"],
        default="symlink",
        help="Organization mode (default: symlink).",
    )
    parser.add_argument(
        "--output",
        default=config.ORGANIZED_DIR,
        help=f"Output directory (default: {config.ORGANIZED_DIR}).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mode: str = args.mode
    output_dir: str = args.output

    db.init_db()
    conn = db.get_connection()

    # Total images in DB (for the no-faces-skipped count).
    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    # Load only images that have detected faces.
    images = load_image_face_data(conn)
    conn.close()

    no_faces_skipped = total_images - len(images)

    if not images:
        log.info("No images with detected faces found in the database.")
        print(
            f"\nSummary: 0 dancers, 0 photos organized, "
            f"0 unlabeled, {no_faces_skipped} skipped (no faces)."
        )
        return

    dancer_photos, unlabeled, _ = classify_images(images)

    # --- Execute the chosen mode ---
    if mode == "csv":
        csv_path = export_csv(images, output_dir)
        print(f"\nCSV exported to {csv_path}")
        print(
            f"  {len(images)} images with faces, "
            f"{len(dancer_photos)} dancers identified, "
            f"{len(unlabeled)} unlabeled, "
            f"{no_faces_skipped} skipped (no faces)."
        )
        return

    summary = organize_files(dancer_photos, unlabeled, mode, output_dir)
    summary["no_faces_skipped"] = no_faces_skipped

    print(
        f"\nOrganization complete ({mode} mode):"
        f"\n  {summary['dancers']} dancer(s) identified"
        f"\n  {summary['organized']} photo(s) organized"
        f"\n  {summary['unlabeled']} unlabeled"
        f"\n  {summary['no_faces_skipped']} skipped (no faces)"
        f"\n  Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
