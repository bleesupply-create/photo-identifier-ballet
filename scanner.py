"""Phase 1: Scan a photo directory and register image files in the database."""

from __future__ import annotations

import logging
import os
import sys

from tqdm import tqdm

import config
import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def discover_images(photo_dir: str) -> list[str]:
    """Walk *photo_dir* recursively and return paths for supported image files.

    Files that cannot be accessed (permission errors, broken symlinks, etc.)
    are logged as warnings and skipped.
    """
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(photo_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in config.SUPPORTED_EXTENSIONS:
                continue
            full_path = os.path.join(dirpath, fname)
            try:
                os.stat(full_path)  # verify the file is accessible
            except OSError as exc:
                log.warning("Skipping unreadable file %s: %s", full_path, exc)
                continue
            found.append(full_path)
    return found


def scan(photo_dir: str | None = None) -> dict[str, int]:
    """Scan *photo_dir* for images and insert new entries into the database.

    Returns a summary dict with keys: ``total``, ``added``, ``skipped``.
    """
    photo_dir = photo_dir or config.PHOTO_DIR

    if not os.path.isdir(photo_dir):
        raise FileNotFoundError(f"Photo directory does not exist: {photo_dir}")

    db.init_db()

    # --- Discover files on disk ---
    log.info("Scanning %s for images...", photo_dir)
    image_paths = discover_images(photo_dir)
    total = len(image_paths)
    log.info("Found %d image file(s).", total)

    if total == 0:
        return {"total": 0, "added": 0, "skipped": 0}

    # --- Load already-scanned paths from the database ---
    conn = db.get_connection()
    cursor = conn.execute("SELECT file_path FROM images")
    existing_paths: set[str] = {row["file_path"] for row in cursor}

    added = 0
    skipped = 0

    for path in tqdm(image_paths, desc="Registering images", unit="file"):
        if path in existing_paths:
            skipped += 1
            continue

        try:
            file_size = os.path.getsize(path)
        except OSError as exc:
            log.warning("Cannot read size for %s: %s", path, exc)
            skipped += 1
            continue

        try:
            conn.execute(
                "INSERT INTO images (file_path, file_size) VALUES (?, ?)",
                (path, file_size),
            )
            added += 1
        except Exception:
            # Should not happen because we checked existing_paths, but guard
            # against races or unexpected duplicates.
            log.warning("Skipping duplicate entry: %s", path)
            skipped += 1

    conn.commit()
    conn.close()

    return {"total": total, "added": added, "skipped": skipped}


def main() -> None:
    summary = scan()
    print(
        f"\nScan complete: {summary['total']} found, "
        f"{summary['added']} added, "
        f"{summary['skipped']} skipped (already in DB)."
    )


if __name__ == "__main__":
    main()
