"""Static gallery site generator for the ballet photo organizer.

Reads organized dancer folders, generates thumbnails, and renders
a self-contained static HTML gallery with responsive grids and a
lightbox viewer.

Usage:
    python3 gallery.py
"""

import os
import shutil
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from PIL import Image
from tqdm import tqdm

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = config.SUPPORTED_EXTENSIONS


def _is_image(path: Path) -> bool:
    """Return True if *path* has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _collect_dancers(organized_dir: str) -> list[dict]:
    """Scan *organized_dir* and return a list of dancer info dicts.

    Each dict has keys: name, dir, photos (list of Paths).
    Dancers are sorted alphabetically by name.
    """
    organized = Path(organized_dir)
    if not organized.is_dir():
        print(f"Error: organized directory not found: {organized}")
        sys.exit(1)

    dancers: list[dict] = []
    for entry in sorted(organized.iterdir()):
        if not entry.is_dir():
            continue
        photos = sorted(p for p in entry.iterdir() if p.is_file() and _is_image(p))
        if photos:
            dancers.append({
                "name": entry.name,
                "dir": entry,
                "photos": photos,
            })
    return dancers


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

def _generate_thumbnail(src: Path, dst: Path, width: int) -> None:
    """Create a JPEG thumbnail of *src* at *dst*, scaled to *width* px wide."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        # Handle EXIF orientation
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Compute height preserving aspect ratio
        w, h = img.size
        new_height = int(h * (width / w))
        img = img.resize((width, new_height), Image.LANCZOS)

        # Convert to RGB if necessary (e.g. RGBA PNGs)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        img.save(dst, "JPEG", quality=85, optimize=True)


def generate_thumbnails(
    dancers: list[dict],
    output_dir: Path,
    thumb_width: int,
) -> dict[str, dict[str, str]]:
    """Generate thumbnails for every photo of every dancer.

    Returns a nested dict: dancer_name -> {photo_filename: thumb_relative_path}.
    Skips thumbnails that already exist and are newer than the source.
    """
    thumbs_dir = output_dir / "thumbs"
    thumb_map: dict[str, dict[str, str]] = {}

    # Build the work list
    tasks: list[tuple[Path, Path, str, str]] = []
    for dancer in dancers:
        dancer_name = dancer["name"]
        thumb_map[dancer_name] = {}
        for photo in dancer["photos"]:
            thumb_filename = photo.stem + ".jpg"
            thumb_path = thumbs_dir / dancer_name / thumb_filename
            # Relative path from gallery_output root for use in HTML
            rel = f"thumbs/{dancer_name}/{thumb_filename}"
            thumb_map[dancer_name][photo.name] = rel

            # Skip if thumbnail is fresh
            if thumb_path.exists() and thumb_path.stat().st_mtime >= photo.stat().st_mtime:
                continue
            tasks.append((photo, thumb_path, dancer_name, photo.name))

    if tasks:
        print(f"Generating {len(tasks)} thumbnails ...")
        for src, dst, _, _ in tqdm(tasks, desc="Thumbnails", unit="img"):
            try:
                _generate_thumbnail(src, dst, thumb_width)
            except Exception as exc:
                tqdm.write(f"  Warning: could not thumbnail {src.name}: {exc}")
    else:
        print("All thumbnails up to date.")

    return thumb_map


# ---------------------------------------------------------------------------
# CSS (inline in output for self-contained gallery)
# ---------------------------------------------------------------------------

GALLERY_CSS = """\
/* Ballet Gallery — generated styles */
:root {
    --bg: #fafafa;
    --surface: #ffffff;
    --text: #1a1a2e;
    --text-muted: #6b7280;
    --accent: #8b5cf6;
    --accent-hover: #7c3aed;
    --border: #e5e7eb;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
    --shadow-lg: 0 10px 30px rgba(0,0,0,.12);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
}

/* ---- Header ---- */
.site-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #fff;
    padding: 2.5rem 1.5rem;
    text-align: center;
}
.site-header h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: .02em;
}
.site-header .subtitle {
    color: rgba(255,255,255,.7);
    font-size: .95rem;
    margin-top: .35rem;
}

/* ---- Container ---- */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

/* ---- Dancer card grid (home page) ---- */
.dancer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1.5rem;
}
.dancer-card {
    background: var(--surface);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
    text-decoration: none;
    color: inherit;
    transition: transform .2s, box-shadow .2s;
}
.dancer-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}
.dancer-card .thumb-wrapper {
    width: 100%;
    aspect-ratio: 4 / 3;
    overflow: hidden;
    background: #e5e7eb;
}
.dancer-card .thumb-wrapper img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform .3s;
}
.dancer-card:hover .thumb-wrapper img {
    transform: scale(1.05);
}
.dancer-card .card-body {
    padding: 1rem 1.25rem;
}
.dancer-card .card-body h2 {
    font-size: 1.15rem;
    font-weight: 600;
}
.dancer-card .card-body .photo-count {
    color: var(--text-muted);
    font-size: .875rem;
    margin-top: .2rem;
}

/* ---- Breadcrumb / back link ---- */
.back-link {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
    margin-bottom: 1.5rem;
    font-size: .95rem;
}
.back-link:hover { text-decoration: underline; }

/* ---- Dancer page toolbar ---- */
.dancer-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.dancer-toolbar h1 {
    font-size: 1.75rem;
    font-weight: 700;
}
.btn {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .6rem 1.25rem;
    border: none;
    border-radius: 8px;
    font-size: .9rem;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    transition: background .2s;
}
.btn-primary {
    background: var(--accent);
    color: #fff;
}
.btn-primary:hover { background: var(--accent-hover); }

/* ---- Photo grid (dancer page) ---- */
.photo-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
}
.photo-cell {
    border-radius: var(--radius);
    overflow: hidden;
    cursor: pointer;
    background: #e5e7eb;
    aspect-ratio: 1;
    box-shadow: var(--shadow);
    transition: transform .2s, box-shadow .2s;
}
.photo-cell:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}
.photo-cell img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

/* ---- Lightbox ---- */
.lightbox-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.92);
    z-index: 9999;
    display: none;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}
.lightbox-overlay.active { display: flex; }

.lightbox-img {
    max-width: 92vw;
    max-height: 85vh;
    object-fit: contain;
    border-radius: 4px;
    user-select: none;
}
.lightbox-close {
    position: absolute;
    top: 1rem;
    right: 1.5rem;
    background: none;
    border: none;
    color: #fff;
    font-size: 2rem;
    cursor: pointer;
    z-index: 10001;
    line-height: 1;
    opacity: .8;
    transition: opacity .2s;
}
.lightbox-close:hover { opacity: 1; }

.lightbox-nav {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(255,255,255,.15);
    border: none;
    color: #fff;
    font-size: 2.5rem;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background .2s;
    z-index: 10001;
    line-height: 1;
}
.lightbox-nav:hover { background: rgba(255,255,255,.3); }
.lightbox-prev { left: 1rem; }
.lightbox-next { right: 1rem; }

.lightbox-counter {
    color: rgba(255,255,255,.7);
    font-size: .9rem;
    margin-top: .75rem;
    user-select: none;
}

/* ---- Footer ---- */
.site-footer {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--text-muted);
    font-size: .8rem;
}

/* ---- Responsive ---- */
@media (max-width: 600px) {
    .site-header h1 { font-size: 1.5rem; }
    .photo-grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: .6rem; }
    .dancer-grid { grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }
    .lightbox-nav { width: 44px; height: 44px; font-size: 1.75rem; }
}
"""


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _create_jinja_env() -> Environment:
    """Create a Jinja2 environment pointing at the project templates dir."""
    template_dir = os.path.join(config.PROJECT_DIR, "templates", "gallery")
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=True,
    )


def render_gallery(
    dancers: list[dict],
    thumb_map: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    """Render the full static gallery into *output_dir*."""
    env = _create_jinja_env()

    # -- Write CSS --
    css_path = output_dir / "style.css"
    css_path.write_text(GALLERY_CSS, encoding="utf-8")

    # -- Build template context for each dancer --
    dancer_contexts: list[dict] = []
    for dancer in dancers:
        name = dancer["name"]
        photos_info = []
        for photo in dancer["photos"]:
            thumb_rel = thumb_map.get(name, {}).get(photo.name, "")
            # Copy full-size image and get path relative to gallery root
            full_root_rel = _copy_full_image(photo, output_dir, name)
            # Paths for dancer page (one level deep, so prefix with ../)
            photos_info.append({
                "filename": photo.name,
                "thumb_url": f"../{thumb_rel}",
                "full_url": f"../{full_root_rel}",
            })
        # Sample thumb for home page — relative to gallery root (no ../ prefix)
        sample_thumb = thumb_map[name][dancer["photos"][0].name] if dancer["photos"] else ""
        dancer_contexts.append({
            "name": name,
            "photo_count": len(photos_info),
            "sample_thumb": sample_thumb,
            "photos": photos_info,
            "url": f"{name}/index.html",
        })

    # -- Render home page --
    home_tpl = env.get_template("home.html")
    home_html = home_tpl.render(
        studio_name=config.STUDIO_NAME,
        dancers=dancer_contexts,
    )
    (output_dir / "index.html").write_text(home_html, encoding="utf-8")
    print(f"  Wrote {output_dir / 'index.html'}")

    # -- Render per-dancer pages --
    dancer_tpl = env.get_template("dancer.html")
    for ctx in dancer_contexts:
        dancer_dir = output_dir / ctx["name"]
        dancer_dir.mkdir(parents=True, exist_ok=True)
        html = dancer_tpl.render(
            studio_name=config.STUDIO_NAME,
            dancer=ctx,
        )
        out_path = dancer_dir / "index.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"  Wrote {out_path}")


def _copy_full_image(src: Path, output_dir: Path, dancer_name: str) -> str:
    """Copy (or symlink) the full-size image into the output tree.

    Returns the path relative to *output_dir*.
    """
    dest_dir = output_dir / "photos" / dancer_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dest)
    return f"photos/{dancer_name}/{src.name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_gallery() -> None:
    """Entry point: scan dancers, generate thumbnails, render gallery."""
    print(f"Ballet Gallery Generator")
    print(f"  Studio: {config.STUDIO_NAME}")
    print(f"  Source: {config.ORGANIZED_DIR}")
    print(f"  Output: {config.GALLERY_OUTPUT_DIR}")
    print()

    # Collect dancer data
    dancers = _collect_dancers(config.ORGANIZED_DIR)
    if not dancers:
        print("No dancer folders with images found. Nothing to generate.")
        sys.exit(0)

    total_photos = sum(len(d["photos"]) for d in dancers)
    print(f"Found {len(dancers)} dancer(s) with {total_photos} total photo(s).\n")

    output_dir = Path(config.GALLERY_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate thumbnails
    thumb_map = generate_thumbnails(
        dancers,
        output_dir,
        config.GALLERY_THUMBNAIL_WIDTH,
    )

    # Render HTML
    print("\nRendering gallery pages ...")
    render_gallery(dancers, thumb_map, output_dir)

    print(f"\nGallery built successfully!")
    print(f"Open {output_dir / 'index.html'} in a browser to view.")


if __name__ == "__main__":
    build_gallery()
