# Ballet Photo Organizer — Face Clustering for 500K Images

## Context
Billy has 500,000 ballet photos from his daughter's dance studio. He needs a tool to automatically organize/categorize them by dancer. Photos include performances (stage lighting, costumes, makeup), rehearsals, and group shots. Manual sorting is impossible at this scale.

## Approach
Detect faces in every photo, extract a numeric "fingerprint" (embedding) per face, then cluster similar faces together. Each cluster = one dancer. Billy labels clusters with names, then photos are organized into folders.

## Tech Stack
- **Python 3.12** (already installed via Homebrew)
- **InsightFace** (buffalo_l model) — face detection + 512-dim embeddings
- **FAISS** (Facebook) — fast approximate nearest neighbor search
- **Chinese Whispers** (via networkx) — clustering algorithm
- **SQLite** — progress tracking + metadata storage
- **Flask** — simple web UI for labeling clusters
- **Pillow** — image loading/thumbnails

### Why InsightFace over face_recognition (dlib)?
- 512-dim embeddings vs 128-dim = much better accuracy
- Handles stage lighting, costumes, non-frontal poses better (critical for ballet)
- ArcFace model achieves ~90% accuracy vs dlib's ~57% on challenging datasets

## Cost Breakdown

| Item | Cost | Notes |
|---|---|---|
| InsightFace | $0 | Open source (MIT license) |
| FAISS | $0 | Open source (MIT license) |
| Python + all libraries | $0 | All open source |
| SQLite | $0 | Built into Python |
| Storage | $0 | Runs entirely local, photos stay on disk |
| Cloud/API fees | $0 | No cloud services, no API calls |
| Hardware | $0 | Runs on existing i9 MacBook |
| **Total** | **$0** | Everything is free and runs locally |

## Timeline

| Phase | What | Time to Build | Processing Time |
|---|---|---|---|
| 1. Core pipeline | Face detection + embedding extraction script | 1 day coding | 8-14 hrs overnight run |
| 2. Clustering | FAISS index + Chinese Whispers grouping | 0.5 day coding | ~5 minutes |
| 3. Label UI | Web interface to name clusters, review faces | 1 day coding | — |
| 4. Organize | Copy/move/symlink photos into dancer folders | 0.5 day coding | ~30 minutes |
| 5. Polish | Error handling, progress bars, resume support | 0.5 day coding | — |
| 6. Gallery site | Static photo gallery for sharing with parents | 1 day coding | — |
| **Total** | | **4-5 days coding** | **~14 hrs processing** |

### Processing Time Estimate (500K photos on 8-core i9)
- InsightFace: ~150-250ms per image per core
- 8 worker processes in parallel: ~20-30ms effective per image
- 500,000 images x 25ms = ~3.5 hours (optimistic) to ~14 hours (conservative)
- **Plan: kick off overnight, done by morning**

## Database Schema (SQLite)

### `images` — Track every photo file
```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    processed_at TIMESTAMP,
    face_count INTEGER DEFAULT 0
);
```

### `faces` — Every detected face
```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY,
    image_id INTEGER REFERENCES images(id),
    embedding BLOB NOT NULL,          -- 512-dim float32 = 2KB per face
    bbox_x1 REAL, bbox_y1 REAL,       -- bounding box (normalized 0-1)
    bbox_x2 REAL, bbox_y2 REAL,
    detection_score REAL,             -- confidence 0-1
    cluster_id INTEGER,               -- assigned after clustering
    dancer_name TEXT                   -- assigned by user in label UI
);
```

### `clusters` — Grouped faces
```sql
CREATE TABLE clusters (
    id INTEGER PRIMARY KEY,
    dancer_name TEXT,
    face_count INTEGER,
    representative_face_id INTEGER REFERENCES faces(id)
);
```

## Project Structure
```
/Users/billylee/Projects/ballet-photo-organizer/
├── requirements.txt
├── config.py              # Photo directory path, DB path, model settings
├── scanner.py             # Walk photo directory, populate images table
├── detector.py            # Face detection + embedding extraction (multiprocessing)
├── cluster.py             # FAISS indexing + Chinese Whispers clustering
├── organizer.py           # Copy/symlink photos into dancer folders
├── app.py                 # Flask web UI for labeling clusters
├── templates/
│   ├── index.html         # Dashboard: all clusters overview
│   ├── cluster.html       # Single cluster: face grid + name input
│   └── image.html         # Single image: all faces highlighted
├── gallery.py             # Generate static photo gallery website
├── templates/
│   ├── gallery/
│   │   ├── home.html      # Dancer directory page
│   │   └── dancer.html    # Per-dancer photo grid + lightbox
│   ├── index.html         # Dashboard: all clusters overview
│   ├── cluster.html       # Single cluster: face grid + name input
│   └── image.html         # Single image: all faces highlighted
├── static/
│   └── style.css
├── gallery_output/        # Generated static site (deploy this)
└── ballet_photos.db       # SQLite database (auto-created)
```

## Implementation Steps

### Step 1: Project Setup + Scanner
- Create project directory, venv, install dependencies
- `requirements.txt`: insightface, onnxruntime, faiss-cpu, networkx, flask, pillow, tqdm
- `config.py`: photo directory path, supported extensions (.jpg, .jpeg, .png, .heic)
- `scanner.py`: Walk the photo directory recursively, insert file paths into `images` table
- Skip already-scanned files on re-run (pause/resume safe)

### Step 2: Face Detection + Embedding Pipeline
- `detector.py`: The heavy lifter
- Load InsightFace `buffalo_l` model with `allowed_modules=['detection', 'recognition']`
- Use Python `multiprocessing.Pool` with 8 workers
- Each worker: load image → detect faces → extract embeddings → write to DB
- Progress bar with tqdm showing images/sec and ETA
- Graceful shutdown on Ctrl+C (finish current image, save progress)
- Resume: skip images where `processed_at IS NOT NULL`
- Handle errors gracefully (corrupt files, no faces found, etc.)

### Step 3: Clustering
- `cluster.py`:
- Load all embeddings from DB into numpy array (~1-3 GB RAM for 500K faces)
- Build FAISS index (IndexFlatL2 or IndexIVFFlat for speed)
- For each face, find k=10 nearest neighbors
- Build sparse graph: edge between faces if distance < threshold (0.6 for InsightFace 512-d)
- Run Chinese Whispers on the graph (networkx implementation)
- Write cluster_id back to faces table
- Create clusters table with representative face (highest detection score per cluster)
- Two-pass approach (Apple's method):
  - Pass 1: tight threshold (0.4) for high-precision clusters
  - Pass 2: relax threshold (0.7) to merge similar clusters

### Step 4: Labeling Web UI
- `app.py`: Flask server on localhost:5000
- **Dashboard** (`/`): Grid of clusters sorted by size (most faces first)
  - Each cluster shows: 6 sample face thumbnails + face count
  - Click cluster → detail view
  - Search/filter by name (after labeling)
- **Cluster detail** (`/cluster/<id>`):
  - Grid of all face thumbnails in the cluster
  - Text input to name the dancer
  - "Merge with..." dropdown to combine clusters (same person, different angle/costume)
  - "Split" to separate wrongly merged faces
  - Mark faces as "not this person" → moves to unclustered
- **Image view** (`/image/<id>`):
  - Full image with face bounding boxes drawn
  - Each box labeled with dancer name or cluster ID
- Thumbnails generated on-the-fly: crop face from original image, resize to 150x150

### Step 5: Photo Organizer
- `organizer.py`: After labeling, organize photos
- Mode 1: **Symlinks** (recommended) — create `organized/<dancer_name>/` folders with symlinks to originals
- Mode 2: **Copy** — duplicate files into dancer folders
- Mode 3: **CSV export** — just output a spreadsheet: filename, dancer_name(s), cluster_id
- Handle multi-face images: photo appears in multiple dancer folders
- Summary report: X dancers identified, Y photos organized, Z unidentified

## Key Design Decisions

### Why Chinese Whispers over DBSCAN/HDBSCAN?
- Best accuracy reported for face clustering tasks
- Minimal tuning (just one distance threshold)
- HDBSCAN needs 4-10 GB RAM at 500K scale and requires PCA first
- DBSCAN is sensitive to eps parameter

### Why FAISS for neighbor search?
- Computing all 500K x 500K pairwise distances = 931 GB (impossible)
- FAISS finds k-nearest neighbors efficiently in ~seconds
- Builds the sparse graph needed for Chinese Whispers

### Why SQLite over PostgreSQL?
- Zero setup, single file, portable
- Sufficient for this workload (reads/writes are sequential, not concurrent)
- Easy to backup (copy one file)

### Ballet-Specific Considerations
- Stage lighting (colored gels, spotlights) changes face appearance — InsightFace handles this
- Costumes + makeup can fool simpler models — 512-d embeddings capture enough detail
- Group shots (corps de ballet) = many faces per image — pipeline handles multi-face
- Dancers at different ages across years — may create separate clusters per "era"
- Two-pass clustering helps avoid merging different dancers who look similar in makeup

## Memory Requirements
- InsightFace model per worker: ~500MB
- 8 workers: ~4 GB for models
- 500K embeddings (512-d float32): ~1 GB
- FAISS index: ~1 GB
- **Total peak: ~6-8 GB RAM** — well within an i9 MacBook's capacity

## Verification
1. Run scanner → DB shows correct image count
2. Run detector on 100 test images → faces detected, embeddings stored
3. Run full detector overnight → all 500K processed, progress bar shows completion
4. Run clustering → clusters created, largest clusters are clearly individual dancers
5. Open web UI → browse clusters, label names, merge/split works
6. Run organizer → folders created with correct photos per dancer
7. Spot-check: open random dancer folder, verify all photos show the same person

## Step 6: Static Photo Gallery Website (Sharing)

### What It Does
After photos are organized by dancer, generate a clean, password-protected photo gallery website. Each dancer gets their own page. Parents receive a link to view/download their kid's photos.

### Tech
- **Static site generator**: Jinja2 templates → plain HTML/CSS/JS (no framework needed)
- **Hosting**: Vercel or Netlify (free tier — unlimited bandwidth, plenty for photos)
- **Image hosting**: Thumbnails embedded in site, full-res served from Cloudflare R2 or Vercel Blob
- **Password protection**: Simple per-dancer password gate (JS-based or Netlify/Vercel auth)

### Cost
| Item | Cost |
|---|---|
| Vercel/Netlify hosting | $0 (free tier) |
| Cloudflare R2 storage (10GB free) | $0 for first ~20K photos |
| Custom domain (optional) | ~$12/year |
| **Total** | **$0 - $12/year** |

### Features
- **Home page**: Studio name/logo, list of dancers (password-gated)
- **Dancer page**: Thumbnail grid of all their photos, sorted by date
- **Lightbox**: Click photo → full-screen view with download button
- **Bulk download**: "Download All" button → ZIP of dancer's photos
- **Mobile-friendly**: Responsive grid, works on phones
- **Auto-generated**: One command rebuilds the entire site after new photos are organized

### How It Works
```
organizer.py outputs folders → gallery.py reads folders → generates HTML per dancer → deploy to Vercel
```

### Workflow for Parents
1. Studio sends parent a link: `photos.yourstudio.com/dancers/emma`
2. Parent enters simple password (dancer's last name, or studio-assigned code)
3. Browse photos, tap to view full-size, download favorites or all

### Implementation
- `gallery.py`: New script added to project
  - Reads organized dancer folders
  - Generates thumbnails (300px wide) for fast loading
  - Renders Jinja2 templates → `gallery_output/` directory
  - Each dancer gets `gallery_output/<dancer_name>/index.html`
- `templates/gallery/`: HTML templates
  - `home.html` — dancer directory
  - `dancer.html` — photo grid with lightbox
- One-command deploy: `vercel gallery_output/` or `netlify deploy`

## Future Enhancements (not in MVP)
- Auto-label using a few tagged reference photos per dancer
- Search by dancer name across all photos
- Generate highlight reels (best photos per dancer per event)
- Export contact sheets (PDF grid of a dancer's best photos)
- HEIC support for iPhone photos
- Duplicate photo detection
