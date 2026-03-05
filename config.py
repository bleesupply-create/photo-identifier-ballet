"""Shared configuration for the ballet photo organizer."""

import os

# --- Paths ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.environ.get("BALLET_PHOTO_DIR", os.path.expanduser("~/Pictures/ballet"))
DB_PATH = os.path.join(PROJECT_DIR, "ballet_photos.db")
ORGANIZED_DIR = os.path.join(PROJECT_DIR, "organized")
GALLERY_OUTPUT_DIR = os.path.join(PROJECT_DIR, "gallery_output")
THUMBNAIL_DIR = os.path.join(PROJECT_DIR, "thumbnails")

# --- Scanner ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".bmp", ".webp"}

# --- Detector ---
MODEL_NAME = "buffalo_l"
NUM_WORKERS = int(os.environ.get("BALLET_NUM_WORKERS", min(8, os.cpu_count() or 4)))
BATCH_SIZE = 32  # images per DB commit
DET_SIZE = (640, 640)  # InsightFace detection input size

# --- Clustering ---
EMBEDDING_DIM = 512
KNN_K = 10  # k-nearest neighbors for graph construction
CLUSTER_THRESHOLD_TIGHT = 0.4  # Pass 1: high precision
CLUSTER_THRESHOLD_LOOSE = 0.7  # Pass 2: merge similar clusters
FAISS_NPROBE = 10  # number of cells to search in IVF index

# --- Flask UI ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
THUMBNAIL_SIZE = (150, 150)
CLUSTER_PREVIEW_COUNT = 6  # sample faces shown per cluster on dashboard

# --- Gallery ---
GALLERY_THUMBNAIL_WIDTH = 300
STUDIO_NAME = os.environ.get("BALLET_STUDIO_NAME", "Dance Studio")
