"""Database initialization and helpers for the ballet photo organizer."""

import sqlite3
import config

def get_connection():
    """Return a new SQLite connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            processed_at TIMESTAMP,
            face_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            image_id INTEGER REFERENCES images(id),
            embedding BLOB NOT NULL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            detection_score REAL,
            cluster_id INTEGER,
            dancer_name TEXT
        );

        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY,
            dancer_name TEXT,
            face_count INTEGER,
            representative_face_id INTEGER REFERENCES faces(id)
        );

        CREATE INDEX IF NOT EXISTS idx_images_processed ON images(processed_at);
        CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
        CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {config.DB_PATH}")
