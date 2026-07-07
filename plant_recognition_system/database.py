"""
database.py
-----------
SQLite persistence layer for the Plant Recognition System.

Schema:
    plants
        id              INTEGER PRIMARY KEY
        name            TEXT UNIQUE NOT NULL
        registered_at   TEXT (ISO timestamp)
        num_images      INTEGER

    embeddings
        id          INTEGER PRIMARY KEY
        plant_id    INTEGER (FK -> plants.id)
        image_path  TEXT
        vector      BLOB (float32 numpy array, serialized)

Each registered plant can have many embeddings (one per captured
image), mirroring how a face-recognition gallery stores several
photos per identity for more robust matching.
"""

import sqlite3
import os
import datetime
import numpy as np

import config


class PlantDatabase:
    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS plants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                registered_at TEXT NOT NULL,
                num_images INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Plant CRUD
    # ------------------------------------------------------------------
    def plant_exists(self, name: str) -> bool:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM plants WHERE name = ?", (name,))
        result = cur.fetchone()
        conn.close()
        return result is not None

    def add_plant(self, name: str) -> int:
        """Create a new plant record (without embeddings yet). Returns plant_id."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO plants (name, registered_at, num_images) VALUES (?, ?, 0)",
            (name, datetime.datetime.now().isoformat(timespec="seconds")),
        )
        plant_id = cur.lastrowid
        conn.commit()
        conn.close()
        return plant_id

    def add_embedding(self, plant_id: int, image_path: str, vector: np.ndarray):
        """Store one embedding (linked to one captured image) for a plant."""
        conn = self._connect()
        cur = conn.cursor()
        vector_blob = vector.astype(np.float32).tobytes()
        cur.execute(
            "INSERT INTO embeddings (plant_id, image_path, vector) VALUES (?, ?, ?)",
            (plant_id, image_path, vector_blob),
        )
        cur.execute(
            "UPDATE plants SET num_images = num_images + 1 WHERE id = ?",
            (plant_id,),
        )
        conn.commit()
        conn.close()

    def get_all_plants(self):
        """Returns list of dicts: id, name, registered_at, num_images."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, registered_at, num_images FROM plants ORDER BY name"
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {"id": r[0], "name": r[1], "registered_at": r[2], "num_images": r[3]}
            for r in rows
        ]

    def delete_plant(self, plant_id: int):
        """Deletes a plant and all its embeddings (cascade)."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM embeddings WHERE plant_id = ?", (plant_id,))
        cur.execute("DELETE FROM plants WHERE id = ?", (plant_id,))
        conn.commit()
        conn.close()

    def delete_all_plants(self):
        """Deletes every plant and every embedding from the database."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM embeddings")
        cur.execute("DELETE FROM plants")
        conn.commit()
        conn.close()

    def get_plant_by_name(self, name: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, registered_at, num_images FROM plants WHERE name = ?",
            (name,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "registered_at": row[2],
            "num_images": row[3],
        }

    # ------------------------------------------------------------------
    # Embedding gallery retrieval (used by the recognition engine)
    # ------------------------------------------------------------------
    def get_all_embeddings(self):
        """
        Returns everything needed for matching in one shot:
            plant_names: list[str]            (len = N embeddings, plant name per row)
            vectors:     np.ndarray (N, D)     (stacked embeddings)
        Loading the whole gallery into memory once per recognition
        session keeps real-time matching fast (no per-frame DB hits).
        """
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT plants.name, embeddings.vector
            FROM embeddings
            JOIN plants ON plants.id = embeddings.plant_id
        """)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return [], np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

        names = []
        vectors = []
        for name, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            names.append(name)
            vectors.append(vec)

        return names, np.vstack(vectors)
