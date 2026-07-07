"""
recognition_engine.py
----------------------
The matching brain of the system. Loads the full embedding "gallery"
(every stored image's embedding, per registered plant) from the
database and compares a new candidate embedding against it using
cosine similarity -- exactly the same core mechanic used by
embedding-based face recognition systems.

Key guarantee from the project spec: an unknown plant must NEVER be
silently assigned to the closest registered plant. We enforce this by
hard-thresholding: only similarities >= SIMILARITY_THRESHOLD are
allowed to count as a match. Below that, regardless of how "close" the
nearest neighbor is, the result is reported as Unknown Plant.
"""

import numpy as np

import config
from database import PlantDatabase


class RecognitionResult:
    def __init__(self, name, confidence, is_known):
        self.name = name                  # str or None
        self.confidence = confidence       # float 0..100 (percentage)
        self.is_known = is_known           # bool

    def __repr__(self):
        label = self.name if self.is_known else "Unknown Plant"
        return f"<RecognitionResult {label} ({self.confidence:.1f}%)>"


class RecognitionEngine:
    def __init__(self, db: PlantDatabase, threshold: float = config.SIMILARITY_THRESHOLD):
        self.db = db
        self.threshold = threshold
        self.gallery_names = []
        self.gallery_vectors = np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        self.refresh_gallery()

    def refresh_gallery(self):
        """
        Reload the full embedding gallery from the database. Call this
        after any registration or deletion so recognition immediately
        reflects the current registered-plant set, without needing to
        restart the app. New plants are usable right away; deleted
        plants stop matching right away.
        """
        self.gallery_names, self.gallery_vectors = self.db.get_all_embeddings()

    def set_threshold(self, threshold: float):
        self.threshold = float(threshold)

    def recognize(self, embedding: np.ndarray) -> RecognitionResult:
        """
        Compare a query embedding against the gallery.

        Matching strategy (k-NN, like a face-recognition gallery search):
        1. Compute cosine similarity against every stored embedding.
        2. Take the TOP_K_MATCHES highest similarities.
        3. If they don't all belong to the same plant, fall back to just
           the single best match (avoids averaging across two different
           plants that happen to look similar).
        4. Average those top-k similarities -> confidence score.
        5. Apply the hard threshold: below threshold => Unknown Plant,
           no matter how close the nearest neighbor was.
        """
        if self.gallery_vectors.shape[0] == 0:
            return RecognitionResult(None, 0.0, False)

        # Embeddings are L2-normalized, so cosine similarity is a dot product.
        similarities = self.gallery_vectors @ embedding  # shape (N,)

        k = min(config.TOP_K_MATCHES, similarities.shape[0])
        top_k_idx = np.argpartition(-similarities, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]  # sort descending

        best_idx = top_k_idx[0]
        best_name = self.gallery_names[best_idx]
        best_sim = float(similarities[best_idx])

        # Only average top-k scores that agree with the best match's identity;
        # this keeps the confidence score meaningful for a single plant.
        agreeing = [similarities[i] for i in top_k_idx if self.gallery_names[i] == best_name]
        score = float(np.mean(agreeing))

        confidence_pct = max(0.0, min(100.0, score * 100.0))

        if score >= self.threshold:
            return RecognitionResult(best_name, confidence_pct, True)
        else:
            # Hard rule from the spec: never assign an unknown plant to
            # the closest registered plant. Report the best score for
            # transparency, but label it Unknown.
            return RecognitionResult(None, confidence_pct, False)
