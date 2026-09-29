"""FAISS wrapper. Vector ids == SQLite chunk ids (IndexIDMap), cosine via IndexFlatIP."""
import logging
import os
import time

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = None

    def load_or_create(self) -> bool:
        """Load the index from disk.

        Returns True if a valid index was loaded; False if the file was
        missing or corrupt (an empty in-memory index is used instead).
        """
        if not os.path.exists(self.index_path):
            return False
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read FAISS index (%s); quarantining it", exc)
            backup = f"{self.index_path}.corrupt-{int(time.time())}"
            try:
                os.replace(self.index_path, backup)
            except OSError:
                pass
            self.index = None
            return False

    @property
    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def add(self, vectors: np.ndarray, ids: list[int]):
        if len(vectors) == 0:
            return
        if self.index is None:  # lazily created once we know the dimension
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch ({vectors.shape[1]} vs {self.index.d}). "
                "Clear all data before switching embedding models."
            )
        self.index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))

    def search(self, query_vector: np.ndarray, k: int):
        """Returns (scores, chunk_ids). Empty arrays when nothing is indexed."""
        if self.index is None or self.index.ntotal == 0:
            return np.array([], dtype="float32"), np.array([], dtype="int64")
        scores, ids = self.index.search(np.asarray(query_vector, dtype="float32").reshape(1, -1), k)
        return scores[0], ids[0]

    def save(self):
        """Write atomically: temp file + rename, so a crash mid-save can
        never leave a half-written index at the real path."""
        if self.index is None:
            return
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        tmp_path = self.index_path + ".tmp"
        faiss.write_index(self.index, tmp_path)
        os.replace(tmp_path, self.index_path)
        logger.info("Saved FAISS index (%d vectors)", self.index.ntotal)

    def clear(self):
        self.index = None
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

    def remove_ids(self, ids: list[int]) -> int:
        
     """
        Remove vectors by SQLite chunk ids.
        Returns number of vectors removed.

     """
     if self.index is None or not ids:
        return 0

     ids_np = np.asarray(ids, dtype="int64")
     selector = faiss.IDSelectorBatch(ids_np)
     removed = self.index.remove_ids(selector)

     return int(removed)