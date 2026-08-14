"""Retrieval pipeline: embed query -> FAISS top-k -> OpenRouter rerank (optional)."""
import logging

import requests

from src.config import Settings
from src.embeddings import EmbeddingClient
from src.metadata_store import MetadataStore
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

RERANK_URL = "https://openrouter.ai/api/v1/rerank"


class Retriever:
    def __init__(self, settings: Settings, embeddings: EmbeddingClient,
                 vector_store: VectorStore, metadata: MetadataStore):
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.metadata = metadata

    def retrieve(self, query: str) -> list[dict]:
        """Returns chunks: [{"chunk_id", "score", "text", "page", "filename"}]."""
        qvec = self.embeddings.embed_query(query).reshape(1, -1)
        scores, ids = self.vector_store.search(qvec, self.settings.top_k)

        candidates = []
        for chunk_id, score in zip(ids, scores):
            if chunk_id == -1:  # FAISS sentinel: no neighbour
                continue
            row = self.metadata.get_chunk_with_source(int(chunk_id))
            if row is None:
                continue  # stale vector whose metadata was deleted
            candidates.append({
                "chunk_id": int(chunk_id),
                "score": float(score),
                "retrieval_score": float(score),
                "text": row["text"],
                "page": row["page"],
                "filename": row["filename"],
            })

        if not candidates:
            return []

        if self.settings.use_rerank:
            try:
                return self._rerank(query, candidates)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash Q&A
                logger.warning("Rerank failed, using FAISS ranking: %s", exc)

        return candidates[: self.settings.rerank_top_n]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": [c["text"] for c in candidates],
            "top_n": self.settings.rerank_top_n,
        }
        resp = requests.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {self.settings.api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        ordered = []
        for r in sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True):
            idx = r.get("index")
            if idx is None or not (0 <= idx < len(candidates)):
                continue
            item = dict(candidates[idx])

            retrieval_score = item.get("score",0.0)
            rerank_score = float(r.get("relevance_score", retrieval_score))

            item["retrieval_score"] = retrieval_score
            item["rerank_score"] = rerank_score
            item["score"] = rerank_score
            ordered.append(item)
        return ordered[: self.settings.rerank_top_n] if ordered else candidates[: self.settings.rerank_top_n]