"""OpenRouter embedding client: batching, retries, cosine normalisation."""
import logging
import time

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class EmbeddingClient:

    def __init__(self,api_key: str,model: str, batch_size: int = 100,max_retries: int = 3):
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL,api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed_texts(self, texts: list[str]) -> np.array:
        """(n,dim) float32 array, L2-normalised so IndexFlatIP == cosine similarity."""
        if not texts:
            raise ValueError("embed_texts called with an empty list")
        vectors = []
        for start in range(0,len(texts),self.batch_size):
            batch = texts[start:start + self.batch_size]
            data = self._call_with_retry(batch)
            by_index = {d.index: d.embedding for d in data} # guard against reordering
            vectors.extend(by_index[i] for i in range(len(batch)))
        arr = np.asarray(vectors,dtype="float32")
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
        norms = np.linalg.norm(arr,axis=1,keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self,text:str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def _call_with_retry(self,batch: list[str]):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.client.embeddings.create(model=self.model,input=batch).data
            except Exception as exc: #noqa: BLE001 - surfaced with context below
                last_error = exc
                wait = 2 ** attempt
                logger.warning("Embedding attempt %d failed (%s); retrying in %ss",attempt + 1,exc,wait)
                time.sleep(wait)
        raise RuntimeError(f"Embedding failed after {self.max_retries} attempts: {last_error}") from last_error