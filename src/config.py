import os

from dotenv import load_dotenv
load_dotenv()

class Settings:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY","")
        self.embedding_model = os.getenv("EMBEDDING_MODEL","openai/text-embedding-3-small")
        self.llm_model = os.getenv("LLM_MODEL","gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE","0.2"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE","600"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP","90"))
        self.top_k = int(os.getenv("TOP_K","20"))
        self.rerank_top_n = int(os.getenv("RERANK_TOP_N","5"))
        self.use_rerank = os.getenv("USE_RERANK","true").lower() in ("1","true","yes")
        self.rerank_model = os.getenv("RERANK_MODEL","cohere/rerank-v3.5")
        self.storage_dir = os.getenv("STORAGE_DIR","storage")
        self.embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE","100"))
        self.max_retries = 3

    @property
    def index_path(self):
        return os.path.join(self.storage_dir,"index.faiss")

    @property
    def db_path(self):
        return os.path.join(self.storage_dir,"metadata.db")