"""Golden-question evaluation: measures how often answers contain expected facts.

Usage:
    python scripts/eval.py path/to/document.pdf path/to/questions.json

questions.json format:
[
  {"question": "What is the refund policy?", "must_contain": ["30 days"]},
  {"question": "Who is the CTO?", "must_contain": ["Alice"]}
  {"question": "What is the company's secret plan for Mars colonization?", "expect_absent":true }
]
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document
from src.config import Settings
from src.embeddings import EmbeddingClient
from src.ingestion import extract_pdf
from src.llm import LLMClient
from src.metadata_store import MetadataStore
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retriever import Retriever
from src.vector_store import VectorStore


def main(pdf_path: str, questions_path: str):
    settings = Settings()
    if not settings.api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set. Add it to .env first.")
    
    settings.storage_dir = "storage_eval"  # keep eval data separate from the app
    Path(settings.storage_dir).mkdir(exist_ok=True)

    embeddings = EmbeddingClient(settings.api_key, settings.embedding_model, settings.embed_batch_size)
    vector_store = VectorStore(settings.index_path)
    index_loaded =vector_store.load_or_create()
    metadata = MetadataStore(settings.db_path)
    if not index_loaded:    #to stop the ignoring of corrupted/missing FAISS index state
        metadata.clear_all()
    retriever = Retriever(settings, embeddings, vector_store, metadata)
    llm = LLMClient(settings)

    digest = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    if metadata.find_document_by_hash(digest):
        print("Already indexed; reusing existing index.")
    else:
        print(f"Indexing {pdf_path} ...")
        pages, total = extract_pdf(pdf_path)
        chunks = chunk_document(pages, settings.chunk_size, settings.chunk_overlap)
        vectors = embeddings.embed_texts([c["text"] for c in chunks])
        doc_id = metadata.add_document(Path(pdf_path).name, digest, total)
        metadata.add_chunks([(doc_id, c["chunk_index"], c["text"], c["page"]) for c in chunks])
        vector_store.add(vectors, metadata.get_chunk_ids_for_doc(doc_id))
        vector_store.save()

    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    passed = 0
    failed = 0
    for q in questions:
        question = q.get("question")

        if not question:
            print("SKIP: question entry has no 'question' field")
            continue

        retrieved = retriever.retrieve(question)
        if not retrieved:
            print(f"FAIL (no retrieval): {question}")
            failed +=1
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, retrieved)},
        ]
        try:
            answer = llm.chat(messages).lower()    # To stop LLM request fails after retries
        except Exception as exc:
            print(f"FAIL (LLM error): {q['question']} -> {exc}")
            failed +=1
            continue
        missing = [kw for kw in q.get("must_contain", []) if kw.lower() not in answer]
        if missing:
            print(f"FAIL (missing {missing}): {question}")
            failed +=1
        else:
            passed += 1
            print(f"PASS: {q['question']}")
    total_q = len(questions)
    print()
    print(f"\nScore: {passed}/{total_q} ({100 * passed / max(total_q, 1):.0f}%)")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])