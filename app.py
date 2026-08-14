"""RAG Document Q&A - Streamlit UI."""

import hashlib
import os
import tempfile
from collections import defaultdict

import numpy as np
import streamlit as st

from src.chunking import chunk_document
from src.config import Settings
from src.embeddings import EmbeddingClient
from src.ingestion import extract_pdf
from src.llm import LLMClient
from src.metadata_store import MetadataStore
from src.prompts import build_llm_messages
from src.retriever import Retriever
from src.vector_store import VectorStore


st.set_page_config(
    page_title="RAG Document Q&A",
    layout="wide",
)


@st.cache_resource
def get_components():
    settings = Settings()
    os.makedirs(settings.storage_dir, exist_ok=True)

    embeddings = EmbeddingClient(
        settings.api_key,
        settings.embedding_model,
        settings.embed_batch_size,
    )

    vector_store = VectorStore(settings.index_path)
    index_loaded = vector_store.load_or_create()

    metadata = MetadataStore(settings.db_path)

    if not index_loaded:
        # Index is missing or unreadable: wipe the DB so the two stores can
        # never disagree about what is indexed.
        metadata.clear_all()

    retriever = Retriever(settings, embeddings, vector_store, metadata)
    llm = LLMClient(settings)

    return settings, embeddings, vector_store, metadata, retriever, llm


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def index_uploaded_file(uploaded_file) -> bool:
    """
    Index a PDF.

    Returns:
        True if newly indexed.
        False if duplicate or failed.
    """
    settings, embeddings, vector_store, metadata, *_ = get_components()

    data = uploaded_file.getvalue()
    digest = file_hash(data)

    if metadata.find_document_by_hash(digest):
        return False

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with st.status("Indexing document...", expanded=True) as status:
            st.write("Extracting text...")
            pages, total_pages = extract_pdf(tmp_path)

            st.write("Splitting into chunks...")
            chunks = chunk_document(
                pages,
                settings.chunk_size,
                settings.chunk_overlap,
            )

            st.write(f"Embedding {len(chunks)} chunks...")
            progress = st.progress(0.0)

            batches = []

            for i in range(0, len(chunks), settings.embed_batch_size):
                batch = chunks[i:i + settings.embed_batch_size]
                batch_vectors = embeddings.embed_texts(
                    [c["text"] for c in batch]
                )
                batches.append(batch_vectors)

                progress.progress(
                    min(
                        (i + settings.embed_batch_size) / max(len(chunks), 1),
                        1.0,
                    )
                )

            all_vectors = np.vstack(batches)

            st.write("Saving to vector store...")

            doc_id = metadata.add_document(
                uploaded_file.name,
                digest,
                total_pages,
            )

            metadata.add_chunks(
                [
                    (
                        doc_id,
                        c["chunk_index"],
                        c["text"],
                        c["page"],
                    )
                    for c in chunks
                ]
            )

            vector_store.add(
                all_vectors,
                metadata.get_chunk_ids_for_doc(doc_id),
            )

            vector_store.save()

            status.update(
                label="Indexing complete",
                state="complete",
            )

        st.success(
            f"Indexed '{uploaded_file.name}' "
            f"({total_pages} pages, {len(chunks)} chunks)."
        )

        return True

    except Exception as exc:
        st.error(f"Failed to index '{uploaded_file.name}': {exc}")
        return False

    finally:
        os.unlink(tmp_path)


def rewrite_search_query(
    prompt: str,
    history: list[dict],
    llm: LLMClient,
) -> str:
    """
    Rewrite a follow-up question into a standalone retrieval query.

    Example:
        history:
            user: What is the revenue?
            assistant: The revenue is $5.2 billion.
        prompt:
            How does that compare to last year?

        rewritten:
            How does the current revenue compare to last year's revenue?
    """
    recent = [
        msg
        for msg in history[-4:]
        if msg.get("role") in ("user", "assistant")
    ]

    if not recent:
        return prompt

    lines = []

    for msg in recent:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()

        if not content:
            continue

        if len(content) > 500:
            content = content[:500] + "... [truncated]"

        lines.append(f"{role}: {content}")

    if not lines:
        return prompt

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the follow-up question into a standalone search query. "
                "Keep it short and precise. "
                "Do not answer the question. "
                "Output only the rewritten query."
            ),
        },
        {
            "role": "user",
            "content": (
                "Conversation so far:\n"
                + "\n".join(lines)
                + "\n\n"
                + f"Follow-up question: {prompt}\n\n"
                + "Standalone search query:"
            ),
        },
    ]

    try:
        rewritten = llm.chat(messages, timeout=20).strip()
        return rewritten or prompt
    except Exception:
        return prompt


def render_sources(sources: list[dict]):
    """
    Phase 7: Better source display.

    Groups sources by filename and shows:
    - page number
    - final score
    - retrieval score
    - rerank score if available
    - short text preview
    """
    if not sources:
        return

    grouped = defaultdict(list)

    for source in sources:
        filename = source.get("filename", "Unknown file")
        grouped[filename].append(source)

    with st.expander(f"Sources ({len(sources)} excerpts)"):
        for filename, items in grouped.items():
            st.markdown(f"**{filename}**")

            items = sorted(
                items,
                key=lambda x: x.get("score", 0.0),
                reverse=True,
            )

            for source in items:
                page = source.get("page", "?")
                score = source.get("score", 0.0)

                retrieval_score = source.get("retrieval_score")
                rerank_score = source.get("rerank_score")

                badges = []

                if rerank_score is not None:
                    badges.append(f"rerank `{rerank_score:.3f}`")

                if retrieval_score is not None:
                    badges.append(f"retrieval `{retrieval_score:.3f}`")

                badge_text = ""

                if badges:
                    badge_text = " · " + " · ".join(badges)

                st.markdown(
                    f"- page {page} · score `{score:.3f}`{badge_text}"
                )

                text = source.get("text", "")

                st.caption(
                    text[:300] + ("..." if len(text) > 300 else "")
                )

            st.divider()


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            render_sources(msg.get("sources") or [])


def render_document_manager(
    metadata: MetadataStore,
    vector_store: VectorStore,
):
    """
    Phase 6: Show indexed documents and delete individual documents.
    """
    docs = metadata.list_documents()

    if not docs:
        st.caption("No documents indexed yet.")
        return

    for doc in docs:
        col1, col2 = st.columns([4, 1])

        col1.markdown(f"**{doc['filename']}**")
        col1.caption(
            f"{doc.get('page_count') or 0} pages "
            f"· {doc.get('chunk_count') or 0} chunks"
        )

        if col2.button(
            "🗑️",
            key=f"delete_doc_{doc['id']}",
            help="Delete this document",
        ):
            chunk_ids = metadata.get_chunk_ids_for_doc(doc["id"])

            if chunk_ids:
                vector_store.remove_ids(chunk_ids)
                vector_store.save()

            metadata.delete_document(doc["id"])

            # Prevent duplicate-upload shortcut from blocking re-upload
            st.session_state.indexed_hashes.discard(
                doc.get("file_hash")
            )

            st.rerun()


def main():
    settings, embeddings, vector_store, metadata, retriever, llm = (
        get_components()
    )

    st.session_state.setdefault("indexed_hashes", set())
    st.session_state.setdefault("messages", [])

    st.title("RAG Document Q&A")

    if not settings.api_key:
        st.error(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
        st.stop()

    with st.sidebar:
        st.header("Documents")

        uploaded = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
        )

        if uploaded:
            digest = file_hash(uploaded.getvalue())

            if digest not in st.session_state.indexed_hashes:
                newly_indexed = index_uploaded_file(uploaded)

                if newly_indexed:
                    st.session_state.indexed_hashes.add(digest)
                    st.rerun()
                else:
                    st.warning("This document is already indexed.")
                    st.session_state.indexed_hashes.add(digest)

        st.divider()

        render_document_manager(metadata, vector_store)

        stats = metadata.stats()

        st.caption(
            f"Indexed documents: {stats['documents']} | "
            f"Chunks: {stats['chunks']}"
        )

        st.caption(f"Embedding: {settings.embedding_model}")
        st.caption(f"LLM: {settings.llm_model}")

        if st.button("Clear all data"):
            metadata.clear_all()
            vector_store.clear()

            st.session_state.messages = []
            st.session_state.indexed_hashes = set()

            st.rerun()

    st.subheader("Ask questions about your documents")

    for msg in st.session_state.messages:
        render_message(msg)

    if prompt := st.chat_input("Ask a question about the document"):
        # History before adding the current user message
        history = st.session_state.messages.copy()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        render_message(
            {
                "role": "user",
                "content": prompt,
            }
        )

        answer = ""
        sources = []

        with st.chat_message("assistant"):
            if vector_store.count == 0:
                answer = (
                    "No documents indexed yet. "
                    "Upload a PDF in the sidebar first."
                )
                st.markdown(answer)

            else:
                # Phase 2: rewrite follow-up questions for better retrieval
                search_query = rewrite_search_query(
                    prompt,
                    history,
                    llm,
                )

                with st.spinner("Searching the document..."):
                    chunks = retriever.retrieve(search_query)

                if not chunks:
                    answer = (
                        "No relevant content found in the indexed documents."
                    )
                    st.markdown(answer)

                else:
                    messages = build_llm_messages(
                        query=prompt,
                        context_blocks=chunks,
                        history=history,
                        max_history_messages=6,
                    )

                    response_box = st.empty()

                    try:
                        # Phase 3: streaming answer
                        for token in llm.chat_stream(messages):
                            answer += token
                            response_box.markdown(answer + "▌")

                        if not answer.strip():
                            answer = "The model returned an empty response."

                        response_box.markdown(answer)

                        sources = chunks

                        # Phase 7: better source display
                        render_sources(sources)

                    except Exception as exc:
                        answer = f"Sorry, I could not generate an answer: {exc}"
                        response_box.error(answer)
                        sources = []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()