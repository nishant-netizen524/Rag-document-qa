"""Sentence-aware chunking with overlap. Chunks never span pages."""
import re

def chunk_page_text(text:str,chunk_size:int = 600,overlap: int = 90) -> list[str]:
    """Split one page's text into ~chunk_size-token chunks with overlap.
    Tokens are approximated as 4 chars/token (good enough for sizing).
    """

    chunk_chars = max(chunk_size*4,200)
    overlap_chars = min(overlap*4,chunk_chars // 2)

    sentences = re.split(r"(?<=[.!?])\s+|\n+",text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks, current = [],""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= chunk_chars or not current:
            current = candidate
        else:
            chunks.append(current)
            #Overlap: start the next chunk with the tail of this one,
            # trimmed back to a sentence boundary when possible
            tail = current[-overlap_chars:]
            punct = [m.start() for m in re.finditer(r"[.!?]", tail)]
            if punct:
                tail = tail[punct[-1] + 1:].lstrip()
            current = (tail + " "+ sent).strip()
    if current:
        chunks.append(current)
    return chunks

def chunk_document(pages:list[dict],chunk_size: int = 600,overlap: int =90) -> list[dict]:
    """pages from ingestion.extract_pdf -> [{"chunk_index","page","text"}]."""
    chunks = []
    idx = 0
    for page in pages:
        for text in chunk_page_text(page["text"],chunk_size,overlap):
            chunks.append({"chunk_index":idx,"page": page["page"],"text":text})
            idx += 1
    return chunks
