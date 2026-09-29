from src.chunking import chunk_page_text, chunk_document


def test_empty_text_returns_no_chunks():
    assert chunk_page_text("") == []


def test_short_text_single_chunk():
    chunks = chunk_page_text("Hello world. This is short.")
    assert len(chunks) == 1
    assert "Hello world." in chunks[0]


def test_long_text_splits_with_overlap():
    text = " ".join(f"Sentence number {i} about topic {i % 7}." for i in range(400))
    chunks = chunk_page_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 * 4 + 200 for c in chunks)


def test_chunk_document_tracks_page_numbers():
    pages = [{"page": 1, "text": "First page content here."},
             {"page": 2, "text": "Second page content here."}]
    chunks = chunk_document(pages)
    assert [c["page"] for c in chunks] == [1, 2]
    assert [c["chunk_index"] for c in chunks] == [0, 1]