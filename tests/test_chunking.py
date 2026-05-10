from graphrag_mvp.chunking import chunk_text


def test_chunk_text_overlap():
    text = "abcdefgh" * 10
    chunks = chunk_text("doc1", text, size=20, overlap=5)
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc1" for c in chunks)
    assert chunks[0].chunk_id.startswith("doc1__")


def test_single_small_chunk():
    chunks = chunk_text("x", "hi", size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hi"
