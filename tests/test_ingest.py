from src.ingest import _ENCODING, chunk_page_text


def test_short_page_is_single_chunk():
    text = "Liquid nitrogen is extremely cold and can cause frostbite."
    chunks = chunk_page_text(text, chunk_tokens=400, overlap_tokens=80)
    assert chunks == [text]


def test_empty_page_produces_no_chunks():
    assert chunk_page_text("") == []
    assert chunk_page_text("   ") == []


def test_long_page_is_split_with_overlap():
    # Build text long enough to need multiple 50-token chunks.
    words = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_page_text(words, chunk_tokens=50, overlap_tokens=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(_ENCODING.encode(chunk)) <= 50

    # Overlap: the tail of chunk[0] should reappear at the head of chunk[1].
    tail_tokens = _ENCODING.encode(chunks[0])[-10:]
    head_tokens = _ENCODING.encode(chunks[1])[:10]
    assert tail_tokens == head_tokens


def test_chunks_cover_full_text_without_gaps():
    words = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_page_text(words, chunk_tokens=50, overlap_tokens=10)
    rebuilt = _ENCODING.decode(_ENCODING.encode(chunks[0]))
    assert rebuilt.startswith("word0")
    assert "word399" in chunks[-1]
