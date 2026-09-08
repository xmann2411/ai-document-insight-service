from app.rag import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=900, overlap=150) == ["hello world"]


def test_long_text_is_split_with_overlap():
    words = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(words, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # reassembled chunks cover the whole input
    assert chunks[0].split()[0] == "word0"
    assert chunks[-1].split()[-1] == "word499"


def test_empty_text():
    assert chunk_text("   ", size=100, overlap=10) == []
