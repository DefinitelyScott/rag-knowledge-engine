import pytest

from ragkb.text import chunk_document, split_sentences, tokenize


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("BM25, TF-IDF; and RRF!") == ["bm25", "tf-idf", "and", "rrf"]


def test_tokenize_can_drop_stopwords():
    assert tokenize("the score of the document", remove_stopwords=True) == [
        "score",
        "document",
    ]


def test_split_sentences_handles_abbreviation_free_text():
    text = "First sentence. Second one? Third one!\n\nA new paragraph."
    assert split_sentences(text) == [
        "First sentence.",
        "Second one?",
        "Third one!",
        "A new paragraph.",
    ]


def test_chunks_never_split_a_sentence():
    text = " ".join("Sentence number {} carries a few tokens.".format(i) for i in range(30))
    chunks = chunk_document("doc", text, target_tokens=30, overlap_tokens=8)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.endswith(".")
        for sentence in split_sentences(chunk.text):
            assert sentence in text


def test_chunks_overlap_and_cover_every_sentence():
    text = " ".join("Fact {} is recorded here plainly.".format(i) for i in range(20))
    chunks = chunk_document("doc", text, target_tokens=25, overlap_tokens=8)
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.sentence_start, chunk.sentence_end + 1))
    assert covered == set(range(len(split_sentences(text))))
    assert any(
        later.sentence_start <= earlier.sentence_end
        for earlier, later in zip(chunks, chunks[1:])
    )


def test_chunk_ids_are_unique_and_namespaced():
    text = " ".join("Token filler sentence {} here.".format(i) for i in range(12))
    chunks = chunk_document("mydoc", text, target_tokens=20, overlap_tokens=5)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("mydoc::") for cid in ids)


def test_oversized_sentence_becomes_its_own_chunk():
    long_sentence = " ".join("word{}".format(i) for i in range(60)) + "."
    text = "Short one. {} Another short one.".format(long_sentence)
    chunks = chunk_document("doc", text, target_tokens=20, overlap_tokens=5)
    assert any(chunk.token_count > 20 for chunk in chunks)


def test_empty_document_yields_no_chunks():
    assert chunk_document("doc", "   \n  ") == []


@pytest.mark.parametrize("target,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_parameters_raise(target, overlap):
    with pytest.raises(ValueError):
        chunk_document("doc", "Some text here.", target_tokens=target, overlap_tokens=overlap)
