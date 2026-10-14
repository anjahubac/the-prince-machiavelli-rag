import pytest

from src.chunking import build_chunks, split_text_by_sentence, split_text_with_overlap
from src.corpus import Document, load_default_corpus


def test_split_text_with_overlap_returns_multiple_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = split_text_with_overlap(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 3


def test_split_text_with_overlap_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text_with_overlap("abc", chunk_size=10, overlap=10)


def test_overlap_and_reconstruction_preserve_normalized_text() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = split_text_with_overlap(text, chunk_size=10, overlap=3)
    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]
    assert chunks[0] + "".join(chunk[3:] for chunk in chunks[1:]) == text
    assert split_text_with_overlap("  alpha\n beta\t gamma  ", 30, 0) == ["alpha beta gamma"]


def test_chunk_ids_and_metadata_are_traceable_to_documents() -> None:
    documents = [Document("one", "First", "abcdefghijkl"), Document("two", "Second", "uvwxyz")]
    chunks = build_chunks(documents, chunk_size=8, overlap=2)
    assert [chunk.chunk_id for chunk in chunks] == ["one-chunk-0", "one-chunk-1", "two-chunk-0"]
    assert [(chunk.doc_id, chunk.title) for chunk in chunks] == [
        ("one", "First"), ("one", "First"), ("two", "Second")
    ]


def test_sentence_strategy_keeps_whole_sentences() -> None:
    text = "Chunking splits text. Overlap keeps boundaries. Metadata helps traceability."
    chunks = split_text_by_sentence(text, chunk_size=30, sentence_overlap=0)
    assert chunks == ["Chunking splits text.", "Overlap keeps boundaries.", "Metadata helps traceability."]


def test_sentence_strategy_groups_short_sentences_up_to_chunk_size() -> None:
    chunks = split_text_by_sentence("One two. Three four five. Six.", chunk_size=30, sentence_overlap=0)
    assert chunks == ["One two. Three four five. Six."]
    assert split_text_by_sentence("One two. Three four five. Six.", chunk_size=25) == [
        "One two. Three four five.", "Six."
    ]


def test_sentence_overlap_repeats_previous_sentence_when_it_fits() -> None:
    chunks = split_text_by_sentence("One two. Three four five. Six.", chunk_size=25, sentence_overlap=1)
    assert chunks == ["One two. Three four five.", "Three four five. Six."]
    # The carried sentence is dropped when it would leave no room for the next one.
    assert split_text_by_sentence("One two. Three four five. Six.", chunk_size=20, sentence_overlap=1) == [
        "One two.", "Three four five.", "Six."
    ]


def test_sentence_longer_than_chunk_size_splits_at_word_boundaries() -> None:
    chunks = split_text_by_sentence("Short. One two three four five six. End.", chunk_size=15)
    assert chunks == ["Short.", "One two three", "four five six.", "End."]


def test_word_longer_than_chunk_size_falls_back_to_characters() -> None:
    chunks = split_text_by_sentence("Short. " + "x" * 25 + ". End.", chunk_size=10)
    assert chunks == ["Short.", "x" * 10, "x" * 10, "xxxxx.", "End."]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_sentence_strategy_never_cuts_words_in_the_default_corpus() -> None:
    documents = {doc.doc_id: doc for doc in load_default_corpus()}
    chunks = build_chunks(list(documents.values()), chunk_size=70, strategy="sentence")
    for chunk in chunks:
        assert len(chunk.text) <= 70
        assert set(chunk.text.split()) <= set(documents[chunk.doc_id].text.split())
    assert "Good chunk size and overlap improve retrieval quality." in [chunk.text for chunk in chunks]


@pytest.mark.parametrize(
    "kwargs",
    [{"strategy": "paragraph"}, {"strategy": "sentence", "sentence_overlap": -1}, {"strategy": "sentence", "chunk_size": 0}],
)
def test_invalid_sentence_chunking_parameters_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        build_chunks(load_default_corpus(), **kwargs)
