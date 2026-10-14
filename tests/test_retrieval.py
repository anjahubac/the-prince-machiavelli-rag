from src.chunking import Chunk
import pytest

from src.retrieval import content_tokens, lexical_score, retrieve_top_k, stem


def test_retrieve_top_k_orders_by_score_desc() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="rag retrieval citations evidence"),
        Chunk(chunk_id="b", doc_id="1", title="B", text="weather forecast rain tomorrow"),
    ]
    results = retrieve_top_k("rag citations", chunks, k=2)
    assert results
    assert results[0].chunk.chunk_id == "a"


def test_retrieve_top_k_returns_empty_for_unrelated_query() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="vector database retrieval"),
    ]
    results = retrieve_top_k("basketball playoffs", chunks, k=2)
    assert results == []


@pytest.mark.parametrize("k", [2, 3])
def test_top_k_has_expected_scores_and_order(k: int) -> None:
    chunks = [
        Chunk("weak", "1", "Weak", "citations chunk size overlap metadata source retrieval"),
        Chunk("strong", "2", "Strong", "rag citations evidence"),
        Chunk("medium", "3", "Medium", "rag retrieval evidence database"),
        Chunk("unrelated", "4", "Other", "the in of"),
    ]
    results = retrieve_top_k("rag citations", chunks, k=k)
    assert [item.chunk.chunk_id for item in results] == ["strong", "medium", "weak"][:k]
    assert [item.score for item in results] == pytest.approx([0.8, 1 / 3, 2 / 9][:k])


def test_common_words_do_not_retrieve_unrelated_evidence() -> None:
    chunks = [Chunk("a", "1", "A", "The model should answer using retrieved evidence.")]
    assert retrieve_top_k("Who won the world cup in 1958?", chunks) == []
    assert retrieve_top_k("the in of", chunks, min_score=0) == []


def test_repeated_terms_do_not_inflate_score() -> None:
    assert lexical_score("rag citations", "rag rag evidence") == lexical_score(
        "rag citations", "rag evidence"
    )


def test_ties_preserve_input_order_and_threshold_filters() -> None:
    chunks = [Chunk(chunk_id, "1", chunk_id, "rag evidence") for chunk_id in ["b", "a"]]
    assert [item.chunk.chunk_id for item in retrieve_top_k("rag", chunks)] == ["b", "a"]
    assert retrieve_top_k("rag", chunks, min_score=0.7) == []


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_raises(k: int) -> None:
    with pytest.raises(ValueError, match="k must be > 0"):
        retrieve_top_k("rag", [], k=k)


@pytest.mark.parametrize("min_score", [-0.1, 1.1, float("nan")])
def test_invalid_min_score_raises(min_score: float) -> None:
    with pytest.raises(ValueError, match="min_score"):
        retrieve_top_k("rag", [], min_score=min_score)


@pytest.mark.parametrize(
    "word, expected",
    [
        ("helps", "help"),
        ("studies", "study"),
        ("running", "run"),
        ("chunking", "chunk"),
        ("falling", "fall"),
        ("class", "class"),
        ("status", "status"),
        ("analysis", "analysis"),
        ("rag", "rag"),
    ],
)
def test_stem_reduces_word_forms(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_stem_maps_related_forms_to_the_same_term() -> None:
    assert stem("retrieve") == stem("retrieved")
    assert stem("chunk") == stem("chunks") == stem("chunking")


def test_stemming_is_off_by_default_and_keeps_stop_words_out() -> None:
    assert content_tokens("Metadata helps") == {"metadata", "helps"}
    assert content_tokens("What does metadata help with?", use_stemming=True) == {"metadata", "help"}


def test_stemmed_forms_count_once_and_cannot_inflate_score() -> None:
    assert content_tokens("help helps helping", use_stemming=True) == {"help"}
    assert lexical_score("help", "help helps helping", use_stemming=True) == 1.0


def test_stemming_lets_word_forms_retrieve_a_chunk() -> None:
    chunks = [Chunk("a", "1", "A", "Metadata helps filtering.")]
    assert lexical_score("help filter", chunks[0].text) == 0.0
    assert [item.chunk.chunk_id for item in retrieve_top_k("help filter", chunks, use_stemming=True)] == ["a"]
