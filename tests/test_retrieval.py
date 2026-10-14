from src.chunking import Chunk
import pytest

from src.retrieval import BM25Index, content_tokens, lexical_score, query_term_groups, retrieve_top_k, stem


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


def test_bm25_ranks_rare_term_above_common_term() -> None:
    chunks = [Chunk(chunk_id=f"common-{i}", doc_id="1", title="C", text="the prince rules the state") for i in range(4)]
    chunks.append(Chunk(chunk_id="rare", doc_id="1", title="R", text="the prince hires mercenaries"))
    results = retrieve_top_k("prince mercenaries", chunks, k=2, scorer="bm25")
    assert results[0].chunk.chunk_id == "rare"
    assert results[0].score > 2 * results[1].score


def test_bm25_prefers_shorter_chunk_for_same_matches() -> None:
    chunks = [
        Chunk(chunk_id="long", doc_id="1", title="L", text="fortresses " + "filler words about other matters " * 5),
        Chunk(chunk_id="short", doc_id="1", title="S", text="fortresses are useful"),
    ]
    results = retrieve_top_k("fortresses", chunks, k=2, scorer="bm25")
    assert [item.chunk.chunk_id for item in results] == ["short", "long"]


def test_bm25_counts_repeats_with_diminishing_returns() -> None:
    once = Chunk(chunk_id="once", doc_id="1", title="O", text="fortune favours bold men today")
    twice = Chunk(chunk_id="twice", doc_id="1", title="T", text="fortune fortune bold men today")
    other = Chunk(chunk_id="other", doc_id="1", title="X", text="armies and laws")
    scores = {item.chunk.chunk_id: item.score for item in retrieve_top_k("fortune", [once, twice, other], k=3, scorer="bm25")}
    assert scores["once"] < scores["twice"] < 2 * scores["once"]


def test_bm25_drops_chunks_without_query_terms_and_handles_stemming() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="1", title="A", text="he hired mercenaries"),
        Chunk(chunk_id="b", doc_id="1", title="B", text="weather forecast"),
    ]
    assert retrieve_top_k("mercenary", chunks, scorer="bm25") == []
    assert [r.chunk.chunk_id for r in retrieve_top_k("mercenary", chunks, scorer="bm25", use_stemming=True)] == ["a"]


def test_unknown_scorer_raises() -> None:
    with pytest.raises(ValueError, match="scorer"):
        retrieve_top_k("rag", [], scorer="cosine")


@pytest.mark.parametrize("scorer", ["dice", "bm25"])
def test_title_prefix_lets_a_title_match_rank_its_chunk(scorer: str) -> None:
    body = Chunk("body", "ch-24", "Princes Lost States", "They were idle in quiet times.")
    titled = Chunk("body", "ch-24", "Princes Lost States", body.text, search_prefix=body.title)
    other = Chunk("other", "ch-03", "Mixed", "A prince may take back a state.")
    assert [r.chunk.chunk_id for r in retrieve_top_k("princes lost states", [body, other], scorer=scorer, use_stemming=True)] == ["other"]
    ranked = retrieve_top_k("princes lost states", [titled, other], scorer=scorer, use_stemming=True)
    assert ranked[0].chunk.chunk_id == "body"


def test_query_term_groups_without_expansion_are_the_content_tokens() -> None:
    question = "Should a prince keep his word?"
    groups = query_term_groups(question, use_stemming=True)
    assert set(groups) == content_tokens(question, use_stemming=True)
    assert all(forms == frozenset({term}) for term, forms in groups.items())


def test_query_term_groups_expand_only_listed_words_after_stemming() -> None:
    groups = query_term_groups("generous ministers and troop", use_stemming=True, expand_synonyms=True)
    assert groups["generous"] == {"generous", "liberal", "liberality"}  # -us is kept
    assert groups["minister"] == {"minister", "secretary"}
    assert groups["troop"] == {"troop", "soldiery"}  # singular matches the "troops" entry
    plain = query_term_groups("generous", expand_synonyms=True)
    assert plain == {"generous": frozenset({"generous", "liberal", "liberality"})}


@pytest.mark.parametrize("scorer", ["dice", "bm25"])
def test_synonyms_retrieve_the_older_wording_only_when_enabled(scorer: str) -> None:
    chunks = [
        Chunk("ch-16", "16", "L", "A prince should not fear the name of meanness."),
        Chunk("other", "15", "P", "A prince is praised for some qualities."),
    ]
    plain = retrieve_top_k("miserly", chunks, scorer=scorer)
    expanded = retrieve_top_k("miserly", chunks, scorer=scorer, expand_synonyms=True)
    assert plain == []
    assert [r.chunk.chunk_id for r in expanded] == ["ch-16"]


def test_a_word_and_its_synonym_count_once() -> None:
    both = "liberality and generous deeds"
    one = "liberality and kind deeds"
    assert lexical_score("generous", both, expand_synonyms=True) == lexical_score("generous", one, expand_synonyms=True)
    chunks = [Chunk("both", "1", "B", both), Chunk("one", "1", "O", one), Chunk("x", "1", "X", "armies")]
    index = BM25Index(chunks)
    plain = index.scores("generous")
    expanded = index.scores("generous", expand_synonyms=True)
    assert expanded[0] == pytest.approx(max(plain[0], index.scores("liberality")[0]))
    assert expanded[0] < plain[0] + index.scores("liberality")[0]
