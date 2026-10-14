import pytest

from src.chunking import build_chunks
from src.corpus import load_default_corpus
from src.rag import REFUSAL_MESSAGE, build_grounded_answer


def test_build_grounded_answer_returns_citations_for_supported_query() -> None:
    chunks = build_chunks(load_default_corpus(), chunk_size=120, overlap=20)
    result = build_grounded_answer("What is retrieval augmented generation?", chunks, top_k=2)
    assert result.refused is False
    assert len(result.citations) >= 1


def test_build_grounded_answer_refuses_unsupported_query() -> None:
    chunks = build_chunks(load_default_corpus(), chunk_size=120, overlap=20)
    result = build_grounded_answer("Who was emperor of Mars?", chunks, top_k=2)
    assert result.refused is True
    assert result.citations == []


@pytest.mark.parametrize("question", ["Who won the world cup in 1958?", "", "the in of"])
def test_refusal_has_no_citations_without_relevant_context(question: str) -> None:
    result = build_grounded_answer(question, build_chunks(load_default_corpus()))
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.retrieved == []
    assert result.citations == []


def test_topic_overlap_is_insufficient_for_an_inventor_question() -> None:
    result = build_grounded_answer(
        "Who invented retrieval augmented generation?", build_chunks(load_default_corpus())
    )
    assert result.retrieved  # Related topics are still visible as evidence.
    assert result.refused is True
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []


def test_citations_reference_only_used_retrieved_chunks() -> None:
    result = build_grounded_answer(
        "What is retrieval augmented generation?", build_chunks(load_default_corpus())
    )
    assert result.refused is False
    assert result.citations == ["doc-1-chunk-0"]
    retrieved = {item.chunk.chunk_id: item.chunk for item in result.retrieved}
    body, references = result.answer.split("\n\nReferences:\n")
    for chunk_id in result.citations:
        chunk = retrieved[chunk_id]
        assert f"[{chunk_id}] {chunk.text}" in body
        assert f"[{chunk_id}] {chunk.title} ({chunk.doc_id})" in references
    assert "[doc-2-chunk-0]" not in result.answer


def test_overlap_preserves_evidence_lost_at_a_character_boundary() -> None:
    question = "How do chunk size and overlap improve retrieval quality?"
    without_overlap = build_grounded_answer(
        question, build_chunks(load_default_corpus(), chunk_size=70, overlap=0), top_k=3
    )
    with_overlap = build_grounded_answer(
        question, build_chunks(load_default_corpus(), chunk_size=70, overlap=20), top_k=3
    )
    assert without_overlap.refused is True
    assert with_overlap.refused is False
    assert with_overlap.citations == ["doc-2-chunk-1"]


def test_stemming_answers_a_question_that_exact_matching_refuses() -> None:
    chunks = build_chunks(load_default_corpus())
    question = "What does metadata help with?"
    assert build_grounded_answer(question, chunks).refused is True
    stemmed = build_grounded_answer(question, chunks, use_stemming=True)
    assert stemmed.refused is False
    assert stemmed.citations == ["doc-2-chunk-0"]
    assert "Metadata helps filtering and source traceability." in stemmed.answer


@pytest.mark.parametrize("question", ["Who invented retrieval augmented generation?", "Who won the world cup in 1958?"])
def test_stemming_does_not_weaken_refusal(question: str) -> None:
    result = build_grounded_answer(question, build_chunks(load_default_corpus()), use_stemming=True)
    assert result.refused is True
    assert result.citations == []


def test_sentence_chunks_give_a_clean_citation_for_the_overlap_question() -> None:
    chunks = build_chunks(load_default_corpus(), chunk_size=70, strategy="sentence")
    result = build_grounded_answer("How do chunk size and overlap improve retrieval quality?", chunks, top_k=3)
    assert result.refused is False
    assert result.citations == ["doc-2-chunk-1"]
    assert result.retrieved[0].chunk.text == "Good chunk size and overlap improve retrieval quality."


def test_min_coverage_answers_partial_match_that_full_coverage_refuses() -> None:
    from src.chunking import Chunk

    chunks = [Chunk(chunk_id="c", doc_id="d", title="T", text="It is much safer to be feared than loved.")]
    question = "Is it better to be feared than loved?"  # 'better' is not in the text
    assert build_grounded_answer(question, chunks, scorer="bm25").refused is True
    partial = build_grounded_answer(question, chunks, scorer="bm25", min_coverage=0.6)
    assert partial.refused is False
    assert partial.citations == ["c"]


@pytest.mark.parametrize("bad", [0, -0.5, 1.5])
def test_min_coverage_must_be_a_fraction(bad: float) -> None:
    with pytest.raises(ValueError, match="min_coverage"):
        build_grounded_answer("rag", build_chunks(load_default_corpus()), min_coverage=bad)


def test_title_words_count_for_coverage_but_are_not_quoted() -> None:
    from src.chunking import Chunk

    text = "Princes who were idle in quiet times lost everything when war came."
    plain = Chunk("ch-24-chunk-0", "ch-24", "Why Princes Of Italy Lost Their States", text)
    titled = Chunk(plain.chunk_id, plain.doc_id, plain.title, text, search_prefix=plain.title)
    question = "Why did the princes of Italy lose their states?"
    assert build_grounded_answer(question, [plain], scorer="bm25", use_stemming=True).refused is True
    # Terms: did, princes, italy, lose, states; "lose" never matches "Lost", so 3 of 5.
    assert build_grounded_answer(question, [plain], scorer="bm25", use_stemming=True, min_coverage=0.6).refused
    result = build_grounded_answer(question, [titled], scorer="bm25", use_stemming=True, min_coverage=0.6)
    assert result.refused is False
    assert result.citations == ["ch-24-chunk-0"]
    assert f"[ch-24-chunk-0] {text}\n" in result.answer
    assert f"[ch-24-chunk-0] {plain.title} (ch-24)" in result.answer


def _feared_chunks():
    from src.chunking import Chunk

    return [
        Chunk("strong", "ch-17", "T17", "It is much safer to be feared than loved, feared and loved."),
        Chunk("weak", "ch-03", "T03", "It is better to act at once than to wait."),
    ]


def test_min_citation_ratio_drops_a_weak_chunk_that_only_adds_a_word() -> None:
    question = "Is it better to be loved or feared?"
    loose = build_grounded_answer(question, _feared_chunks(), scorer="bm25", min_coverage=0.6)
    assert loose.citations == ["strong", "weak"]
    ratio = loose.retrieved[1].score / loose.retrieved[0].score
    tight = build_grounded_answer(
        question, _feared_chunks(), scorer="bm25", min_coverage=0.6, min_citation_ratio=ratio + 0.01
    )
    assert tight.refused is False
    assert tight.citations == ["strong"]
    assert "[weak]" not in tight.answer
    assert [item.chunk.chunk_id for item in tight.retrieved] == ["strong", "weak"]  # still shown


def test_min_citation_ratio_also_limits_coverage_evidence() -> None:
    question = "Is it better to be loved or feared?"  # terms: better, loved, feared
    loose = build_grounded_answer(question, _feared_chunks(), scorer="bm25", min_coverage=1.0)
    assert loose.refused is False  # "better" comes from the weak chunk
    tight = build_grounded_answer(question, _feared_chunks(), scorer="bm25", min_coverage=1.0, min_citation_ratio=0.9)
    assert tight.refused is True
    assert tight.citations == []


def test_min_citation_ratio_zero_and_one_keep_the_top_chunk() -> None:
    question = "Is it better to be loved or feared?"
    for ratio in (0.0, 1.0):
        result = build_grounded_answer(
            question, _feared_chunks(), scorer="bm25", min_coverage=0.6, min_citation_ratio=ratio
        )
        assert result.citations[0] == "strong"


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_min_citation_ratio_must_be_a_fraction(bad: float) -> None:
    with pytest.raises(ValueError, match="min_citation_ratio"):
        build_grounded_answer("rag", build_chunks(load_default_corpus()), min_citation_ratio=bad)
