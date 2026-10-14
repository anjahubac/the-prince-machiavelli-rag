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
