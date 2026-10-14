import subprocess
import sys


def run_cli(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def test_cli_shows_full_supported_evidence() -> None:
    output = run_cli(
        "--question", "What is retrieval augmented generation?",
        "--chunk-size", "220", "--overlap", "40", "--top-k", "3",
    )
    assert "Chunking: chunk_size=220, overlap=40 (characters)" in output
    assert "Top-k: 3; corpus chunks: 3" in output
    assert "doc-1-chunk-0 | score=" in output
    assert "Retrieval augmented generation combines retrieval with text generation." in output
    assert "References:\n[doc-1-chunk-0] RAG Fundamentals (doc-1)" in output
    assert "Refused: False\nCitations: doc-1-chunk-0" in output


def test_cli_preserves_retrieved_context_on_refusal() -> None:
    output = run_cli("--question", "Who invented retrieval augmented generation?")
    assert "Retrieved context (descending score):\n1. doc-1-chunk-0 | score=" in output
    assert "I do not have enough grounded evidence" in output
    assert "Refused: True\nCitations: None" in output


def test_cli_reports_sentence_strategy_and_stemming() -> None:
    output = run_cli(
        "--question", "What does metadata help with?",
        "--strategy", "sentence", "--chunk-size", "70", "--sentence-overlap", "1", "--stem",
    )
    assert "Chunking: strategy=sentence, chunk_size=70 (characters), sentence_overlap=1 (sentences)" in output
    assert "Matching: stemming=on" in output
    assert "Refused: False" in output


def test_cli_default_output_has_no_stemming_line() -> None:
    assert "Matching:" not in run_cli("--question", "What is retrieval augmented generation?")


def test_cli_reports_scorer_and_coverage() -> None:
    output = run_cli(
        "--question", "What is retrieval augmented generation?", "--scorer", "bm25", "--min-coverage", "0.75",
    )
    assert "Scoring: scorer=bm25, min_coverage=0.75" in output
    assert "Refused: False\nCitations: doc-1-chunk-0" in output


def test_cli_default_output_has_no_scoring_line() -> None:
    assert "Scoring:" not in run_cli("--question", "What is retrieval augmented generation?")


def test_cli_answers_from_the_prince() -> None:
    output = run_cli(
        "--corpus-profile", "prince", "--strategy", "sentence", "--chunk-size", "700",
        "--sentence-overlap", "1", "--stem", "--scorer", "bm25", "--top-k", "3",
        "--question", "Is it better for a prince to be loved or feared?",
    )
    assert "1. prince-ch-17-chunk-" in output
    assert "Refused: False" in output


def test_cli_title_in_chunk_finds_the_chapter_named_by_the_question() -> None:
    output = run_cli(
        "--corpus-profile", "prince", "--strategy", "sentence", "--chunk-size", "700",
        "--sentence-overlap", "1", "--stem", "--scorer", "bm25", "--min-coverage", "0.75", "--top-k", "3",
        "--title-in-chunk", "--question", "Why did the princes of Italy lose their states?",
    )
    assert "Matching: stemming=on, title_in_chunk=on" in output
    assert "1. prince-ch-24-chunk-" in output
    assert "Citations: prince-ch-24-chunk-" in output


def test_cli_min_citation_ratio_drops_the_weak_citation() -> None:
    output = run_cli(
        "--corpus-profile", "prince", "--strategy", "sentence", "--chunk-size", "700",
        "--sentence-overlap", "1", "--stem", "--scorer", "bm25", "--min-coverage", "0.75", "--top-k", "3",
        "--min-citation-ratio", "0.9", "--question", "Is it better for a prince to be loved or feared?",
    )
    assert "Citation filter: min_citation_ratio=0.9" in output
    assert "2. prince-ch-03-chunk-12 | score=" in output
    assert "Refused: False\nCitations: prince-ch-17-chunk-3\n" in output


def test_cli_default_output_has_no_citation_filter_line() -> None:
    assert "Citation filter:" not in run_cli("--question", "What is retrieval augmented generation?")


def test_cli_expand_synonyms_finds_meanness_for_miserly() -> None:
    output = run_cli(
        "--corpus-profile", "prince", "--strategy", "sentence", "--chunk-size", "700",
        "--sentence-overlap", "1", "--stem", "--scorer", "bm25", "--min-coverage", "0.75", "--top-k", "3",
        "--title-in-chunk", "--expand-synonyms",
        "--question", "Is it better for a prince to be generous or miserly?",
    )
    assert "Matching: stemming=on, title_in_chunk=on, synonyms=on" in output
    assert "1. prince-ch-16-chunk-" in output
    assert "Citations: prince-ch-16-chunk-" in output
