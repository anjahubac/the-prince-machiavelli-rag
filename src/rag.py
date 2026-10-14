from dataclasses import dataclass, field

from .chunking import Chunk
from .retrieval import RetrievalResult, content_tokens, matched_groups, query_term_groups, retrieve_top_k


REFUSAL_MESSAGE = "I do not have enough grounded evidence in the provided corpus to answer that question."


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[str]
    refused: bool
    retrieved: list[RetrievalResult] = field(default_factory=list)


def citable_results(results: list[RetrievalResult], min_citation_ratio: float) -> list[RetrievalResult]:
    """Keep results scoring at least min_citation_ratio x the top score.

    Scores are only compared within one query, so this works for Dice and BM25.
    """
    if not results:
        return []
    threshold = min_citation_ratio * results[0].score
    return [item for item in results if item.score >= threshold]


def build_grounded_answer(
    question: str,
    chunks: list[Chunk],
    top_k: int = 2,
    use_stemming: bool = False,
    scorer: str = "dice",
    min_coverage: float = 1.0,
    min_citation_ratio: float = 0.0,
    expand_synonyms: bool = False,
) -> RagAnswer:
    """Answer by quoting retrieved chunks, or refuse.

    min_citation_ratio > 0 keeps only chunks scoring at least that fraction of
    the top chunk's score as evidence: only they can be cited, and only their
    terms count for min_coverage. 0.0 (default) uses every retrieved chunk.

    expand_synonyms=True lets a question word match its SYNONYMS (retrieval.py)
    in retrieval, coverage and citation checks alike.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if not 0 <= min_citation_ratio <= 1:
        raise ValueError("min_citation_ratio must be between 0 and 1")

    results = retrieve_top_k(
        question, chunks, k=top_k, use_stemming=use_stemming, scorer=scorer, expand_synonyms=expand_synonyms
    )
    evidence = citable_results(results, min_citation_ratio)
    # Retrieval, refusal, and citation checks must compare terms the same way.
    query_groups = query_term_groups(question, use_stemming, expand_synonyms)
    evidence_terms = set().union(*(content_tokens(item.chunk.search_text, use_stemming) for item in evidence))
    # A topic match alone does not justify an answer. Require at least
    # min_coverage of the question's content terms in the retrieved context
    # (1.0 = every term). This lexical check is not semantic entailment.
    coverage = len(matched_groups(query_groups, evidence_terms)) / len(query_groups) if query_groups else 0.0
    if not results or coverage < min_coverage:
        return RagAnswer(answer=REFUSAL_MESSAGE, citations=[], refused=True, retrieved=results)

    supporting_results = []
    uncovered = dict(query_groups)
    for item in evidence:
        matched = matched_groups(uncovered, content_tokens(item.chunk.search_text, use_stemming))
        if matched:
            supporting_results.append(item)
            for term in matched:
                del uncovered[term]

    citations = [item.chunk.chunk_id for item in supporting_results]
    supporting_points = [f"[{item.chunk.chunk_id}] {item.chunk.text}" for item in supporting_results]
    references = [
        f"[{item.chunk.chunk_id}] {item.chunk.title} ({item.chunk.doc_id})"
        for item in supporting_results
    ]
    answer = (
        "Grounded response based on retrieved context:\n"
        + "\n".join(supporting_points)
        + "\n\nReferences:\n"
        + "\n".join(references)
    )
    return RagAnswer(answer=answer, citations=citations, refused=False, retrieved=results)
