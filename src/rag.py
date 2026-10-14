from dataclasses import dataclass, field

from .chunking import Chunk
from .retrieval import RetrievalResult, content_tokens, retrieve_top_k


REFUSAL_MESSAGE = "I do not have enough grounded evidence in the provided corpus to answer that question."


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[str]
    refused: bool
    retrieved: list[RetrievalResult] = field(default_factory=list)


def build_grounded_answer(
    question: str,
    chunks: list[Chunk],
    top_k: int = 2,
    use_stemming: bool = False,
    scorer: str = "dice",
    min_coverage: float = 1.0,
) -> RagAnswer:
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")

    results = retrieve_top_k(question, chunks, k=top_k, use_stemming=use_stemming, scorer=scorer)
    # Retrieval, refusal, and citation checks must compare terms the same way.
    query_terms = content_tokens(question, use_stemming)
    evidence_terms = set().union(*(content_tokens(item.chunk.text, use_stemming) for item in results))
    # A topic match alone does not justify an answer. Require at least
    # min_coverage of the question's content terms in the retrieved context
    # (1.0 = every term). This lexical check is not semantic entailment.
    coverage = len(query_terms & evidence_terms) / len(query_terms) if query_terms else 0.0
    if not results or coverage < min_coverage:
        return RagAnswer(answer=REFUSAL_MESSAGE, citations=[], refused=True, retrieved=results)

    supporting_results = []
    uncovered_terms = query_terms.copy()
    for item in results:
        matched_terms = uncovered_terms & content_tokens(item.chunk.text, use_stemming)
        if matched_terms:
            supporting_results.append(item)
            uncovered_terms -= matched_terms

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
