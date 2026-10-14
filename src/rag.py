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
    question: str, chunks: list[Chunk], top_k: int = 2, use_stemming: bool = False
) -> RagAnswer:
    results = retrieve_top_k(question, chunks, k=top_k, use_stemming=use_stemming)
    # Retrieval, refusal, and citation checks must compare terms the same way.
    query_terms = content_tokens(question, use_stemming)
    evidence_terms = set().union(*(content_tokens(item.chunk.text, use_stemming) for item in results))
    # A topic match alone does not justify an answer. Conservatively require
    # every content term in the retrieved context before quoting evidence.
    # This lexical check is intentionally simple, not semantic entailment.
    if not results or not query_terms or not query_terms <= evidence_terms:
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
