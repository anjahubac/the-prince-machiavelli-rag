"""Compare retrieval settings on The Prince with a small gold question set.

Run from the project root:  python scripts/eval_prince.py

For each setting it reports:
- hit@1 / hit@3: the expected chapter is the top / one of the top-3 retrieved chunks
- answered: supported questions answered (not refused)
- correct: answered AND a citation comes from the expected chapter
- refused: unsupported questions correctly refused
"""

from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import build_chunks  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.rag import build_grounded_answer  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "prince_questions.json"
CHUNK_SIZE = 700
SENTENCE_OVERLAP = 1
TOP_K = 3


def evaluate(chunks, questions, scorer, use_stemming, min_coverage):
    supported = questions["supported"]
    hit1 = hit3 = answered = correct = 0
    for item in supported:
        result = build_grounded_answer(
            item["question"], chunks, top_k=TOP_K, use_stemming=use_stemming,
            scorer=scorer, min_coverage=min_coverage,
        )
        ranked_docs = [r.chunk.doc_id for r in result.retrieved]
        hit1 += ranked_docs[:1] == [item["doc_id"]]
        hit3 += item["doc_id"] in ranked_docs
        if not result.refused:
            answered += 1
            correct += any(c.startswith(item["doc_id"] + "-chunk-") for c in result.citations)
    refused = sum(
        build_grounded_answer(
            q, chunks, top_k=TOP_K, use_stemming=use_stemming, scorer=scorer, min_coverage=min_coverage
        ).refused
        for q in questions["unsupported"]
    )
    return hit1, hit3, answered, correct, refused


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    chunks = build_chunks(
        load_corpus("prince"), chunk_size=CHUNK_SIZE, strategy="sentence", sentence_overlap=SENTENCE_OVERLAP
    )
    n_sup, n_uns = len(questions["supported"]), len(questions["unsupported"])
    print(
        f"The Prince: {len(chunks)} chunks (sentence, chunk_size={CHUNK_SIZE}, "
        f"sentence_overlap={SENTENCE_OVERLAP}), top_k={TOP_K}; "
        f"{n_sup} supported, {n_uns} unsupported questions\n"
    )
    print("| scorer | stem | min_coverage | hit@1 | hit@3 | answered | correct | refused |")
    print("|---|---|---|---|---|---|---|---|")
    for scorer, use_stemming, min_coverage in product(("dice", "bm25"), (False, True), (1.0, 0.75, 0.5)):
        hit1, hit3, answered, correct, refused = evaluate(chunks, questions, scorer, use_stemming, min_coverage)
        print(
            f"| {scorer} | {'on' if use_stemming else 'off'} | {min_coverage} | {hit1}/{n_sup} | {hit3}/{n_sup} "
            f"| {answered}/{n_sup} | {correct}/{n_sup} | {refused}/{n_uns} |"
        )


if __name__ == "__main__":
    main()
