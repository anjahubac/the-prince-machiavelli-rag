"""Compare retrieval settings on The Prince with a small gold question set.

Run from the project root:  python scripts/eval_prince.py [--misses]

For each setting it reports:
- hit@1 / hit@3: the expected chapter is the top / one of the top-3 retrieved chunks
- answered: supported questions answered (not refused)
- correct: answered AND a citation comes from the expected chapter
- refused: unsupported questions correctly refused

--misses also lists, for the recommended setting, every supported question
that was not answered with a citation from the expected chapter.
"""

import argparse
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
RECOMMENDED = {"scorer": "bm25", "use_stemming": True, "min_coverage": 0.75, "include_title": True}


def answer(question, chunks_by_title, settings):
    chunks = chunks_by_title[settings["include_title"]]
    return build_grounded_answer(
        question, chunks, top_k=TOP_K, use_stemming=settings["use_stemming"],
        scorer=settings["scorer"], min_coverage=settings["min_coverage"],
    )


def is_correct(result, doc_id):
    return not result.refused and any(c.startswith(doc_id + "-chunk-") for c in result.citations)


def evaluate(chunks_by_title, questions, settings):
    hit1 = hit3 = answered = correct = 0
    for item in questions["supported"]:
        result = answer(item["question"], chunks_by_title, settings)
        ranked_docs = [r.chunk.doc_id for r in result.retrieved]
        hit1 += ranked_docs[:1] == [item["doc_id"]]
        hit3 += item["doc_id"] in ranked_docs
        answered += not result.refused
        correct += is_correct(result, item["doc_id"])
    refused = sum(answer(q, chunks_by_title, settings).refused for q in questions["unsupported"])
    return hit1, hit3, answered, correct, refused


def print_misses(chunks_by_title, questions, settings):
    print("\n| Question | Expected | Refused | Top-3 docs | Citations |")
    print("|---|---|---|---|---|")
    for item in questions["supported"]:
        result = answer(item["question"], chunks_by_title, settings)
        if is_correct(result, item["doc_id"]):
            continue
        ranked = ", ".join(r.chunk.doc_id for r in result.retrieved)
        cited = ", ".join(result.citations) or "None"
        print(f"| {item['question']} | {item['doc_id']} | {result.refused} | {ranked} | {cited} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--misses", action="store_true", help="List misses for the recommended setting")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus("prince")
    chunks_by_title = {
        include_title: build_chunks(
            corpus, chunk_size=CHUNK_SIZE, strategy="sentence",
            sentence_overlap=SENTENCE_OVERLAP, include_title=include_title,
        )
        for include_title in (False, True)
    }
    n_sup, n_uns = len(questions["supported"]), len(questions["unsupported"])
    print(
        f"The Prince: {len(chunks_by_title[False])} chunks (sentence, chunk_size={CHUNK_SIZE}, "
        f"sentence_overlap={SENTENCE_OVERLAP}), top_k={TOP_K}; "
        f"{n_sup} supported, {n_uns} unsupported questions\n"
    )
    print("| scorer | stem | title | min_coverage | hit@1 | hit@3 | answered | correct | refused |")
    print("|---|---|---|---|---|---|---|---|---|")
    grid = product(("dice", "bm25"), (False, True), (False, True), (1.0, 0.75, 0.5))
    for scorer, use_stemming, include_title, min_coverage in grid:
        settings = {
            "scorer": scorer, "use_stemming": use_stemming,
            "min_coverage": min_coverage, "include_title": include_title,
        }
        hit1, hit3, answered, correct, refused = evaluate(chunks_by_title, questions, settings)
        print(
            f"| {scorer} | {'on' if use_stemming else 'off'} | {'on' if include_title else 'off'} "
            f"| {min_coverage} | {hit1}/{n_sup} | {hit3}/{n_sup} "
            f"| {answered}/{n_sup} | {correct}/{n_sup} | {refused}/{n_uns} |"
        )
    if args.misses:
        print_misses(chunks_by_title, questions, RECOMMENDED)


if __name__ == "__main__":
    main()
