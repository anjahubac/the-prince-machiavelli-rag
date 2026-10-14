"""Compare retrieval settings on The Prince with a gold question set.

Run from the project root:  python scripts/eval_prince.py [--misses]

eval/prince_questions.json has a "tuning" and a "held_out" split. Every
setting in the grid is scored on the tuning split only, and the best one is
picked by SELECTION_RULE. Only then is the held-out split run, for the picked
setting and a few reference settings; held-out numbers never feed back.

Metrics:
- hit@1 / hit@3: the expected chapter is the top / one of the top-3 retrieved chunks
- answered: supported questions answered (not refused)
- correct: answered AND a citation comes from the expected chapter
- refused: unsupported questions correctly refused
- off-chapter: citations (over all answered supported questions) not from the
  expected chapter, out of all citations. Lower is tighter; some off-chapter
  citations can still be relevant passages.

--misses lists, for the picked setting, every supported question that was not
answered with a citation from the expected chapter, on both splits.
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
COVERAGES = (1.0, 0.75, 0.5)
CITATION_RATIOS = (0.0, 0.5, 0.8, 0.9)
SELECTION_RULE = (
    "most (correct + refused) on tuning; ties -> lowest off-chapter share, "
    "then most hit@1, then earliest in grid order (fewer options on)"
)

# Reference settings shown on the held-out split next to the picked one.
REFERENCES = {
    "homework default": {"scorer": "dice", "use_stemming": False, "min_coverage": 1.0},
    "baseline (before steps 1-3)": {"scorer": "bm25", "use_stemming": True, "min_coverage": 0.75},
    "+ title": {"scorer": "bm25", "use_stemming": True, "min_coverage": 0.75, "include_title": True},
    "+ title + synonyms": {
        "scorer": "bm25", "use_stemming": True, "min_coverage": 0.75,
        "include_title": True, "expand_synonyms": True,
    },
}


def on_off(flag):
    return "on" if flag else "off"


def describe(settings):
    return (
        f"{settings['scorer']}, stem {on_off(settings['use_stemming'])}, "
        f"title {on_off(settings.get('include_title', False))}, "
        f"synonyms {on_off(settings.get('expand_synonyms', False))}, "
        f"min_coverage {settings['min_coverage']}, min_citation_ratio {settings.get('min_citation_ratio', 0.0)}"
    )


def answer(question, chunks_by_title, settings):
    chunks = chunks_by_title[settings.get("include_title", False)]
    return build_grounded_answer(
        question, chunks, top_k=TOP_K, use_stemming=settings["use_stemming"],
        scorer=settings["scorer"], min_coverage=settings["min_coverage"],
        min_citation_ratio=settings.get("min_citation_ratio", 0.0),
        expand_synonyms=settings.get("expand_synonyms", False),
    )


def is_correct(result, doc_id):
    return not result.refused and any(c.startswith(doc_id + "-chunk-") for c in result.citations)


def evaluate(chunks_by_title, split, settings):
    """Return a dict of metric counts for one setting on one split."""
    hit1 = hit3 = answered = correct = cited = off_chapter = 0
    for item in split["supported"]:
        result = answer(item["question"], chunks_by_title, settings)
        ranked_docs = [r.chunk.doc_id for r in result.retrieved]
        hit1 += ranked_docs[:1] == [item["doc_id"]]
        hit3 += item["doc_id"] in ranked_docs
        answered += not result.refused
        correct += is_correct(result, item["doc_id"])
        cited += len(result.citations)
        off_chapter += sum(not c.startswith(item["doc_id"] + "-chunk-") for c in result.citations)
    refused = sum(answer(q, chunks_by_title, settings).refused for q in split["unsupported"])
    return {
        "hit1": hit1, "hit3": hit3, "answered": answered, "correct": correct,
        "refused": refused, "off": off_chapter, "cited": cited,
    }


def metric_cells(m, split):
    n_sup, n_uns = len(split["supported"]), len(split["unsupported"])
    return (
        f"{m['hit1']}/{n_sup} | {m['hit3']}/{n_sup} | {m['answered']}/{n_sup} | {m['correct']}/{n_sup} "
        f"| {m['refused']}/{n_uns} | {m['off']}/{m['cited']}"
    )


def selection_key(m):
    """Sort key for SELECTION_RULE (grid order breaks the remaining ties)."""
    off_share = m["off"] / m["cited"] if m["cited"] else 0.0
    return (-(m["correct"] + m["refused"]), off_share, -m["hit1"])


def grid_settings():
    """Retrieval grid at min_citation_ratio 0, then a citation-ratio sweep at stem on, coverage 0.75."""
    retrieval = [
        {"scorer": scorer, "use_stemming": stem, "include_title": title,
         "expand_synonyms": synonyms, "min_coverage": coverage, "min_citation_ratio": 0.0}
        for scorer, stem, title, synonyms, coverage in product(
            ("dice", "bm25"), (False, True), (False, True), (False, True), COVERAGES
        )
    ]
    ratios = [
        {"scorer": scorer, "use_stemming": True, "include_title": title,
         "expand_synonyms": synonyms, "min_coverage": 0.75, "min_citation_ratio": ratio}
        for scorer, title, synonyms, ratio in product(
            ("dice", "bm25"), (False, True), (False, True), CITATION_RATIOS[1:]
        )
    ]
    return retrieval, ratios


def print_misses(chunks_by_title, split, settings):
    print("| Question | Expected | Refused | Top-3 docs | Citations |")
    print("|---|---|---|---|---|")
    for item in split["supported"]:
        result = answer(item["question"], chunks_by_title, settings)
        if is_correct(result, item["doc_id"]):
            continue
        ranked = ", ".join(r.chunk.doc_id for r in result.retrieved)
        cited = ", ".join(result.citations) or "None"
        print(f"| {item['question']} | {item['doc_id']} | {result.refused} | {ranked} | {cited} |")
    wrongly_answered = [q for q in split["unsupported"] if not answer(q, chunks_by_title, settings).refused]
    for question in wrongly_answered:
        print(f"| {question} | (unsupported) | False | | answered |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--misses", action="store_true", help="List misses for the picked setting")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    tuning, held_out = questions["tuning"], questions["held_out"]
    corpus = load_corpus("prince")
    chunks_by_title = {
        include_title: build_chunks(
            corpus, chunk_size=CHUNK_SIZE, strategy="sentence",
            sentence_overlap=SENTENCE_OVERLAP, include_title=include_title,
        )
        for include_title in (False, True)
    }
    print(
        f"The Prince: {len(chunks_by_title[False])} chunks (sentence, chunk_size={CHUNK_SIZE}, "
        f"sentence_overlap={SENTENCE_OVERLAP}), top_k={TOP_K}\n"
        f"Tuning: {len(tuning['supported'])} supported, {len(tuning['unsupported'])} unsupported; "
        f"held-out: {len(held_out['supported'])} supported, {len(held_out['unsupported'])} unsupported\n"
    )

    retrieval, ratios = grid_settings()
    scored = []
    print("## Tuning split: retrieval grid (min_citation_ratio 0)\n")
    print("| scorer | stem | title | synonyms | min_coverage | hit@1 | hit@3 | answered | correct | refused | off-chapter |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for settings in retrieval:
        metrics = evaluate(chunks_by_title, tuning, settings)
        scored.append((settings, metrics))
        print(
            f"| {settings['scorer']} | {on_off(settings['use_stemming'])} | {on_off(settings['include_title'])} "
            f"| {on_off(settings['expand_synonyms'])} | {settings['min_coverage']} | {metric_cells(metrics, tuning)} |"
        )

    print("\n## Tuning split: citation threshold (stem on, min_coverage 0.75; ratio 0 rows are above)\n")
    print("| scorer | title | synonyms | min_citation_ratio | hit@1 | hit@3 | answered | correct | refused | off-chapter |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for settings in ratios:
        metrics = evaluate(chunks_by_title, tuning, settings)
        scored.append((settings, metrics))
        print(
            f"| {settings['scorer']} | {on_off(settings['include_title'])} | {on_off(settings['expand_synonyms'])} "
            f"| {settings['min_citation_ratio']} | {metric_cells(metrics, tuning)} |"
        )

    # sorted() is stable, so equal keys keep grid order.
    ranking = sorted(scored, key=lambda pair: selection_key(pair[1]))
    picked, _ = ranking[0]
    print(f"\n## Picked on tuning\n\nRule: {SELECTION_RULE}.\n")
    print("| rank | setting | hit@1 | hit@3 | answered | correct | refused | off-chapter |")
    print("|---|---|---|---|---|---|---|---|")
    for rank, (settings, metrics) in enumerate(ranking[:5], start=1):
        print(f"| {rank} | {describe(settings)} | {metric_cells(metrics, tuning)} |")

    print("\n## Held-out split (not used for picking)\n")
    print("| setting | hit@1 | hit@3 | answered | correct | refused | off-chapter |")
    print("|---|---|---|---|---|---|---|")
    rows = list(REFERENCES.items()) + [("picked on tuning: " + describe(picked), picked)]
    for name, settings in rows:
        print(f"| {name} | {metric_cells(evaluate(chunks_by_title, held_out, settings), held_out)} |")

    if args.misses:
        for name, split in (("tuning", tuning), ("held-out", held_out)):
            print(f"\n## Misses of the picked setting, {name} split\n")
            print_misses(chunks_by_title, split, picked)


if __name__ == "__main__":
    main()
