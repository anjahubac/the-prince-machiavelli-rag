import importlib.util
import json
from pathlib import Path

from src.corpus import load_corpus

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = json.loads((ROOT / "eval" / "prince_questions.json").read_text(encoding="utf-8"))


def load_eval_script():
    spec = importlib.util.spec_from_file_location("eval_prince", ROOT / "scripts" / "eval_prince.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_question_file_has_tuning_and_held_out_splits() -> None:
    assert set(QUESTIONS) >= {"tuning", "held_out"}
    for split in ("tuning", "held_out"):
        assert QUESTIONS[split]["supported"] and QUESTIONS[split]["unsupported"]
    n_supported = sum(len(QUESTIONS[s]["supported"]) for s in ("tuning", "held_out"))
    n_unsupported = sum(len(QUESTIONS[s]["unsupported"]) for s in ("tuning", "held_out"))
    assert (n_supported, n_unsupported) == (30, 8)


def test_splits_do_not_share_questions() -> None:
    def all_questions(split):
        return {item["question"] for item in QUESTIONS[split]["supported"]} | set(QUESTIONS[split]["unsupported"])

    assert not all_questions("tuning") & all_questions("held_out")


def test_every_gold_doc_id_is_a_section_of_the_book() -> None:
    doc_ids = {doc.doc_id for doc in load_corpus("prince")}
    for split in ("tuning", "held_out"):
        for item in QUESTIONS[split]["supported"]:
            assert item["doc_id"] in doc_ids, item


def test_selection_prefers_right_outcomes_then_tighter_citations() -> None:
    key = load_eval_script().selection_key
    base = {"hit1": 5, "correct": 10, "refused": 4, "off": 4, "cited": 20}
    assert key({**base, "correct": 11}) < key(base)
    assert key({**base, "off": 2}) < key(base)
    assert key({**base, "hit1": 6}) < key(base)
    assert key({**base, "correct": 11, "off": 15}) < key({**base, "off": 0})
    assert key({**base, "cited": 0, "off": 0}) < key(base)
