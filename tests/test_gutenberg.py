from pathlib import Path
import shutil

from src import gutenberg
from src.corpus import list_corpus_profiles, load_corpus
from src.gutenberg import fetch_book, format_title, load_prince_corpus, parse_sections


FIXTURE = Path(__file__).parent / "fixtures" / "mini_prince.html"


def fixture_sections():
    return parse_sections(FIXTURE.read_text(encoding="utf-8"))


def test_sections_follow_reading_order_with_stable_ids() -> None:
    assert [doc.doc_id for doc in fixture_sections()] == [
        "prince-intro-1",
        "prince-dedication",
        "prince-ch-01",
        "prince-ch-18",
    ]


def test_gutenberg_header_footer_and_license_are_excluded() -> None:
    all_text = " ".join(doc.text for doc in fixture_sections())
    assert "Project Gutenberg" not in all_text
    assert "License" not in all_text
    assert "anyone anywhere" not in all_text


def test_table_of_contents_and_empty_title_section_are_not_documents() -> None:
    sections = fixture_sections()
    assert sum(doc.doc_id == "prince-ch-01" for doc in sections) == 1
    assert all("Contents" not in doc.title for doc in sections)
    assert all(doc.title != "The Prince" for doc in sections)


def test_chapter_text_is_whitespace_normalized_without_footnote_markers() -> None:
    chapter_18 = fixture_sections()[-1]
    assert chapter_18.text.startswith("A prudent lord cannot keep faith when such observance")
    assert "[2]" not in chapter_18.text
    assert "The footnote text stays with its chapter." in chapter_18.text
    assert "Transcriber" not in chapter_18.text


def test_titles_are_readable_and_keep_chapter_numerals() -> None:
    titles = {doc.doc_id: doc.title for doc in fixture_sections()}
    assert titles["prince-ch-18"] == "Chapter XVIII — Concerning The Way In Which Princes Should Keep Faith"
    assert titles["prince-intro-1"] == "Introduction (Marriott)"
    assert titles["prince-dedication"] == "Dedication"
    assert format_title("YOUTH — Æt. 1-25", "Introduction (Marriott)") == "Introduction (Marriott) — Youth — Æt. 1-25"


def test_fetch_book_reads_existing_cache_without_network(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "book.html"
    shutil.copy(FIXTURE, cache)

    def no_network(*args, **kwargs):
        raise AssertionError("network must not be used when the cache exists")

    monkeypatch.setattr(gutenberg.urllib.request, "urlopen", no_network)
    assert fetch_book(cache_path=cache) == FIXTURE.read_text(encoding="utf-8")
    assert [doc.doc_id for doc in load_prince_corpus(cache_path=cache)][-1] == "prince-ch-18"


def test_prince_profile_is_listed_and_loads_the_cached_book() -> None:
    assert "prince" in list_corpus_profiles()
    docs = load_corpus("prince")
    ids = [doc.doc_id for doc in docs]
    assert [f"prince-ch-{n:02d}" for n in range(1, 27)] == [i for i in ids if i.startswith("prince-ch-")]
    assert "prince-dedication" in ids
    chapter_17 = docs[ids.index("prince-ch-17")]
    assert "it is much safer to be feared than loved" in chapter_17.text
