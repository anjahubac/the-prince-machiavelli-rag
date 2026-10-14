"""Load The Prince (Project Gutenberg #1232) as one Document per section.

The Gutenberg HTML marks each section with an <h2> holding an anchor id
(pref01..pref06 for the introduction and dedication, chap01..chap28 for the
chapters and two extra works) and ends it with an <!--end chapter--> comment.
The table of contents only links to those anchors, so it never opens a section.
"""

from html.parser import HTMLParser
from pathlib import Path
import re
from string import capwords
import urllib.request

from .corpus import Document


PRINCE_URL = "https://www.gutenberg.org/cache/epub/1232/pg1232-images.html"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "pg1232.html"
USER_AGENT = "the-prince-machiavelli-rag/0.1 (educational project)"

START_MARKER = "*** START OF"
END_MARKER = "*** END OF"
SKIPPED_TAGS = frozenset({"script", "style", "sup"})
BLOCK_TAGS = frozenset({"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "blockquote"})
FOOTNOTE_REF_RE = re.compile(r"\[\d+\]")

# Anchor id -> (doc_id, title prefix). Chapters are handled by CHAPTER_ANCHOR_RE.
SPECIAL_SECTIONS = {
    "pref01": ("prince-intro-1", "Introduction (Marriott)"),
    "pref02": ("prince-intro-2", "Introduction (Marriott)"),
    "pref03": ("prince-intro-3", "Introduction (Marriott)"),
    "pref04": ("prince-intro-4", "Introduction (Marriott)"),
    "pref05": ("prince-intro-5", "Introduction (Marriott)"),
    "pref06": ("prince-dedication", "Dedication"),
    "chap27": ("extra-valentino", "Extra work"),
    "chap28": ("extra-castruccio", "Extra work"),
}
CHAPTER_ANCHOR_RE = re.compile(r"chap(\d\d)$")
LAST_PRINCE_CHAPTER = 26


def fetch_book(url: str = PRINCE_URL, cache_path: Path = DEFAULT_CACHE_PATH) -> str:
    """Return the book HTML, downloading it once into cache_path."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")
    return cache_path.read_text(encoding="utf-8")


def section_ids(anchor: str):
    """Map a heading anchor to (doc_id, title prefix), or None to skip it."""
    if anchor in SPECIAL_SECTIONS:
        return SPECIAL_SECTIONS[anchor]
    match = CHAPTER_ANCHOR_RE.match(anchor)
    if match and 1 <= int(match.group(1)) <= LAST_PRINCE_CHAPTER:
        return f"prince-ch-{match.group(1)}", None
    return None


def clean_text(text: str) -> str:
    return " ".join(FOOTNOTE_REF_RE.sub("", text).split())


def format_title(heading: str, prefix) -> str:
    """'CHAPTER XVII. CONCERNING CRUELTY' -> 'Chapter XVII — Concerning Cruelty'."""
    heading = clean_text(heading)
    chapter = re.match(r"CHAPTER ([IVXL]+)\.\s*(.*)", heading)
    if chapter:
        return f"Chapter {chapter.group(1)} — {capwords(chapter.group(2))}"
    if prefix and prefix.upper().startswith(heading.upper()):
        return prefix
    if prefix:
        return f"{prefix} — {capwords(heading)}"
    return capwords(heading)


class _SectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_book = False
        self.skip_depth = 0
        self.in_heading = False
        self.current = None  # dict with doc_id, prefix, heading, parts
        self.sections: list = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIPPED_TAGS:
            self.skip_depth += 1
            return
        if tag == "h2" and self.in_book:
            self.in_heading = True
            self.current = {"anchor": None, "heading": [], "parts": []}
        elif tag == "a" and self.in_heading and self.current["anchor"] is None:
            self.current["anchor"] = dict(attrs).get("id")
        if tag in BLOCK_TAGS:
            self._append(" ")

    def handle_endtag(self, tag):
        if tag in SKIPPED_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if tag == "h2" and self.in_heading:
            self.in_heading = False
            ids = section_ids(self.current["anchor"] or "")
            if ids is None:
                self.current = None
            else:
                self.current["doc_id"], self.current["prefix"] = ids
        if tag in BLOCK_TAGS:
            self._append(" ")

    def handle_comment(self, data):
        if data.strip() == "end chapter":
            self._close_section()

    def handle_data(self, data):
        if START_MARKER in data:
            self.in_book = True
            return
        if END_MARKER in data:
            self._close_section()
            self.in_book = False
            return
        if self.in_book and not self.skip_depth:
            self._append(data)

    def _append(self, text: str) -> None:
        if self.current is None:
            return
        self.current["heading" if self.in_heading else "parts"].append(text)

    def _close_section(self) -> None:
        if self.current is not None and "doc_id" in self.current:
            text = clean_text("".join(self.current["parts"]))
            if text:
                title = format_title("".join(self.current["heading"]), self.current["prefix"])
                self.sections.append(Document(doc_id=self.current["doc_id"], title=title, text=text))
        self.current = None


def parse_sections(html: str) -> list:
    """Return the book's sections as Documents, in reading order."""
    parser = _SectionParser()
    parser.feed(html)
    parser.close()
    return parser.sections


def load_prince_corpus(cache_path: Path = DEFAULT_CACHE_PATH) -> list:
    return parse_sections(fetch_book(cache_path=cache_path))
