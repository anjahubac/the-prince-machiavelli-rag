from dataclasses import dataclass
import re

from .corpus import Document


# Character-based defaults keep this small corpus's source statements intact.
# Compare smaller windows and different overlaps explicitly in the CLI.
DEFAULT_CHUNK_SIZE = 220
DEFAULT_OVERLAP = 40
DEFAULT_SENTENCE_OVERLAP = 0
CHUNKING_STRATEGIES = ("char", "sentence")

# A sentence ends at . ! or ? followed by whitespace. Abbreviations such as
# "e.g." are also treated as sentence ends; the default corpus has none.
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    # Extra words that count when matching but are not quoted in answers,
    # e.g. the chapter title (see build_chunks(include_title=True)).
    search_prefix: str = ""

    @property
    def search_text(self) -> str:
        """Text used for retrieval and coverage checks: search_prefix + text."""
        return f"{self.search_prefix} {self.text}" if self.search_prefix else self.text


def split_text_with_overlap(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start += step

    return chunks


def split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return SENTENCE_END_RE.split(cleaned)


def split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Pack whole words into chunks; only a single over-long word is cut."""
    parts: list[str] = []
    current = ""
    for word in sentence.split():
        if len(word) > chunk_size:
            if current:
                parts.append(current)
                current = ""
            parts.extend(split_text_with_overlap(word, chunk_size=chunk_size, overlap=0))
        elif current and len(current) + 1 + len(word) > chunk_size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def split_text_by_sentence(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    sentence_overlap: int = DEFAULT_SENTENCE_OVERLAP,
) -> list[str]:
    """Group whole sentences into chunks of at most chunk_size characters.

    Overlap is measured in sentences: each new chunk starts by repeating up to
    sentence_overlap trailing sentences of the previous chunk, if they still
    leave room for the next sentence. A sentence longer than chunk_size is
    split at word boundaries so no chunk exceeds the limit.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if sentence_overlap < 0:
        raise ValueError("sentence_overlap must be >= 0")

    chunks: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(split_long_sentence(sentence, chunk_size))
            continue

        if current and len(" ".join(current + [sentence])) > chunk_size:
            chunks.append(" ".join(current))
            carried = current[-sentence_overlap:] if sentence_overlap else []
            # Drop carried sentences until the next sentence fits, so every
            # chunk adds new text and the loop always makes progress.
            while carried and len(" ".join(carried + [sentence])) > chunk_size:
                carried = carried[1:]
            current = carried
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = "char",
    sentence_overlap: int = DEFAULT_SENTENCE_OVERLAP,
    include_title: bool = False,
) -> list[Chunk]:
    """Split each document into chunks.

    With include_title=True every chunk also matches on its document title
    (as Chunk.search_prefix); the quoted chunk text is unchanged.
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"strategy must be one of {CHUNKING_STRATEGIES}")

    chunks: list[Chunk] = []
    for doc in documents:
        if strategy == "sentence":
            parts = split_text_by_sentence(doc.text, chunk_size=chunk_size, sentence_overlap=sentence_overlap)
        else:
            parts = split_text_with_overlap(doc.text, chunk_size=chunk_size, overlap=overlap)
        for i, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-chunk-{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=part,
                    search_prefix=doc.title if include_title else "",
                )
            )
    return chunks
