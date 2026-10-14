from dataclasses import dataclass
import re

from .chunking import Chunk


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
STOP_WORDS = frozenset(
    "a an and are as at be been but by can do does for from had has have "
    "how i if in is it of on or should that the their these this those to "
    "was were what when where which who why will with would".split()
)


VOWELS_AND_SAFE_DOUBLES = frozenset("aeiouls")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def stem(word: str) -> str:
    """Light suffix stemmer so word forms match, e.g. helps/help.

    Rules, applied in order and only when at least 3 letters remain:
    -ies -> -y; -sses -> -ss; -s removed (but -ss, -us, -is kept);
    -ing or -ed removed, collapsing a doubled final consonant; final -e removed.
    Irregular forms (ran/run) are not handled and some words are over-stemmed.
    """
    if len(word) <= 3:
        return word

    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]

    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if word[-1] == word[-2] and word[-1] not in VOWELS_AND_SAFE_DOUBLES:
                word = word[:-1]
            break

    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def content_tokens(text: str, use_stemming: bool = False) -> set[str]:
    """Keep distinct content terms; optionally stem them. No semantic matching."""
    terms = set(tokenize(text)) - STOP_WORDS
    if use_stemming:
        return {stem(term) for term in terms}
    return terms


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


def lexical_score(query: str, chunk_text: str, use_stemming: bool = False) -> float:
    """Dice similarity balances query coverage and chunk specificity.

    Repeated words do not increase the score. Common function words cannot
    cause unrelated chunks to be retrieved.
    """
    q_tokens = content_tokens(query, use_stemming)
    c_tokens = content_tokens(chunk_text, use_stemming)
    if not q_tokens or not c_tokens:
        return 0.0

    return 2 * len(q_tokens & c_tokens) / (len(q_tokens) + len(c_tokens))


def retrieve_top_k(
    query: str, chunks: list[Chunk], k: int = 3, min_score: float = 0.01, use_stemming: bool = False
) -> list[RetrievalResult]:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")

    scored = [
        RetrievalResult(chunk=chunk, score=lexical_score(query, chunk.text, use_stemming))
        for chunk in chunks
    ]
    filtered = [item for item in scored if item.score > 0 and item.score >= min_score]
    # Python's stable sort preserves corpus order for equal scores.
    filtered.sort(key=lambda item: item.score, reverse=True)
    return filtered[:k]
