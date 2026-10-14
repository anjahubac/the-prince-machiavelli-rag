from collections import Counter
from dataclasses import dataclass
import math
import re

from .chunking import Chunk


SCORERS = ("dice", "bm25")
BM25_K1 = 1.5
BM25_B = 0.75


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


def content_terms(text: str, use_stemming: bool = False) -> list[str]:
    """Content terms in order, repeats kept; optionally stemmed."""
    terms = [token for token in tokenize(text) if token not in STOP_WORDS]
    if use_stemming:
        return [stem(term) for term in terms]
    return terms


def content_tokens(text: str, use_stemming: bool = False) -> set[str]:
    """Keep distinct content terms; optionally stem them. No semantic matching."""
    return set(content_terms(text, use_stemming))


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


class BM25Index:
    """Okapi BM25 over chunk texts.

    Unlike Dice, a term is weighted by how rare it is across chunks (IDF), so
    "prince" in a book about princes counts far less than "mercenaries".
    Repeats count, with diminishing returns (k1), and long chunks are
    penalised relative to the average chunk length (b).
    """

    def __init__(
        self, chunks: list[Chunk], use_stemming: bool = False, k1: float = BM25_K1, b: float = BM25_B
    ) -> None:
        self.use_stemming = use_stemming
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(content_terms(chunk.search_text, use_stemming)) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avg_length = sum(self.lengths) / len(chunks) if chunks else 0.0
        doc_freq = Counter(term for counts in self.term_counts for term in counts)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        q_terms = content_tokens(query, self.use_stemming)
        results = []
        for counts, length in zip(self.term_counts, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length) if self.avg_length else self.k1
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            results.append(score)
        return results


def retrieve_top_k(
    query: str,
    chunks: list[Chunk],
    k: int = 3,
    min_score: float = 0.01,
    use_stemming: bool = False,
    scorer: str = "dice",
) -> list[RetrievalResult]:
    """Rank chunks by Dice (scores 0-1) or BM25 (unbounded, comparable only within one query)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {SCORERS}")

    if scorer == "bm25":
        bm25_scores = BM25Index(chunks, use_stemming).scores(query)
        scored = [RetrievalResult(chunk=chunk, score=score) for chunk, score in zip(chunks, bm25_scores)]
    else:
        scored = [
            RetrievalResult(chunk=chunk, score=lexical_score(query, chunk.search_text, use_stemming))
            for chunk in chunks
        ]
    filtered = [item for item in scored if item.score > 0 and item.score >= min_score]
    # Python's stable sort preserves corpus order for equal scores.
    filtered.sort(key=lambda item: item.score, reverse=True)
    return filtered[:k]
