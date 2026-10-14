# The Prince Machiavelli RAG

Retrieval augmented generation over Niccolò Machiavelli's *The Prince*
([Project Gutenberg #1232](https://www.gutenberg.org/cache/epub/1232/pg1232-images.html),
W. K. Marriott translation).

This project starts from the Week 6 RAG homework pipeline
(`corpus → chunking → retrieval → rag → cli`): Python 3.9, standard library only,
extractive answers with chunk citations, and refusal when evidence is missing.
It adds what a real book needs: an HTML loader, one document per chapter,
BM25 ranking, and a tunable refusal threshold.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run_tests.sh
```

## Ask the book a question

```bash
python -m src.cli --corpus-profile prince --strategy sentence --chunk-size 700 --sentence-overlap 1 --stem --scorer dice --title-in-chunk --expand-synonyms --min-coverage 0.75 --top-k 3 --question "Is it better for a prince to be loved or feared?"
```

These settings were picked automatically on the **tuning** questions only (see
`python scripts/eval_prince.py` and `evidence.md` §11). On the 16 held-out questions they
answer 7 correctly and refuse 3 of 4 unsupported questions, so expect many refusals on
questions phrased unlike the book.

Caveat: with `--title-in-chunk`, every chunk of a chapter shares the title words, and Dice
then prefers short chunks of that chapter. For the question above it cites the right chapter
but a less direct passage (`prince-ch-17-chunk-13`, on Scipio). `--scorer bm25` (ranked third
on tuning, 6/16 correct on held-out) quotes `prince-ch-17-chunk-3` ("it is much safer to be
feared than loved") first. The evaluation only checks the chapter, not the passage.

Without any of the new flags the CLI behaves exactly like the homework
(`--scorer dice --min-coverage 1.0`).

## How it works

| Stage | File | What it does for the book |
|---|---|---|
| Load | `src/gutenberg.py` | Reads `data/pg1232.html` (downloaded once from Gutenberg if missing), drops the Gutenberg header, footer, license and table of contents, and returns one `Document` per section. |
| Corpus | `src/corpus.py` | `load_corpus("prince")` profile. |
| Chunk | `src/chunking.py` | `--strategy sentence` packs whole sentences into ≤700-character chunks within one chapter, repeating one sentence between chunks. `--title-in-chunk` lets the chapter title count when matching (it is not quoted in the answer). |
| Rank | `src/retrieval.py` | `--scorer bm25`: rare words count more than common ones and long chunks are penalised. `dice` (homework default) is kept. |
| Answer | `src/rag.py` | Quotes the supporting chunks with `[chunk-id]` citations, or refuses when fewer than `--min-coverage` of the question's content words appear in the retrieved text. |

Document ids:

| doc_id | Section |
|---|---|
| `prince-intro-1` … `prince-intro-5` | Marriott's introduction (Machiavelli's life and works) |
| `prince-dedication` | Dedication to Lorenzo de' Medici |
| `prince-ch-01` … `prince-ch-26` | The 26 chapters of *The Prince* |
| `extra-valentino`, `extra-castruccio` | Two shorter Machiavelli works included in the same Gutenberg file |

Chunk ids are `<doc_id>-chunk-<n>`, e.g. `prince-ch-17-chunk-3`. They depend on the chunking
settings, so record the settings with any citation.

## Opt-in flags

All of these are off by default, so the homework behaviour and its tests are unchanged.

| Flag | Python keyword | Effect |
|---|---|---|
| `--scorer bm25` | `scorer="bm25"` | BM25 ranking instead of Dice. |
| `--stem` | `use_stemming=True` | Light suffix stemming (helps = help). |
| `--min-coverage 0.75` | `min_coverage=0.75` | Answer when ≥75% of the question's content words are in the retrieved text. |
| `--title-in-chunk` | `build_chunks(..., include_title=True)` | The chapter title counts for retrieval and coverage (`Chunk.search_text`), but only the chunk text is quoted. |
| `--expand-synonyms` | `expand_synonyms=True` | Question words also match the translation's older words, from a hand-written map (`SYNONYMS` in `src/retrieval.py`): word → faith, generous → liberal/liberality, miserly/stingy → mean/meanness, troops → soldiery, ministers → secretaries. A word and its synonym count as one question word. Applied to the question only. |
| `--min-citation-ratio 0.9` | `min_citation_ratio=0.9` | Only chunks scoring ≥ 90% of the top chunk can be cited, and only their words count for `--min-coverage`. Weaker chunks are still listed as retrieved context. Not part of the recommended settings (see `evidence.md` §9). |

## Why BM25 for a book

The homework's Dice score treats every shared word equally. In *The Prince*, "prince" appears
in most chunks, so it says little about relevance; "mercenaries" or "fortresses" say a lot.
BM25 weights each word by how rare it is across chunks (IDF), rewards repeats with diminishing
returns, and normalises for chunk length. On the original 14 questions it doubles top-1
accuracy over Dice. Once chapter titles and synonyms are added, Dice catches up on the
chapter-level metrics (one more correct answer on tuning and on held-out, lower hit@1), which
is why the automatic pick is Dice; see the caveat above.

## Evaluate

```bash
python scripts/eval_prince.py            # add --misses to list every miss of the picked setting
```

`eval/prince_questions.json` has two splits:

| Split | Supported | Unsupported | Used for |
|---|---|---|---|
| `tuning` | 14 | 4 | designing the new options and picking settings |
| `held_out` | 16 | 4 | reporting only; written before the new options were implemented |

The script scores 72 settings on the tuning split (scorer × stemming × title × synonyms ×
min_coverage, plus a citation-ratio sweep), picks one with a fixed rule (most correct answers
+ correct refusals; ties → fewer off-chapter citations, then hit@1, then fewer options), and
only then runs the held-out split for the pick and four reference settings. It takes about
40 seconds.

## Limitations

- Matching is lexical. `--expand-synonyms` covers only the handful of words in its map, which
  was written from the misses on the tuning questions; any other paraphrase still fails. The
  map is context-free: "word" always also matches "faith", and with `--stem` "mean" also
  matches "means". There are no embeddings (standard library only).
- Without `--title-in-chunk`, a question phrased like a chapter title can miss its chapter.
  With it, title words count for every chunk of the chapter, so the first chunk of the right
  chapter can be retrieved and cited even when its own text is only loosely related, and a
  question can be answered on title words alone.
- Citation selection adds every retrieved chunk that covers a not-yet-covered question word,
  so a weakly related chunk can be cited next to the right one. `--min-citation-ratio` can
  drop it, but BM25 scores of right and wrong chunks are often within 10–20% of each other,
  so a threshold high enough to drop weak chunks also drops right ones and causes refusals.
- Abbreviations and verse quotations can split sentences oddly (e.g. a chunk starting with ".").
- The coverage check causes many held-out misses: the right chapter is in the top 3 but the
  retrieved text lacks everyday question words (e.g. "his", "used", "gift", "Machiavelli"),
  so the system refuses. Lowering `--min-coverage` to 0.5 answers them but also answers unsupported
  questions. Common words count as much as rare ones in this check, so title words like
  "many" can push an unsupported question over the threshold.
- The evaluation is small (30 + 8 questions; 16 + 4 held out) and chapter-level only: a
  "correct" answer cites the right chapter, not necessarily the passage that answers.
  Differences of one or two questions are within noise.

## Project structure

```text
the-prince-machiavelli-rag/
  data/pg1232.html            cached Gutenberg HTML (public domain)
  eval/prince_questions.json  gold questions (tuning and held-out splits)
  scripts/run_tests.sh
  scripts/eval_prince.py
  src/gutenberg.py  corpus.py  chunking.py  retrieval.py  rag.py  cli.py
  tests/                      pytest suite (offline; uses tests/fixtures/mini_prince.html)
  evidence.md
```
