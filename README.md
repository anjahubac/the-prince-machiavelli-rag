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
python -m src.cli --corpus-profile prince --strategy sentence --chunk-size 700 --sentence-overlap 1 --stem --scorer bm25 --min-coverage 0.75 --top-k 3 --title-in-chunk --question "Is it better for a prince to be loved or feared?"
```

These are the recommended settings for the book (see `evidence.md`). Without the new
flags the CLI behaves exactly like the homework (`--scorer dice --min-coverage 1.0`).

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

## Why BM25 for a book

The homework's Dice score treats every shared word equally. In *The Prince*, "prince" appears
in most chunks, so it says little about relevance; "mercenaries" or "fortresses" say a lot.
BM25 weights each word by how rare it is across chunks (IDF), rewards repeats with diminishing
returns, and normalises for chunk length. On the evaluation set it doubles top-1 accuracy.

## Evaluate

```bash
python scripts/eval_prince.py
```

Runs the 14 supported and 4 unsupported questions in `eval/prince_questions.json` across
scorer × stemming × min_coverage and prints a Markdown table (hit@1, hit@3, answered,
correctly cited chapter, correctly refused).

## Limitations

- Matching is lexical: synonyms and paraphrases fail ("keep his word" does not match the
  text's "keep faith"; "generous" does not match "liberality").
- Without `--title-in-chunk`, a question phrased like a chapter title can miss its chapter.
  With it, title words count for every chunk of the chapter, so the first chunk of the right
  chapter can be retrieved and cited even when its own text is only loosely related, and a
  question can be answered on title words alone.
- Citation selection adds every retrieved chunk that covers a not-yet-covered question word,
  so a weakly related chunk can be cited next to the right one.
- Abbreviations and verse quotations can split sentences oddly (e.g. a chunk starting with ".").
- The evaluation set is small (18 questions) and was used to pick the settings, so the
  numbers are indicative, not a held-out benchmark.

## Project structure

```text
the-prince-machiavelli-rag/
  data/pg1232.html            cached Gutenberg HTML (public domain)
  eval/prince_questions.json  gold questions
  scripts/run_tests.sh
  scripts/eval_prince.py
  src/gutenberg.py  corpus.py  chunking.py  retrieval.py  rag.py  cli.py
  tests/                      pytest suite (offline; uses tests/fixtures/mini_prince.html)
  evidence.md
```
