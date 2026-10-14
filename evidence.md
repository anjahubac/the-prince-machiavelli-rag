# Evidence: The Prince Machiavelli RAG

All outputs below were produced on 2026-10-14 from the code in this repository, with
`data/pg1232.html` downloaded the same day from
https://www.gutenberg.org/cache/epub/1232/pg1232-images.html.

## 1. Corpus

`load_corpus("prince")` returns 34 documents (276,465 characters): 5 introduction parts,
the dedication, chapters I–XXVI, and two extra works. With `--strategy sentence
--chunk-size 700 --sentence-overlap 1` this gives **676 chunks** (500 → 886 chunks,
1000 → 440 chunks).

## 2. Dice vs. BM25 on one question

Question: "Is it better for a prince to be loved or feared?" (`--stem --top-k 3`, same chunking).

| Rank | Dice (homework) | BM25 |
|---|---|---|
| 1 | `prince-ch-03-chunk-12` (0.140) | `prince-ch-17-chunk-3` (12.81) |
| 2 | `prince-ch-17-chunk-13` (0.140) | `prince-ch-03-chunk-12` (10.31) |
| 3 | `prince-ch-17-chunk-3` (0.128) | `prince-ch-17-chunk-13` (9.72) |
| Cited | `prince-ch-03-chunk-12` only | `prince-ch-17-chunk-3`, `prince-ch-03-chunk-12` |

`prince-ch-17-chunk-3` is the passage that answers the question ("it is much safer to be
feared than loved"). Dice ranked it third and did not cite it; BM25 ranks it first.

## 3. Supported question (recommended settings)

```bash
python -m src.cli --corpus-profile prince --strategy sentence --chunk-size 700 --sentence-overlap 1 --stem --scorer bm25 --min-coverage 0.75 --top-k 3 --question "Is it better for a prince to be loved or feared?"
```

```text
Scoring: scorer=bm25, min_coverage=0.75
Top-k: 3; corpus chunks: 676
...
References:
[prince-ch-17-chunk-3] Chapter XVII — Concerning Cruelty And Clemency, And Whether It Is Better To Be Loved Than Feared (prince-ch-17)
[prince-ch-03-chunk-12] Chapter III — Concerning Mixed Principalities (prince-ch-03)

Refused: False
Citations: prince-ch-17-chunk-3, prince-ch-03-chunk-12
```

The second citation is weak: it is cited only because it contains "better", a question word
the first chunk lacks. See Limitations in `README.md`.

## 4. Unsupported question (same settings)

```bash
python -m src.cli --corpus-profile prince --strategy sentence --chunk-size 700 --sentence-overlap 1 --stem --scorer bm25 --min-coverage 0.75 --top-k 3 --question "Who won the world cup in 1958?"
```

```text
Retrieved context (descending score):
1. prince-intro-3-chunk-13 | score=4.729357 | Introduction (Marriott) — Office — Æt. 25-43—1494-1512
2. prince-ch-07-chunk-27 | score=4.729357 | Chapter VII — ...
3. prince-ch-07-chunk-31 | score=4.729357 | Chapter VII — ...

Answer:
I do not have enough grounded evidence in the provided corpus to answer that question.

Refused: True
Citations: None
```

BM25 always finds something (here, chunks containing "won"), so the coverage check, not
retrieval, is what produces the refusal.

## 5. Evaluation grid

`python scripts/eval_prince.py` (676 chunks, top_k=3; 14 supported, 4 unsupported questions):

| scorer | stem | min_coverage | hit@1 | hit@3 | answered | correct | refused |
|---|---|---|---|---|---|---|---|
| dice | off | 1.0 | 4/14 | 7/14 | 6/14 | 6/14 | 4/4 |
| dice | off | 0.75 | 4/14 | 7/14 | 10/14 | 6/14 | 4/4 |
| dice | off | 0.5 | 4/14 | 7/14 | 14/14 | 6/14 | 3/4 |
| dice | on | 1.0 | 4/14 | 8/14 | 8/14 | 7/14 | 4/4 |
| dice | on | 0.75 | 4/14 | 8/14 | 11/14 | 7/14 | 4/4 |
| dice | on | 0.5 | 4/14 | 8/14 | 14/14 | 7/14 | 2/4 |
| bm25 | off | 1.0 | 8/14 | 10/14 | 9/14 | 7/14 | 4/4 |
| bm25 | off | 0.75 | 8/14 | 10/14 | 12/14 | 8/14 | 4/4 |
| bm25 | off | 0.5 | 8/14 | 10/14 | 14/14 | 8/14 | 2/4 |
| bm25 | on | 1.0 | 8/14 | 11/14 | 10/14 | 8/14 | 4/4 |
| **bm25** | **on** | **0.75** | **8/14** | **11/14** | **13/14** | **9/14** | **4/4** |
| bm25 | on | 0.5 | 8/14 | 11/14 | 14/14 | 9/14 | 2/4 |

- *hit@k*: the expected chapter is among the top k chunks. *correct*: answered and a citation
  comes from the expected chapter. *refused*: unsupported questions correctly refused.
- BM25 doubles hit@1 (4 → 8). Stemming adds one hit@3. `min_coverage=0.75` answers more
  paraphrased questions without answering any unsupported one; 0.5 starts answering them.
- Chosen settings: BM25, stemming on, `min_coverage=0.75`.

Misses with the chosen settings (from a per-question run):

| Question | Expected | What happened |
|---|---|---|
| Should a prince keep his word when it is against his interest? | ch-18 | Text says "keep faith"; not retrieved |
| How did Cesare Borgia gain and lose his state? | ch-07 | Borgia appears in many chapters; ch-07 not in top 3 |
| Is it better for a prince to be generous or miserly? | ch-16 | ch-16 at rank 2 but not cited (ch-15 covered the words first) |
| Why did the princes of Italy lose their states? | ch-24 | Words match the chapter title, which is not in the chunk text |
| What are auxiliary troops? | ch-13 | Retrieved, but refused: the text says "soldiery", not "troops" |

## 6. Tests

`./scripts/run_tests.sh` → **76 passed** (57 from the homework, 19 new). The new tests run
offline against `tests/fixtures/mini_prince.html` and cover: section order and ids, removal of
Gutenberg header/footer/license and table of contents, footnote markers, titles, reading the
cache without network, the `prince` profile on the real book, BM25 rare-term weighting,
length normalisation and repeat saturation, `min_coverage` (answers at 0.6, refuses at 1.0,
rejects invalid values), and the new CLI output lines.

## 7. Next steps

Sections 1–6 above are the baseline. Each step below is opt-in and documented in its own
section, with the same 14 + 4 question set as before so the numbers are comparable.

- [x] Prepend the chapter title to each chunk's text for retrieval (fixes the ch-24 miss). → §8
- [x] Cite only chunks above a score threshold, not every chunk that adds a question word. → §9
- [x] Add a small synonym list or embeddings for paraphrases (keep word / keep faith). → §10
- [ ] Grow the question set and keep part of it held out when choosing settings.

## 8. Step 1 — chapter titles in retrieval (`--title-in-chunk`)

`build_chunks(..., include_title=True)` stores the document title in `Chunk.search_prefix`.
Retrieval (Dice and BM25) and the coverage check read `Chunk.search_text` (title + text);
the answer still quotes only `Chunk.text`, and the title appears in the References line as
before. The chunk count is unchanged (676).

ch-24 question, recommended settings (`--stem --scorer bm25 --min-coverage 0.75 --top-k 3`):

```text
# without --title-in-chunk
Matching: stemming=on
1. prince-ch-03-chunk-5 | score=8.101192 | Chapter III — Concerning Mixed Principalities
2. prince-ch-15-chunk-5 | score=7.918907 | Chapter XV — Concerning Things For Which Men, And Especially Princes, Are Praised Or Blamed
3. prince-ch-14-chunk-0 | score=7.698934 | Chapter XIV — That Which Concerns A Prince On The Subject Of The Art Of War
Refused: False
Citations: prince-ch-03-chunk-5, prince-ch-15-chunk-5

# with --title-in-chunk
Matching: stemming=on, title_in_chunk=on
1. prince-ch-24-chunk-1 | score=8.840370 | Chapter XXIV — Why The Princes Of Italy Have Lost Their States
2. prince-ch-15-chunk-5 | score=8.103736 | Chapter XV — Concerning Things For Which Men, And Especially Princes, Are Praised Or Blamed
3. prince-ch-03-chunk-5 | score=7.890301 | Chapter III — Concerning Mixed Principalities
Refused: False
Citations: prince-ch-24-chunk-1
```

`python scripts/eval_prince.py` now adds a `title` column (24 rows; full table in the script
output). The rows that matter:

| scorer | stem | title | min_coverage | hit@1 | hit@3 | answered | correct | refused |
|---|---|---|---|---|---|---|---|---|
| dice | on | off | 0.75 | 4/14 | 8/14 | 11/14 | 7/14 | 4/4 |
| dice | on | on | 0.75 | 9/14 | 12/14 | 14/14 | 12/14 | 4/4 |
| bm25 | on | off | 0.75 | 8/14 | 11/14 | 13/14 | 9/14 | 4/4 |
| **bm25** | **on** | **on** | **0.75** | **10/14** | **13/14** | **13/14** | **11/14** | **4/4** |

- Before → after for the recommended BM25 settings: hit@1 8 → 10, hit@3 11 → 13,
  correct 9 → 11, refused 4/4 → 4/4.
- Fixed: ch-24 (title "Why The Princes Of Italy Have Lost Their States") and ch-18 (title
  "...Should Keep Faith" supplies "keep"; the chapter is now retrieved and cited).
- Titles help Dice even more (correct 7 → 12), and Dice + stem + title is now ahead of BM25
  on *correct* (12 vs 11). Several questions in this set are phrased close to chapter titles,
  so this set flatters the title option. The final choice is left to the tuning/held-out
  split in a later step.

Remaining misses (bm25, stem, title, 0.75; `python scripts/eval_prince.py --misses`):

| Question | Expected | Refused | Top-3 docs | Citations |
|---|---|---|---|---|
| How did Cesare Borgia gain and lose his state? | prince-ch-07 | False | prince-ch-13, prince-intro-3, prince-ch-20 | prince-ch-13-chunk-6, prince-intro-3-chunk-5 |
| Is it better for a prince to be generous or miserly? | prince-ch-16 | False | prince-ch-15, prince-ch-16, extra-castruccio | prince-ch-15-chunk-3 |
| What are auxiliary troops? | prince-ch-13 | True | prince-ch-13, prince-ch-13, prince-ch-13 | None |

Tests: 81 passed (5 new: title in `search_text` but not in `text`, title match ranks its
chunk for Dice and BM25, title words count for coverage but are not quoted, CLI flag).

## 9. Step 2 — citation threshold (`--min-citation-ratio`)

`build_grounded_answer(..., min_citation_ratio=r)` keeps as evidence only the retrieved chunks
whose score is ≥ r × the top chunk's score. Only those chunks can be cited, and only their
words count toward `min_coverage`, so the system never answers on evidence it does not show.
All top-k chunks are still printed as retrieved context. Default 0.0 = old behaviour.

Loved/feared question, `--stem --scorer bm25 --min-coverage 0.75 --top-k 3`, no title:

```text
1. prince-ch-17-chunk-3 | score=12.807098 | Chapter XVII — ...Whether It Is Better To Be Loved Than Feared
2. prince-ch-03-chunk-12 | score=10.307162 | Chapter III — Concerning Mixed Principalities
3. prince-ch-17-chunk-13 | score=9.724769 | Chapter XVII — ...Whether It Is Better To Be Loved Than Feared

--min-citation-ratio 0    Citations: prince-ch-17-chunk-3, prince-ch-03-chunk-12
--min-citation-ratio 0.8  Citations: prince-ch-17-chunk-3, prince-ch-03-chunk-12   (10.31 / 12.81 = 0.805)
--min-citation-ratio 0.9  Citations: prince-ch-17-chunk-3
```

With `--title-in-chunk` (step 1) the weak chunk is no longer retrieved at all: the top 3 are
`prince-ch-17-chunk-3`, `-13`, `-5`, and the citations are `prince-ch-17-chunk-3, prince-ch-17-chunk-13`.

Sweep from `python scripts/eval_prince.py` (stem on, min_coverage 0.75; *off-chapter* =
citations not from the expected chapter / all citations on answered supported questions):

| scorer | title | min_citation_ratio | answered | correct | refused | off-chapter |
|---|---|---|---|---|---|---|
| dice | on | 0.0 | 14/14 | 12/14 | 4/4 | 10/26 |
| dice | on | 0.5 | 14/14 | 12/14 | 4/4 | 10/26 |
| dice | on | 0.8 | 11/14 | 10/14 | 4/4 | 5/17 |
| dice | on | 0.9 | 7/14 | 6/14 | 4/4 | 2/9 |
| bm25 | off | 0.0 | 13/14 | 9/14 | 4/4 | 12/22 |
| bm25 | off | 0.8 | 13/14 | 9/14 | 4/4 | 11/21 |
| bm25 | off | 0.9 | 9/14 | 6/14 | 4/4 | 4/10 |
| **bm25** | **on** | **0.0** | **13/14** | **11/14** | **4/4** | **5/18** |
| bm25 | on | 0.5 | 13/14 | 11/14 | 4/4 | 5/18 |
| bm25 | on | 0.8 | 12/14 | 10/14 | 4/4 | 5/16 |
| bm25 | on | 0.9 | 10/14 | 8/14 | 4/4 | 2/11 |

- The threshold does what it says: off-chapter citations fall (bm25 + title: 5/18 → 2/11 at 0.9).
- But it costs correct answers every time it bites (11 → 10 at 0.8, 11 → 8 at 0.9). Within one
  query, right and wrong chunks score close together (e.g. ch-22's right chunk is at 0.87 of
  the top, ch-18's at 0.90), so a cut that removes weak chunks also removes right ones, and the
  coverage check then refuses. 0.5 never bites with top_k=3.
- Titles (step 1) reduce off-chapter citations more cheaply (bm25: 12/22 → 5/18) while raising
  *correct*.
- Decision: keep the option (default 0.0), but do not add it to the recommended settings.

Tests: 88 passed (7 new: weak chunk dropped but still listed as retrieved, the threshold
limits coverage evidence and can cause refusal, 0 and 1 keep the top chunk, invalid values
rejected, CLI line and output).

## 10. Step 3 — synonym map (`--expand-synonyms`)

`SYNONYMS` in `src/retrieval.py` maps 7 modern question words to the translation's wording
(word → faith; generous, generosity → liberal, liberality; miserly, stingy → mean, meanness;
troops → soldiery; ministers → secretaries). With `expand_synonyms=True`,
`query_term_groups` turns each question word into a group {word, synonyms…}. A group counts
as **one** question word everywhere: Dice counts matched groups, BM25 adds the best-scoring
form of each group (not the sum), and coverage / citation selection count covered groups.
With expansion off every group is {word}, so all scores are identical to before. Chunk text
is never changed. No embeddings; still standard library only.

**Caveat:** the map was written by looking at the misses of the 14 questions below, so these
numbers are tuning-set numbers and flatter the option. The held-out check is in step 4.

Before/after (`--stem --scorer bm25 --min-coverage 0.75 --top-k 3 --title-in-chunk`):

| Question | Without synonyms | With `--expand-synonyms` |
|---|---|---|
| Is it better for a prince to be generous or miserly? (ch-16) | top-1 `prince-ch-15-chunk-3`; cited ch-15 only | top-1 `prince-ch-16-chunk-10`; cited `prince-ch-16-chunk-10` ✓ |
| What are auxiliary troops? (ch-13) | top-3 all ch-13, **refused** ("troops" not covered) | cited `prince-ch-13-chunk-6` ✓ |
| How should a prince choose his ministers? (ch-22) | top-1 `prince-ch-04-chunk-1`; cited ch-04 and `prince-ch-22-chunk-0` | top-1/2 ch-22; cited `prince-ch-22-chunk-0` only ✓ |
| Should a prince keep his word when it is against his interest? (ch-18) | top-1 `prince-intro-5-chunk-10`; cited intro-5 and `prince-ch-18-chunk-0` ✓ | top-3 all ch-18, but **refused** ✗ |

The ch-18 regression: with synonyms, BM25 ranks three ch-18 chunks first, but none of them
contains "interest" (and the coverage check finds 3 of 5 question words < 0.75). Without
synonyms, an introduction chunk supplied those words and ch-18's first chunk was cited
alongside it. The retrieval is better; the coverage check is what fails. This is a safe
failure (refusal), not a wrong answer.

Grid rows (`python scripts/eval_prince.py`, 48 rows in full; stem on, min_coverage 0.75):

| scorer | title | synonyms | hit@1 | hit@3 | answered | correct | refused | off-chapter |
|---|---|---|---|---|---|---|---|---|
| dice | off | off | 4/14 | 8/14 | 11/14 | 7/14 | 4/4 | 12/19 |
| dice | off | on | 5/14 | 9/14 | 12/14 | 8/14 | 4/4 | 12/20 |
| dice | on | off | 9/14 | 12/14 | 14/14 | 12/14 | 4/4 | 10/26 |
| dice | on | on | 12/14 | 13/14 | 14/14 | 13/14 | 4/4 | 4/23 |
| bm25 | off | off | 8/14 | 11/14 | 13/14 | 9/14 | 4/4 | 12/22 |
| bm25 | off | on | 10/14 | 12/14 | 12/14 | 10/14 | 4/4 | 7/18 |
| bm25 | on | off | 10/14 | 13/14 | 13/14 | 11/14 | 4/4 | 5/18 |
| **bm25** | **on** | **on** | **13/14** | **13/14** | **13/14** | **12/14** | **4/4** | **2/16** |

- bm25 + stem + title: adding synonyms gives hit@1 10 → 13, correct 11 → 12, off-chapter
  citations 5/18 → 2/16, unsupported still 4/4 refused.
- Dice + stem + title + synonyms answers one more question correctly (13 vs 12) but cites
  far more off-chapter chunks (4/23 vs 2/16) and has lower hit@1 (12 vs 13).
- At min_coverage 0.5 every setting answers 2 of the 4 unsupported questions, as before.

Remaining misses (bm25, stem, title, synonyms, 0.75; `--misses`):

| Question | Expected | Refused | Top-3 docs | Citations |
|---|---|---|---|---|
| Should a prince keep his word when it is against his interest? | prince-ch-18 | True | prince-ch-18, prince-ch-18, prince-ch-18 | None |
| How did Cesare Borgia gain and lose his state? | prince-ch-07 | False | prince-ch-13, prince-intro-3, prince-ch-20 | prince-ch-13-chunk-6, prince-intro-3-chunk-5 |

Tests: 96 passed (8 new: groups equal content tokens when off, listed words expand after
stemming, synonym retrieval for Dice and BM25 only when enabled, a word and its synonym
count once (Dice and BM25 max not sum), auxiliary-troops answer, refusal of unsupported
questions unchanged, CLI flag).
