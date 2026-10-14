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
- [x] Grow the question set and keep part of it held out when choosing settings. → §11

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

## 11. Step 4 — tuning / held-out split and final settings

### Method

- `eval/prince_questions.json` now has `tuning` (the original 14 supported + 4 unsupported,
  which shaped steps 1–3) and `held_out` (16 supported + 4 unsupported, new). Total 30 + 8.
- The held-out questions were written **before** steps 1–3 were implemented, from the chapter
  list only, and kept out of the repository until this step. Their gold chapters were checked
  by plain text search of the book (e.g. "Oliverotto" in ch-08, "neutral" in ch-21, "Moses" and
  "Theseus" in ch-06), not with the retriever. The unsupported ones were checked to be absent
  ("America", "legion", "pasta" do not occur; "president" occurs once in ch-07 in another sense).
  ch-15's question uses a common modern phrasing ("how not to be good"); Marriott writes "how
  to do wrong", so it is a deliberate paraphrase.
- `scripts/eval_prince.py` scores 72 settings on the tuning split only, then picks one with a
  rule fixed in code before the held-out split was ever run: **most (correct + refused) on
  tuning; ties → lowest off-chapter share, then most hit@1, then earliest in grid order.**
  The held-out split is run afterwards for the pick and four reference settings.

### Picked on tuning

| rank | setting | hit@1 | hit@3 | answered | correct | refused | off-chapter |
|---|---|---|---|---|---|---|---|
| **1** | **dice, stem on, title on, synonyms on, min_coverage 0.75, min_citation_ratio 0.0** | **12/14** | **13/14** | **14/14** | **13/14** | **4/4** | **4/23** |
| 2 | dice, stem on, title on, synonyms on, min_coverage 0.75, min_citation_ratio 0.5 | 12/14 | 13/14 | 14/14 | 13/14 | 4/4 | 4/23 |
| 3 | bm25, stem on, title on, synonyms on, min_coverage 0.75, min_citation_ratio 0.0 | 13/14 | 13/14 | 13/14 | 12/14 | 4/4 | 2/16 |
| 4 | bm25, stem on, title on, synonyms on, min_coverage 0.75, min_citation_ratio 0.5 | 13/14 | 13/14 | 13/14 | 12/14 | 4/4 | 2/16 |
| 5 | dice, stem off, title on, synonyms on, min_coverage 0.75, min_citation_ratio 0.0 | 9/14 | 12/14 | 13/14 | 12/14 | 4/4 | 5/21 |

Recommended command (the pick):

```bash
python -m src.cli --corpus-profile prince --strategy sentence --chunk-size 700 --sentence-overlap 1 --stem --scorer dice --title-in-chunk --expand-synonyms --min-coverage 0.75 --top-k 3 --question "..."
```

### Held-out results (not used for picking)

| setting | hit@1 | hit@3 | answered | correct | refused | off-chapter |
|---|---|---|---|---|---|---|
| homework default (dice, no stem, min_coverage 1.0) | 4/16 | 9/16 | 2/16 | 0/16 | 4/4 | 5/5 |
| baseline before steps 1–3 (bm25, stem, 0.75) | 8/16 | 11/16 | 10/16 | 6/16 | 4/4 | 12/20 |
| + title | 11/16 | 12/16 | 9/16 | 6/16 | 3/4 | 8/17 |
| + title + synonyms | 11/16 | 12/16 | 9/16 | 6/16 | 3/4 | 8/17 |
| **picked: dice, stem, title, synonyms, 0.75** | **9/16** | **11/16** | **11/16** | **7/16** | **3/4** | **11/22** |

Before → after, both splits:

| | tuning correct | tuning refused | held-out hit@1 | held-out correct | held-out refused |
|---|---|---|---|---|---|
| Baseline (bm25, stem, 0.75) | 9/14 | 4/4 | 8/16 | 6/16 | 4/4 |
| Picked (dice, stem, title, synonyms, 0.75) | 13/14 | 4/4 | 9/16 | 7/16 | 3/4 |

What the held-out split says:

- **Most of the tuning gain does not carry over.** On tuning the steps took *correct* from
  9/14 to 13/14; on held-out from 6/16 to 7/16, and one unsupported question is now answered
  (3/4 refused, was 4/4). With 16 + 4 questions, a one-question difference is noise.
- **Titles help retrieval on new questions** (bm25 hit@1 8 → 11, hit@3 11 → 12), but not the
  final answer (correct 6 → 6): the extra hits end in refusals.
- **The synonym map does nothing on held-out** (identical rows): none of its 7 words occur in
  the new questions. It fixed exactly the tuning misses it was written for, as expected for a
  hand-written list.
- **The coverage check is a main bottleneck.** Several refusals have the expected chapter in
  the top 3, but fewer than 75% of the question's words are in those chunks. The uncovered
  words are mostly everyday words that are not in the stop list, plus the author's name:

  | Expected | Setting | Uncovered question words (stemmed) | Coverage |
  |---|---|---|---|
  | ch-05 | both | conqueror, used | 5/7 |
  | ch-26 | both | call, machiavelli | 4/6 |
  | dedication | both | gift, offer | 4/6 |
  | ch-03 | bm25 + title + synonyms | conqueror, new, own, provinc, whos | 5/10 |
  | ch-10 | bm25 + title + synonyms | prepar, withstand | 4/6 |
  | ch-21 | bm25 + title + synonyms | go, his, stay | 5/8 |

  That is 3 of the picked setting's 9 held-out misses and 6 of the BM25 setting's 10.
- **Title words can cause a false answer.** With bm25 + title, "How many soldiers served in a
  Roman legion?" is answered: ch-12's title "How **Many** Kinds Of Soldiery…" supplies "many",
  giving 4 of 5 question words. The picked Dice setting instead answers "Who was the first
  president of the United States?" from `prince-ch-04-chunk-9` and `prince-ch-14-chunk-1`
  ("first", "united", "states" = 3 of 4 words). In both, common words count as much as rare
  ones in the coverage check.

Held-out misses of the picked setting (`python scripts/eval_prince.py --misses`):

| Question | Expected | Refused | Top-3 docs | Citations |
|---|---|---|---|---|
| Is it hard for a prince to hold a state he inherited? | prince-ch-02 | False | prince-ch-15, prince-ch-15, prince-ch-12 | prince-ch-15-chunk-2, prince-ch-15-chunk-4, prince-ch-12-chunk-2 |
| What should a conqueror do with a free city that is used to its own laws? | prince-ch-05 | True | prince-ch-05, prince-ch-05, prince-ch-05 | None |
| Was the cruelty of Oliverotto da Fermo well used? | prince-ch-08 | False | extra-valentino, prince-ch-19, extra-valentino | extra-valentino-chunk-13, prince-ch-19-chunk-19 |
| How should a prince prepare his city to withstand a siege? | prince-ch-10 | True | prince-ch-15, prince-ch-09, prince-ch-09 | None |
| How did the Church gain so much temporal power? | prince-ch-11 | False | prince-ch-03, prince-ch-21, prince-ch-12 | prince-ch-03-chunk-29, prince-ch-21-chunk-1 |
| Why must a prince learn how not to be good? | prince-ch-15 | False | prince-ch-23, prince-ch-15, prince-ch-18 | prince-ch-23-chunk-9 |
| Should a prince stay neutral when two of his neighbours go to war? | prince-ch-21 | True | prince-ch-14, prince-ch-14, prince-ch-15 | None |
| Why does Machiavelli call for a leader to free Italy from the barbarians? | prince-ch-26 | True | prince-ch-26, prince-ch-26, prince-ch-26 | None |
| What gift does Machiavelli offer to Lorenzo de' Medici? | prince-dedication | True | prince-dedication, prince-intro-1, prince-intro-4 | None |
| Who was the first president of the United States? | (unsupported) | False | | answered |

Tuning misses of the picked setting: only "How did Cesare Borgia gain and lose his state?"
(ch-07; top 3 = ch-13, ch-04, ch-03).

For comparison, the bm25 + title + synonyms setting (rank 3) misses 10 held-out questions:
7 refusals (6 with the expected chapter in the top 3, listed above; ch-08 with three
`extra-valentino` chunks) and 3 answers citing other chapters (ch-02, ch-11, ch-15).

### Spot check: the metric is chapter-level

Loved/feared question with the picked settings:

```text
Matching: stemming=on, title_in_chunk=on, synonyms=on
Scoring: scorer=dice, min_coverage=0.75
1. prince-ch-17-chunk-13 | score=0.156863 | Chapter XVII — ...Whether It Is Better To Be Loved Than Feared
2. prince-ch-17-chunk-0 | score=0.142857 | Chapter XVII — ...
3. prince-ch-17-chunk-8 | score=0.140351 | Chapter XVII — ...
Refused: False
Citations: prince-ch-17-chunk-13
```

All three are from ch-17, so this counts as correct, but the cited chunk is about Scipio, not
the passage that answers ("it is much safer to be feared than loved", `prince-ch-17-chunk-3`).
With titles on, every ch-17 chunk shares "better", "loved", "feared" through the title, and
Dice (which divides by the number of distinct words in the chunk) then prefers short chunks.
BM25 + title cites `prince-ch-17-chunk-3` first (§9). The selection rule cannot see this,
because the gold labels are chapters, not passages. I kept the rule's pick as the
recommendation because that was the agreed method, and flagged BM25 + title + synonyms as the
alternative in `README.md`.

### Tests

`./scripts/run_tests.sh` → **100 passed** (76 baseline + 24 new across the four steps; 4 new
here: both splits present with 30 + 8 questions, no question shared between splits, every gold
doc_id is a section of the book, and the selection rule's ordering).

### Next steps (not done)

- Passage-level gold labels (which chunk answers), so the evaluation can see the ch-17 problem.
- An IDF-weighted coverage check, so common words ("many", "first", "his", "used") neither
  carry an unsupported question over the threshold nor block a supported one, and rare
  question words count more.
- Match titles only once per chapter (e.g. a small bonus to the chapter's best chunk) instead
  of adding the title to every chunk.
- More held-out questions; 16 + 4 is too few to separate settings that differ by one answer.

<details>
<summary>Full tuning-split grids from <code>python scripts/eval_prince.py</code></summary>

#### Tuning split: retrieval grid (min_citation_ratio 0)

| scorer | stem | title | synonyms | min_coverage | hit@1 | hit@3 | answered | correct | refused | off-chapter |
|---|---|---|---|---|---|---|---|---|---|---|
| dice | off | off | off | 1.0 | 4/14 | 7/14 | 6/14 | 6/14 | 4/4 | 3/10 |
| dice | off | off | off | 0.75 | 4/14 | 7/14 | 10/14 | 6/14 | 4/4 | 11/18 |
| dice | off | off | off | 0.5 | 4/14 | 7/14 | 14/14 | 6/14 | 3/4 | 19/26 |
| dice | off | off | on | 1.0 | 5/14 | 8/14 | 6/14 | 6/14 | 4/4 | 3/10 |
| dice | off | off | on | 0.75 | 5/14 | 8/14 | 11/14 | 7/14 | 4/4 | 11/19 |
| dice | off | off | on | 0.5 | 5/14 | 8/14 | 14/14 | 7/14 | 3/4 | 17/25 |
| dice | off | on | off | 1.0 | 7/14 | 9/14 | 7/14 | 6/14 | 4/4 | 3/10 |
| dice | off | on | off | 0.75 | 7/14 | 9/14 | 12/14 | 9/14 | 4/4 | 8/20 |
| dice | off | on | off | 0.5 | 7/14 | 9/14 | 14/14 | 9/14 | 2/4 | 11/23 |
| dice | off | on | on | 1.0 | 9/14 | 12/14 | 6/14 | 6/14 | 4/4 | 1/8 |
| dice | off | on | on | 0.75 | 9/14 | 12/14 | 13/14 | 12/14 | 4/4 | 5/21 |
| dice | off | on | on | 0.5 | 9/14 | 12/14 | 14/14 | 12/14 | 2/4 | 6/22 |
| dice | on | off | off | 1.0 | 4/14 | 8/14 | 8/14 | 7/14 | 4/4 | 5/12 |
| dice | on | off | off | 0.75 | 4/14 | 8/14 | 11/14 | 7/14 | 4/4 | 12/19 |
| dice | on | off | off | 0.5 | 4/14 | 8/14 | 14/14 | 7/14 | 2/4 | 18/25 |
| dice | on | off | on | 1.0 | 5/14 | 9/14 | 8/14 | 7/14 | 4/4 | 5/12 |
| dice | on | off | on | 0.75 | 5/14 | 9/14 | 12/14 | 8/14 | 4/4 | 12/20 |
| dice | on | off | on | 0.5 | 5/14 | 9/14 | 14/14 | 8/14 | 2/4 | 16/24 |
| dice | on | on | off | 1.0 | 9/14 | 12/14 | 10/14 | 8/14 | 4/4 | 7/18 |
| dice | on | on | off | 0.75 | 9/14 | 12/14 | 14/14 | 12/14 | 4/4 | 10/26 |
| dice | on | on | off | 0.5 | 9/14 | 12/14 | 14/14 | 12/14 | 2/4 | 10/26 |
| dice | on | on | on | 1.0 | 12/14 | 13/14 | 9/14 | 8/14 | 4/4 | 2/13 |
| dice | on | on | on | 0.75 | 12/14 | 13/14 | 14/14 | 13/14 | 4/4 | 4/23 |
| dice | on | on | on | 0.5 | 12/14 | 13/14 | 14/14 | 13/14 | 2/4 | 4/23 |
| bm25 | off | off | off | 1.0 | 8/14 | 10/14 | 9/14 | 7/14 | 4/4 | 5/14 |
| bm25 | off | off | off | 0.75 | 8/14 | 10/14 | 12/14 | 8/14 | 4/4 | 9/20 |
| bm25 | off | off | off | 0.5 | 8/14 | 10/14 | 14/14 | 8/14 | 2/4 | 12/23 |
| bm25 | off | off | on | 1.0 | 9/14 | 10/14 | 9/14 | 8/14 | 4/4 | 4/14 |
| bm25 | off | off | on | 0.75 | 9/14 | 10/14 | 12/14 | 10/14 | 4/4 | 6/19 |
| bm25 | off | off | on | 0.5 | 9/14 | 10/14 | 14/14 | 10/14 | 2/4 | 9/22 |
| bm25 | off | on | off | 1.0 | 9/14 | 11/14 | 8/14 | 6/14 | 4/4 | 4/11 |
| bm25 | off | on | off | 0.75 | 9/14 | 11/14 | 11/14 | 9/14 | 4/4 | 5/17 |
| bm25 | off | on | off | 0.5 | 9/14 | 11/14 | 14/14 | 10/14 | 2/4 | 8/21 |
| bm25 | off | on | on | 1.0 | 11/14 | 12/14 | 7/14 | 7/14 | 4/4 | 1/9 |
| bm25 | off | on | on | 0.75 | 11/14 | 12/14 | 11/14 | 10/14 | 4/4 | 2/15 |
| bm25 | off | on | on | 0.5 | 11/14 | 12/14 | 14/14 | 11/14 | 2/4 | 5/19 |
| bm25 | on | off | off | 1.0 | 8/14 | 11/14 | 10/14 | 8/14 | 4/4 | 8/16 |
| bm25 | on | off | off | 0.75 | 8/14 | 11/14 | 13/14 | 9/14 | 4/4 | 12/22 |
| bm25 | on | off | off | 0.5 | 8/14 | 11/14 | 14/14 | 9/14 | 2/4 | 13/23 |
| bm25 | on | off | on | 1.0 | 10/14 | 12/14 | 8/14 | 8/14 | 4/4 | 3/11 |
| bm25 | on | off | on | 0.75 | 10/14 | 12/14 | 12/14 | 10/14 | 4/4 | 7/18 |
| bm25 | on | off | on | 0.5 | 10/14 | 12/14 | 14/14 | 11/14 | 2/4 | 8/20 |
| bm25 | on | on | off | 1.0 | 10/14 | 13/14 | 8/14 | 8/14 | 4/4 | 1/10 |
| bm25 | on | on | off | 0.75 | 10/14 | 13/14 | 13/14 | 11/14 | 4/4 | 5/18 |
| bm25 | on | on | off | 0.5 | 10/14 | 13/14 | 14/14 | 12/14 | 2/4 | 5/19 |
| bm25 | on | on | on | 1.0 | 13/14 | 13/14 | 9/14 | 9/14 | 4/4 | 0/10 |
| bm25 | on | on | on | 0.75 | 13/14 | 13/14 | 13/14 | 12/14 | 4/4 | 2/16 |
| bm25 | on | on | on | 0.5 | 13/14 | 13/14 | 14/14 | 13/14 | 2/4 | 2/17 |

#### Tuning split: citation threshold (stem on, min_coverage 0.75; ratio 0 rows are above)

| scorer | title | synonyms | min_citation_ratio | hit@1 | hit@3 | answered | correct | refused | off-chapter |
|---|---|---|---|---|---|---|---|---|---|
| dice | off | off | 0.5 | 4/14 | 8/14 | 11/14 | 7/14 | 4/4 | 12/19 |
| dice | off | off | 0.8 | 4/14 | 8/14 | 8/14 | 6/14 | 4/4 | 7/13 |
| dice | off | off | 0.9 | 4/14 | 8/14 | 6/14 | 5/14 | 4/4 | 3/8 |
| dice | off | on | 0.5 | 5/14 | 9/14 | 12/14 | 8/14 | 4/4 | 12/20 |
| dice | off | on | 0.8 | 5/14 | 9/14 | 9/14 | 7/14 | 4/4 | 7/14 |
| dice | off | on | 0.9 | 5/14 | 9/14 | 7/14 | 6/14 | 4/4 | 3/9 |
| dice | on | off | 0.5 | 9/14 | 12/14 | 14/14 | 12/14 | 4/4 | 10/26 |
| dice | on | off | 0.8 | 9/14 | 12/14 | 11/14 | 10/14 | 4/4 | 5/17 |
| dice | on | off | 0.9 | 9/14 | 12/14 | 7/14 | 6/14 | 4/4 | 2/9 |
| dice | on | on | 0.5 | 12/14 | 13/14 | 14/14 | 13/14 | 4/4 | 4/23 |
| dice | on | on | 0.8 | 12/14 | 13/14 | 10/14 | 10/14 | 4/4 | 0/12 |
| dice | on | on | 0.9 | 12/14 | 13/14 | 9/14 | 9/14 | 4/4 | 0/10 |
| bm25 | off | off | 0.5 | 8/14 | 11/14 | 13/14 | 9/14 | 4/4 | 11/21 |
| bm25 | off | off | 0.8 | 8/14 | 11/14 | 13/14 | 9/14 | 4/4 | 11/21 |
| bm25 | off | off | 0.9 | 8/14 | 11/14 | 9/14 | 6/14 | 4/4 | 4/10 |
| bm25 | off | on | 0.5 | 10/14 | 12/14 | 12/14 | 10/14 | 4/4 | 7/18 |
| bm25 | off | on | 0.8 | 10/14 | 12/14 | 12/14 | 10/14 | 4/4 | 7/18 |
| bm25 | off | on | 0.9 | 10/14 | 12/14 | 9/14 | 7/14 | 4/4 | 3/10 |
| bm25 | on | off | 0.5 | 10/14 | 13/14 | 13/14 | 11/14 | 4/4 | 5/18 |
| bm25 | on | off | 0.8 | 10/14 | 13/14 | 12/14 | 10/14 | 4/4 | 5/16 |
| bm25 | on | off | 0.9 | 10/14 | 13/14 | 10/14 | 8/14 | 4/4 | 2/11 |
| bm25 | on | on | 0.5 | 13/14 | 13/14 | 13/14 | 12/14 | 4/4 | 2/16 |
| bm25 | on | on | 0.8 | 13/14 | 13/14 | 12/14 | 11/14 | 4/4 | 2/14 |
| bm25 | on | on | 0.9 | 13/14 | 13/14 | 11/14 | 11/14 | 4/4 | 0/12 |

</details>
