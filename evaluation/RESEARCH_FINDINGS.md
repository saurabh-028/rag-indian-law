# Research Findings — ShastraShaw as a Controlled RAG-vs-Parametric-Knowledge Baseline

**Date:** 2026-10-01
**For:** HECA Research Conference 2026 abstract submission
**Branch:** `research/heca-eval-overnight`

## 1. The question

A plain RAG legal assistant has no obvious edge over a general-purpose chatbot for an
end user who could just ask ChatGPT the same question. The pivot tested here: treat
ShastraShaw not as a product, but as a **controlled research baseline**, and ask a
sharper, checkable question —

> When a general-purpose LLM (GPT-4o, no retrieval, no legal-domain prompting) answers
> an Indian criminal-law question, does it reliably cite the **current** law (Bharatiya
> Nyaya Sanhita / Bharatiya Nagarik Suraksha Sanhita, in force since 1 July 2024), or
> does it default to the **repealed** IPC/CrPC section numbers it saw far more of in
> pretraining?

Three independent, increasingly strict instruments were used to test this, against the
exact same 44-question gold set (and a 32-item Hindi/Marathi sub-set), so the numbers
below triangulate rather than rest on a single metric:

1. **`citation_accuracy.py`** — deterministic regex match: does the answer cite the
   gold dataset's exact expected section number? (reused, not newly built for this study)
2. **`outdated_law_check.py`** (new) — deterministic regex scan for law-regime primacy
   (which code family — new or repealed — is named *first*) and, for 9 offences with an
   unambiguous published 1:1 BNS/BNSS↔IPC/CrPC concordance, whether the answer cites the
   *old* section number as if it were current.
3. **`llm_judge.py`** — GPT-4o-as-expert-lawyer, 5-criterion Likert rubric (reused,
   already in the repo), scored independently per condition.

## 2. Method

- **Conditions compared:** ShastraShaw (BM25+FAISS hybrid retrieval, top_k=10, sector
  prompt, citation-verification retry) vs. **frontier baseline** — GPT-4o given only the
  raw question and a generic "answer this legal question" system prompt, no retrieval
  context, no sector prompt, no citation verification (`evaluation/frontier_baseline.py`,
  new). Both use `gpt-4o` for generation, for a fair comparison.
- **English set:** all 44 questions in `evaluation/gold_dataset.json` (traffic,
  criminal_law, rental_law, matrimonial — 12/12/8/12 split; matrimonial folds in the
  former `HM_`/`MAT_` subsets).
- **Hindi/Marathi sub-study:** 16 of the 44 questions (4 per sector, chosen to overlap
  with the 9 clean BNS/IPC-mapped offences), hand-translated into Hindi and Marathi
  (`evaluation/build_multilingual_subset.py`, `evaluation/gold_dataset_multilingual.json`
  — **32 datapoints total, a deliberately smaller added sub-study, not a replacement for
  the 44-question English set**). ShastraShaw's run mirrors `app/main.py`'s actual
  `/query` pipeline exactly (translate-for-retrieval → hybrid retrieve → sector-prompted,
  language-instructed generate → citation verify/retry); sector and language are taken
  directly from the dataset rather than from the router/langdetect, since
  `CLAUDE.md` already documents langdetect misfiring on short queries and we already
  know the ground truth here.
- **Judge model:** `gpt-4o` for both generation and judging (the scripts' existing
  defaults), kept rather than swapped to `gpt-4o-mini`, so the ~150 judge calls stay
  consistent with the generation condition.

## 3. Headline result: outdated-law citation rate

For the 9 (English) / 10 (Hindi+Marathi) questions with an unambiguous published
BNS/BNSS↔IPC/CrPC section mapping (murder, theft, rape, cheating, grievous hurt,
anticipatory bail, FIR registration, dowry cruelty):

| Condition | Cited **only** the repealed section (no current number at all) |
|---|---|
| Frontier baseline — English | **8/9 (89%)** |
| Frontier baseline — Hindi/Marathi | **10/10 (100%)** |
| ShastraShaw — English | **0/9 (0%)** |
| ShastraShaw — Hindi/Marathi | **0/10 (0%)** |

Every single frontier-baseline miss on these 9 English questions cited the *exact*
pre-2024 number: murder → "302" not 103, theft → "379" not 303, rape → "376" not 64,
cheating → "420" not 318, grievous hurt → "325" not 117, anticipatory bail → "438" not
482, FIR → "154"/"156" not 173, dowry cruelty → "498A" not 85. In Hindi, the model is
if anything *more* explicit about it — e.g. for "BNS 2023 के तहत हत्या की सज़ा क्या है?"
(what is the punishment for murder under BNS 2023, asked by name) it answered: *"भारतीय
दंड संहिता (आईपीसी) के तहत... धारा 302"* — "Under the Indian Penal Code (IPC)...
Section 302" — naming the wrong, repealed code **even though the question explicitly
named BNS 2023**.

A second, broader cut (which code family — BNS/BNSS or IPC/CrPC — is named *first* in
the answer, across all criminal_law + matrimonial questions, not just the 9 clean
mappings):

| Condition | Defaults to repealed law (named first, or exclusively) |
|---|---|
| Frontier baseline — English (n=24) | 10/24 (42%) |
| Frontier baseline — Hindi/Marathi (n=16) | 10/16 (62%) |
| ShastraShaw — English (n=24) | 0/24 (0%) |
| ShastraShaw — Hindi/Marathi (n=16) | 0/16 (0%) |

## 4. Citation accuracy (deterministic, `citation_accuracy.py`)

Exact match against `expected_section` in the gold set, only for questions with a
single numeric correct answer (31/44 English, 30/32 multilingual qualify):

| Sector | ShastraShaw (EN) | Frontier (EN) | ShastraShaw (hi/mr) | Frontier (hi/mr) |
|---|---|---|---|---|
| Criminal Law | **8/10 (80%)** | 0/10 (0%) | **4/8 (50%)** | 0/8 (0%) |
| Matrimonial & Family | 7/7 (100%) | 6/7 (86%) | 3/6 (50%) | 2/6 (33%) |
| Rental Law | 3/5 (60%) | 1/5 (20%) | 2/8 (25%) | 0/8 (0%) |
| Traffic Law | 2/9 (22%) | 4/9 (44%) | 2/8 (25%) | 2/8 (25%) |
| **Overall** | **20/31 (64.5%)** | **11/31 (35.5%)** | **11/30 (36.7%)** | **4/30 (13.3%)** |

Criminal Law is the starkest gap and the cleanest story — 0% for the frontier baseline
in *both* languages, every miss a repealed section. Traffic Law is ShastraShaw's weakest
sector in both conditions (and the frontier baseline edges it in English, 44% vs. 22%)
— consistent with `CLAUDE.md`'s documented InLegalBERT vocabulary-gap issue
("red light" vs. "jumping signal") biting the retriever specifically in traffic, where
MV Act section numbers for fines (e.g. 184 "mobile phone", 194 "red signal", 196
"insurance") aren't being surfaced in the top-10 chunks as reliably as in the other three
sectors (Hit@10 is 93% in aggregate, but a correct chunk present in context ≠ the model
citing the right number from it). This is a genuine retriever weakness, not a byproduct
of the baseline comparison, and is flagged here rather than glossed over.

## 5. LLM-as-expert-lawyer rubric (`llm_judge.py`, gpt-4o, 5-criterion Likert)

| Condition | Overall avg | Legal correctness | Criminal Law overall | Criminal Law correctness failures (1/5) |
|---|---|---|---|---|
| ShastraShaw — EN | 3.45/5.00 | 2.52/5.00 | 3.30/5.00 | 3/12 (25%) |
| Frontier — EN | 3.49/5.00 | 2.41/5.00 | 2.83/5.00 | 7/12 (58%) |
| ShastraShaw — hi/mr | 3.46/5.00 | 2.34/5.00 | 3.02/5.00 | 2/8 (25%) |
| Frontier — hi/mr | 3.14/5.00 | 2.16/5.00 | 2.08/5.00 | **8/8 (100%)** |

The **overall** 5-criterion average is close between conditions in English (3.45 vs.
3.49 — essentially a wash on the holistic rubric) — ShastraShaw does **not** win on
every axis; its completeness/actionability/clarity scores are comparable to or
marginally behind the frontier baseline's, since the baseline's longer, confident,
IPC-flavoured prose reads just as fluently to the judge when it isn't checking section
numbers against anything. The gap is narrow, specific, and large exactly where the
outdated-law problem predicts it should be: **Criminal Law**, where the frontier
baseline's correctness-failure rate (legal_correctness scored 1/5) goes from 58%
(English) to a perfect **100%** (8/8) in Hindi/Marathi — every single Hindi/Marathi
criminal-law answer from the frontier baseline was judged a critical legal-correctness
failure. ShastraShaw's Criminal Law failure rate holds steady at 25% in both languages.

This is useful triangulation: an LLM judge with no knowledge of our regex rules, shown
only the question/reference/answer, independently converges on the same criminal-law
cliff that the two deterministic checks found.

(Baseline note: `llm_judge.py` ships a hardcoded comparison to a prior one-off May 2026
report — 3.94/5.00 overall, under a different gold set and pre-dating the retriever/
router/verifier fixes already on this branch. All four conditions above score below
that number; that comparison is **not relevant to this study** and shouldn't be read as
"ShastraShaw got worse" — it's a different dataset and different era of the pipeline.
The only comparison that matters for this abstract is the four rows above, which share
an identical dataset, identical day, identical judge model, and identical method.)

## 6. Cross-lingual degradation

Both conditions get worse in Hindi/Marathi, not just ShastraShaw — so this is not a
"translation breaks RAG" story, it's "both systems are weaker cross-lingually, and the
gap between them narrows on raw citation accuracy but does not disappear, and is
unchanged-to-widened on the headline outdated-law metric":

- ShastraShaw citation accuracy: 64.5% (EN) → 36.7% (hi/mr), a real drop.
- Frontier baseline citation accuracy: 35.5% (EN) → 13.3% (hi/mr), also a real drop, and
  proportionally steeper.
- Outdated-law-citation rate: unchanged for both — 0% for ShastraShaw, ~90-100% for the
  frontier baseline, in both languages.

**Caveat, stated plainly:** this sub-study is 16 questions × 2 languages = 32
datapoints, hand-translated by the author (not a certified translator), scored by a
single LLM judge. It is a first signal, not a generalisable result — see §7.

## 7. Limitations (stated for an honest abstract, not a journal submission)

- **Sample size.** 44 English questions, 32 multilingual datapoints, single-run (no
  repeated sampling / variance estimate), single LLM judge, single generation
  temperature (0.1). None of the percentages above have a confidence interval.
- **Translation quality.** The Hindi/Marathi questions were hand-translated by the
  author aiming for natural/colloquial register (matching the production system's own
  language instructions), not professionally certified — translation artefacts could
  affect either condition's retrieval or generation quality independent of the
  phenomenon being measured.
- **Single grader, same model family as the generator.** `llm_judge.py` uses `gpt-4o`
  both to generate the frontier-baseline answers *and* to judge them — a same-family
  judge could have unknown leniency/severity biases relative to grading GPT-4o's own
  style of answer. No second, independent judge model was available (no Anthropic key
  in this environment) to cross-check.
- **The 9/10-question "clean mapping" set is hand-curated**, not exhaustive — it
  deliberately excludes offences (arrest-rights, bailable/non-bailable classification,
  kidnapping) where BNSS reorganised CrPC into a many-to-many mapping, to keep the
  headline number defensible rather than contestable. The broader law-regime-primacy
  check (§3, second table) covers all 24/16 criminal+matrimonial questions and shows the
  same direction of effect, so the headline number isn't cherry-picked to be the only
  one that works — but it is the strictest and narrowest cut.
- **Traffic Law citation accuracy** is weak for ShastraShaw too (22% EN / 25% hi/mr) —
  a genuine retriever gap (§4), not hidden by this study's framing, and worth noting as
  a reason ShastraShaw's advantage is domain-specific (strongest in criminal_law and
  matrimonial, where BNS/IPC and PWDVA/498A content is deep in the index) rather than
  universal.
- **BERTScore failed to compute** during the English run (`int too big to convert`,
  likely a `bert-score`/`transformers` version mismatch in this environment) — BLEU/
  ROUGE are reported for ShastraShaw where available; this doesn't affect any of the
  citation-accuracy or LLM-judge numbers above, which don't depend on it.
- **This OpenAI org's rate limit (30,000 TPM on gpt-4o)** caused one mid-run generation
  failure (ShastraShaw EN, question CL_006) during the original evaluation pass; it was
  regenerated and merged back in before any scoring ran, and a retry-with-backoff fix
  was committed to `app/generator.py`/`app/language.py` (and the eval scripts) so this
  can't recur silently. No other condition's run had any missing answers (44/44, 44/44,
  32/32, 32/32 generated across the four runs).

## 8. Bugs found and fixed along the way (committed separately, see git log)

1. **Rate-limit silent failures** — `Generator.generate()` / `Translator._translate()`
   had no retry logic; a transient 429 recorded a blank `""` answer that then scored as
   a total failure in every downstream metric, and would surface as a hard 502 to a real
   `/query` caller in production. Fixed with `app/retry_utils.call_with_retry()`.
2. **Citation/law-name regexes were English-only** — `app/verifier.py`'s production
   citation-verification, `citation_accuracy.py`, and `outdated_law_check.py` all only
   matched the English word "Section" and Latin abbreviations (IPC/BNS/etc.). Hindi/
   Marathi answers correctly say "धारा 302"/"कलम 302" and "भारतीय दंड संहिता (आईपीसी)"
   per the production language instructions — meaning `app/verifier.py`'s
   citation-verification safety net has been **silently inert for every Hindi/Marathi
   production answer** since it shipped (it never found anything to flag, not because
   nothing was wrong, but because it wasn't reading the language the model was
   answering in). Fixed in all three places.
3. **`evaluate.py` crashed on Windows** when stdout was redirected to a file (`cp1252`
   can't encode the script's box-drawing section-header characters) — fixed with a
   UTF-8 stdout/stderr reconfigure at startup.

## 9. Raw data

Per-condition results (not committed — gitignored under `evaluation/results/`, since
that's this repo's existing convention — but reproducible by re-running the commands
below against this branch) live under:

```
evaluation/results/shastrashaw/                    (English, ShastraShaw)
evaluation/results/frontier_baseline/               (English, frontier baseline)
evaluation/results/shastrashaw_multilingual/        (Hindi+Marathi, ShastraShaw)
evaluation/results/frontier_baseline_multilingual/  (Hindi+Marathi, frontier baseline)
```

Each directory has `generated_answers.json`, `outdated_law_check.json`,
`lawyer_rating_detailed.json` / `lawyer_rating_summary.json`, and (English only)
`retrieval_results.csv` / `generation_results.csv` / `summary.json` / `plots/`.

To reproduce from scratch:
```bash
python evaluation/evaluate.py --index_dir ./index --dataset ./evaluation/gold_dataset.json \
    --out_dir ./evaluation/results/shastrashaw --top_k 10 --skip_ragas

python evaluation/frontier_baseline.py --dataset ./evaluation/gold_dataset.json \
    --out_dir ./evaluation/results/frontier_baseline

python evaluation/evaluate_multilingual.py --index_dir ./index \
    --dataset ./evaluation/gold_dataset_multilingual.json \
    --out_dir ./evaluation/results/shastrashaw_multilingual

python evaluation/frontier_baseline.py --dataset ./evaluation/gold_dataset_multilingual.json \
    --out_dir ./evaluation/results/frontier_baseline_multilingual

# then, for each of the 4 output directories:
python evaluation/citation_accuracy.py --generated <dir>/generated_answers.json --dataset <matching gold file>
python evaluation/outdated_law_check.py --generated <dir>/generated_answers.json --dataset <matching gold file> --label "<name>" --out <dir>/outdated_law_check.json
python evaluation/llm_judge.py --generated <dir>/generated_answers.json --dataset <matching gold file> --out_dir <dir>
```
