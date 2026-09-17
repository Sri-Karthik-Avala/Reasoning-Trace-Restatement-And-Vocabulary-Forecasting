# Reasoning Trace Restatement And Vocabulary Forecasting

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Sequence To Sequence |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 2 |
| Last submission | 2026-07-22 |

## Problem statement

**Reasoning Trace Restatement And Vocabulary Forecasting**

**Overview**

When a person or an automated system works a problem, it leaves a **reasoning trace** — a running first-person record of the attempt, including the assumptions it commits to, the branches it explores, the steps it abandons, and the resolution it eventually reaches. A trace is written *while* thinking and is therefore full of hedging, self-correction, and second-guessing. A **canonical restatement** is what the same work looks like once it is settled: the committed assumptions and the load-bearing steps, expressed in the settled vocabulary of someone who already knows how it ended.

A trace and its restatement do not share a vocabulary. A trace says *"let me try dividing through and see if that helps"*; a restatement says *"I divided through to simplify"*. The restatement systematically introduces content terms the trace never contains — the vocabulary of settled retrospection — and both the size and the identity of that introduced vocabulary vary sharply from trace to trace, depending on how much the writer hedged and how much of the exploration survived into the final account. Across the training data a restatement introduces a median of 13 content terms absent from its trace, and two randomly chosen items' introduced-term sets overlap at a set F1 of about 0.03.

For each held-out trace a submission provides two outputs: its canonical restatement, and the set of content terms that the true restatement introduces relative to the trace. Both are scored, and the score for the term set is zero for any submission that reproduces the trace's own vocabulary rather than the settled vocabulary.

The challenge targets a CPU-only solver environment: **10 CPU cores, 62 GB RAM, no GPU, ≤ 1.5 hours** end-to-end (data loading, any training, and inference). Traces are capped in length so the full task fits this budget.

**Dataset**

**Public files**

- **public/train.csv** — training items with canonical restatements. Columns: `id`, `reasoning_trace`, `canonical_restatement`. 2,000 rows.
- **public/test.csv** — test items, traces only. Columns: `id`, `reasoning_trace`. 250 rows.
- **public/sample_submission.csv** — a valid baseline submission in the exact required format. Columns: `id`, `predicted_restatement`, `introduced_terms`. 250 rows.

**Private file**

- **private/answers.csv** — organizer-only. Columns: `id`, `canonical_restatement`, `introduced_terms`. Holds the canonical restatement and the true introduced-term set for each test item; never distributed.

**Column descriptions**

The public CSVs use the following columns.

- **id** (string) — unique item identifier, formatted `item_<hex>`. Matches across train/test/submission/answers.
- **reasoning_trace** (string) — the raw first-person record of an attempt in progress. 40–256 words. Covers diverse work: quantitative derivation, code inspection, tool-use planning, and multi-step analysis. Contains hedging, abandoned branches, and a resolution.
- **canonical_restatement** (string) — present in `train.csv` and `private/answers.csv` only. A 28–90 word settled account of the same work (median ~42 words), written in retrospect.
- **predicted_restatement** (string) — submission column. The canonical restatement produced for a test trace. Must be non-empty.
- **introduced_terms** (string) — submission column. A semicolon-separated set of lowercase content terms predicted to appear in the true canonical restatement but nowhere in the reasoning trace, e.g. `assumed;divided;simplify`. Order is ignored and duplicates are collapsed. May be left empty, which scores zero on that item.

**Data example**

A training row (`train.csv`):

```
id,reasoning_trace,canonical_restatement
item_a3f82c1d,"If x = 0 the expression is 0, so assume x nonzero. Dividing numerator and denominator by x^4 gives a cleaner form; then applying AM-GM to the denominator terms bounds it below, which bounds the whole expression above. Checking the equality condition confirms the maximum is attained...","I assumed x nonzero and divided numerator and denominator by x^4 to simplify. I then applied AM-GM to the denominator to obtain a lower bound, which gave an upper bound on the expression, and verified the equality condition to confirm the maximum."
```

**Submission Format**

Submit a CSV with **exactly** these three columns, in any row order, with a header row:

```
id,predicted_restatement,introduced_terms
```

- Exactly one row per test `id` — all 250 test IDs present, no missing IDs, no unknown IDs, no duplicates.
- **predicted_restatement** — a non-empty string. Quote fields containing commas or newlines per standard CSV rules.
- **introduced_terms** — a semicolon-separated list of single alphanumeric words, e.g. `verified;assumed;simplify`. Whitespace around terms is stripped and terms are lowercased. A term containing anything other than word characters is rejected. The field may be left empty.
- No extra or unexpected columns are permitted; a submission with additional columns is rejected.

Sample submission row:

```
id,predicted_restatement,introduced_terms
item_a3f82c1d,"The trace divides through by x^4, applies AM-GM to bound the denominator, and confirms the maximum via the equality condition.","assumed;divided;simplify;verified;confirm"
```

**Evaluation**

**Trace Settlement Score (TSS).** Higher is better. The final TSS is a single value in [0.02, 1.0] combining restatement quality with vocabulary forecasting.

Tokenization lowercases text, removes punctuation (all non-word, non-whitespace characters), and splits on whitespace. **Content tokens** are the tokens remaining after removing this exact function-word list:

```
the a an is are was were be been being have has had do does did will would
could should may might shall can to of in for on with at by from that this
it i me my we our you your he she they them and but or not so if then else
let get got just also about up out all more some very what which who when
where how why im ive ok okay oh um uh well like yeah yes no its as into
each than much here there now need want think know see use used using make
made go going take taken come look say new way one two first thats dont ill
cant wont their over these those such
```

**1. Restatement Fidelity (CF).** For each item, ROUGE-L F1 between `predicted_restatement` and `canonical_restatement`. With `L` the length of the longest common subsequence of the two token sequences, `P = L / len(pred_tokens)`, `R = L / len(ref_tokens)`, and `CF = 2·P·R / (P + R)`. If either token sequence is empty, `CF = 0`. Range [0, 1].

**2. Restatement quality (G).** The mean of `CF` over all 250 items. Range [0, 1].

**3. True introduced-term set (T).** For each item, let `D` be the set of content tokens in the `canonical_restatement` and `A` the set of content tokens in the `reasoning_trace`. Then `T = D \ A`. Every test item is guaranteed at least 2 introduced terms. This set depends only on organizer-held data and cannot be influenced by the submission.

**4. Vocabulary forecasting (V).** For each item, let `P` be the submitted `introduced_terms` set. The per-item score is set F1:

- `V_i = 1` if `P` and `T` are both empty.
- `V_i = 0` if exactly one of `P`, `T` is empty.
- `V_i = 2·|P∩T| / (|P| + |T|)` otherwise.

`V` is the mean of `V_i` over all 250 items. Range [0, 1].

**Final score.**

```
TSS = clamp( 0.60 · G + 0.40 · V, 0.02, 1.0 )
```

Restatement quality contributes 60% of the ceiling and vocabulary forecasting 40%. The lower clamp of `0.02` is a reporting floor only; it does not otherwise affect the metric.

Pseudocode:

```
def tss(preds, refs, term_sets, true_sets):
    cf = [rouge_l_f1(tok(p), tok(r)) for p, r in zip(preds, refs)]
    G = mean(cf)
    V = mean([set_f1(p, t) for p, t in zip(term_sets, true_sets)])
    return max(0.02, min(1.0, 0.60 * G + 0.40 * V))
```

**Baselines.** Copying the trace's own leading sentences as the restatement scores `G ≈ 0.25` and `V = 0.00` — a restatement made of trace text introduces no new vocabulary — for `TSS ≈ 0.15`. Submitting a single fixed list of the most frequent training introduced-terms for every item scores `V ≈ 0.12`.

Grading configuration: Grade Direction = Maximize, Min Score = 0, Max Score = 1.

**What Not To Use (Prohibited Methods)**

- Do not use external answer keys, pre-computed restatements, or any trace-to-restatement corpus to recover the test-set canonical restatements or their introduced-term sets.
- Do not match test-set trace text, item IDs, or trace content back to any external dataset, corpus, or model output record.
- Do not hardcode id-to-restatement or id-to-term-set mappings, and do not memorize canonical restatements from any external source.
- Do not use the private canonical restatements (or any leaked copy of them) during training or inference.
- Do not train or tune against the private test labels, leaderboard scores, or shared solutions.
- Do not submit a verbatim slice of the reasoning trace in place of a restatement — the canonical restatements are re-phrased settled accounts, not extracted spans, so copied text does not satisfy the task.
