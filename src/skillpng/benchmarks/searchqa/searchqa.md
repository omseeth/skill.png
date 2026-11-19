# SearchQA

Code: [`searchqa.py`](searchqa.py) (the whole adapter — there is no builder and no report script).
Config: [`configs/searchqa.yaml`](../../../configs/searchqa.yaml), inheriting
[`configs/base.yaml`](../../../configs/base.yaml), which is where the system prompt actually lives.
Shipped split: [`data/searchqa_id_split/`](../data/searchqa_id_split).

Corpus measurements below are over the 2,000 shipped items unless stated otherwise, taken from
`lucadiliello/searchqa` as it stands today, with the `Qwen/Qwen3.5-9B` tokenizer where tokens are
counted.

---

## 1. What the task is

One prompt, one answer, nothing retrieved. The model sees a Jeopardy! clue and the ten-or-so web
snippets SearchQA already collected for it, and answers in one turn. **No retrieval step, no tools,
no environment.** The passages are given; the task is reading them.

```
system: <benchmark.system_prompt>          # task instruction + the <answer> schema
user:   ## Context
        {MRQA snippet block, [DOC]/[TLE]/[PAR] markers, verbatim}

        ## Question
        {clue, whitespace-collapsed}
```

The image goes into the **user** turn at `injection.site` (chat templates do not expand an image
block in a system message). `pre_query` puts it ahead of the `## Context` block, `post_query` after
the question. `searchqa.yaml` pins `pre_query`.

Target and answer shape:

```
<answer>Lou Gehrig</answer>
```

The training target is `ANSWER_TEMPLATE.format(answer=answers[0])` — the gold wrapped in exactly
the tags the system prompt asks for — so the CE is taken on the shape that is also scored. The
scorer drops everything up to the last `</think>`, takes the **last** `<answer>` block, then falls
back to the last non-empty line, then to the stripped output.

---

## 2. Where the split comes from

**The splits are fixed, not computed.** `data/searchqa_id_split/{train,val,test}/items.json` lists
32-hex ids matched against the mirror's `key` field. No fraction, no seed, no grouping.

| split | items | drawn from the Hub split |
|---|---|---|
| train | 400 | `train` (117,384 rows) |
| val | 200 | `validation` (16,980 rows) |
| test | 1400 | `validation` |

Three consequences worth knowing:

- **`Item.group` is always `None`.** Nothing is grouped, because nothing is split here —
  `split_by_group` is never called for this benchmark, so `benchmark.allow_ungrouped` has no effect
  on it either. Do not read the absence of a group as evidence that leakage was considered; the
  split was inherited.
- **Items are returned in manifest order**, and `benchmark.limit` takes a prefix of each split.
  That changes the corpus size, not which split anything is in.

### The load is a full scan, every time

`_records_by_id` walks the mirror end to end looking for the manifest's ids.

### Leakage, such as it is

No question string occurs in two splits, and none repeats inside test. Answer *strings* do repeat:
**56 of 1,400 test items (4.0%) have an answer that also appears as a train answer**, 10 of 200 in
val. 1,819 of the 2,000 answers are unique overall; the most common (`italy`, `russia`) occur five
times. A universal image optimized on 400 train clues has that much surface to memorize and no
more — small, but say the n rather than claiming zero.

---

## 3. Token budgets

| | median | p95 | p99 | max |
|---|---|---|---|---|
| user turn (context + question) | 1,061 | 1,185 | 1,282 | **1,723** |
| target | 8 | 11 | 12 | **17** |

System prompt: **108** tokens. Worst-case training sequence ≈ 1,723 + 108 + 17 = **1,848**, plus
image tokens.

> **The `base.yaml` defaults are safe here and `searchqa.yaml` deliberately does not override
> them.** `max_target_tokens: 512` truncates nothing when the longest target is 17 tokens, and
> `max_sequence_tokens: 10240` leaves ~8,000 tokens of headroom.

What *is* tight is the **generation** budget, not the training one. See §6.

---

## 4. Truncation

`truncate_context` cuts the snippet block back to `MAX_CONTEXT_CHARS = 6000` at `[DOC]` boundaries,
preserving a documented quirk (the leading split piece is empty, so the first kept document loses
its `[DOC]` marker). 

Measured on the mirror:

| | shipped 2,000 items | all 134,364 mirror rows |
|---|---|---|
| median context | 3,881 chars | — |
| max context | 4,820 chars | **5,528 chars** |
| rows over 6,000 chars | **0** | **0** |

> **Truncation never fires.** Every item is fed its whole snippet block, and the `[DOC]`-marker 
> quirk the parity tests protect is never observable in a real run.

That is not a reason to delete it. The parity tests are what make the protocol claim ("byte-
identical to the implementation the baseline numbers were measured against") checkable, and a
different mirror or a re-materialized payload could exceed the budget. But **do not describe
6,000-char truncation as a property of these runs** — it is a property of the code, and on this
data it is inert. The median item carries 11 `[DOC]` blocks and keeps all of them.

---

## 5. Ceilings and recoverability

For every item we test whether a gold answer occurs in the context under SQuAD normalization
(casefold, drop punctuation and articles, collapse whitespace):

- **The answer is in the context for 1,999 of 2,000 items (99.95%).** Test and val are at 100.0%;
  the single miss is a train row (`1e2d4776…`, a mangled clue whose gold is `the 19th century`).
- **15 of 1,400 test answers appear in the question itself.**

The gap that *does* exist is between hard and soft scoring. Unigram F1 sits systematically above
exact match on the same extracted string: the difference is answers that are right but differently
surfaced (`(Lou) Gehrig` vs `Lou Gehrig`, an extra qualifier).

---

## 6. The generation budget

`searchqa.yaml` sets `max_new_tokens: 8000`. A reasoning model that does not converge inside that
budget is cut off mid-trace, and because the target is short (§3) nothing else in the config bounds
how long the model thinks first.

The extraction rule then decides the score. Extraction drops everything up to the last `</think>`
and takes the **last** `<answer>` block in what remains; a block never spans a second opener, so a
trace that only names the tag cannot swallow the real block after it.

```
...  *   Maybe the answer is simply **Serve**.
     *   If I ask "Apple, Banana, Orange", the answer is "<answer>...</answer>"
```
---

## 7. Scoring

`score()` returns one number, selected by `benchmark.options.metric` (`f1` in the dataclass
default, `em` in the shipped config), and reports it as `exact_match` or `unigram_f1`.

| step | behavior |
|---|---|
| extraction | text after the last `</think>`: last `<answer>` tag → last non-empty line → stripped output |
| normalization | SQuAD: casefold, strip punctuation, drop `a`/`an`/`the`, collapse whitespace |
| aggregation | `max` over the accepted surface forms in `Item.ground_truth` |

Three things to know about that last row:

1. **The max over surface forms is nearly inert.** Only **16 of 2,000 items carry more than one
   accepted answer** (14 in test, 1 in val, 1 in train). MRQA stores a list because SQuAD does;
   SearchQA almost always fills it with one string.
2. **The per-item JSONL cannot reproduce the reported score exactly.** `PredictionRow` writes
   `target` — which wraps `answers[0]` only — and not `ground_truth`. Rescoring a test JSONL
   offline therefore loses the alternates on those 14 multi-form test items and lands slightly low.
   The bias is small, but it is systematic and it is always downward.
3. **The last-line fallback is a safety net, not a scoring path.** It fires only when the output
   carries no complete tag pair at all.

---

## 8. Running it

```bash
# baseline, then the image — same seed, same split, same eval.limit
uv run skillpng eval -c configs/searchqa.yaml --set eval.condition=none
uv run skillpng eval -c configs/searchqa.yaml \
  --set eval.condition=skillpng \
  --set eval.image_path=runs/searchqa/<model>/<run>/best.png

# the control for "does any image at all move this"
uv run skillpng eval -c configs/searchqa.yaml --set eval.condition=noise

# optimize
uv run skillpng train -c configs/searchqa.yaml

# read the runs: one table over every runs/<benchmark>/<model>/results.jsonl
uv run python analysis/analyze_runs.py
```

`eval.limit` caps the split being scored; `benchmark.limit` caps every split at load. For a smoke
run use `--set eval.limit=32` — `benchmark.limit` also shrinks the val slice, which matters during
training and not during eval.

There is **no build step**: the split ships as ids and the corpus comes from the Hub on first use.
Point `HF_HOME` at shared storage on a cluster so compute nodes reuse one copy of the 134k-row
mirror.

---

## 9. Quick reference

```
corpus            lucadiliello/searchqa, 134,364 rows scanned in full on first items() call
splits            400 / 200 / 1400, fixed ids from data/searchqa_id_split, Item.group is None
                  train from the Hub train split, val + test from the Hub validation split
prompt            system + "## Context\n{block}\n\n## Question\n{clue}", image in the user turn
budgets           user turn max 1,723 tok, target max 17 tok — base.yaml defaults are safe
truncation        MAX_CONTEXT_CHARS 6000; max context in the whole mirror is 5,528 — never fires
ceiling           answer present in the context for 1,999 of 2,000 items; scores are model-limited
headline          exact_match (metric: em); metric: f1 scores the same string and reads higher
watch for         reasoning traces cut off at max_new_tokens, scored on a quoted <answer>...</answer>
report beside it  the delta restricted to items both arms finished, and the noise condition
no digest         only the ids are pinned; the mirror's text is not
```
