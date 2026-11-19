# LiveMathematicianBench

Code: [`livemathematicianbench.py`](livemathematicianbench.py) (the whole adapter — there is no
builder and no report script).
Config: [`configs/livemathematicianbench.yaml`](../../../../configs/livemathematicianbench.yaml),
inheriting [`configs/base.yaml`](../../../../configs/base.yaml). Unlike SearchQA, the system prompt
lives in the experiment config, not in the base.
Shipped split: [`data/livemathematicianbench_id_split/`](../data/livemathematicianbench_id_split/).

Measurements below are over the 345 shipped items (filtered, §2), read from
`LiveMathematicianBench/LiveMathematicianBench` at the pinned revision, with the `Qwen/Qwen3.5-9B`
tokenizer where tokens are counted.

---

## 1. What the task is

One prompt, one answer, nothing retrieved. The model sees one research-level question about a
theorem from a recent arXiv paper and five candidate statements, and answers with a single label.
**No context, no tools, no environment.**

```
system: <benchmark.system_prompt>          # task instruction + the <answer> schema
user:   ## Question
        {mcq.question, verbatim, ends stripped}

        ## Choices
        A. {option text, verbatim, ends stripped}
        B. ...
        C. ...
        D. ...
        E. ...
```

The record's `theorem` and `sketch` fields are **never shown**: they state the result the question
asks about. Question and option texts are passed through verbatim with only the ends stripped —
140 of the 345 questions and 425 of the 1,725 option texts contain a newline, mostly LaTeX display
math — so whitespace is not collapsed.

The image goes into the **user** turn at `injection.site` (chat templates do not expand an image
block in a system message). `pre_query` puts it ahead of the `## Question` block, `post_query` after
the last choice. `livemathematicianbench.yaml` pins `pre_query`.

Target and answer shape:

```
<answer>C</answer>
```

The training target is `ANSWER_TEMPLATE.format(label=gold)` — the shown gold label wrapped in
exactly the tags the system prompt asks for — so the CE is taken on the shape that is also scored.

---

## 2. Where the split comes from

**The splits are fixed, not computed.** `data/livemathematicianbench_id_split/{train,val,test}/items.json`
lists `<month>:<no>` ids; each entry also carries `month`, `no`, `paper_link` and `source_file`.
The adapter reads the split and never recomputes it.

### The source is pinned

`split_manifest.json` pins the dataset to revision `6f53c5ff7227633ea954b2847cd590314d582047` and
names eight files, `data/<month>/qa_<month>_final.json` for the months 202511 to 202606. Each is a
JSON list of records. Together they hold 608 records, and the manifest names 345 of them.

The adapter downloads exactly those files at exactly that revision with
`huggingface_hub.hf_hub_download(repo_type="dataset", revision=...)`. **It never calls
`load_dataset` on the default config at `main`**: that config picks up every month on the Hub and
grows whenever a new one is published, so it would be a different corpus next month.

### How it was made

The manifest records the procedure: records in `source_files` order, then file order;
`random.Random(42).shuffle`; train = the first `int(0.2*n)`, val = the next `int(0.1*n)`, test =
the rest (`split_ratio` 2:1:7), which gives 121 / 60 / 427.

**The split used in this repository is then filtered**: records whose options contain *"One of the
remaining options is correct, but a stronger result can be proven."* (`exclude_option_text`) are
removed, since that option is the gold whenever present (§5). The rest keep their split and order.
The unfiltered split is kept unchanged in
[`data/livemathematicianbench_full_id_split/`](../data/livemathematicianbench_full_id_split/).

| month | train | val | test | total |
|---|---|---|---|---|
| 202511 | 5 | 1 | 9 | 15 |
| 202512 | 4 | 2 | 20 | 26 |
| 202601 | 5 | 1 | 20 | 26 |
| 202602 | 6 | 3 | 16 | 25 |
| 202603 | 13 | 7 | 48 | 68 |
| 202604 | 15 | 6 | 39 | 60 |
| 202605 | 14 | 4 | 38 | 56 |
| 202606 | 11 | 9 | 49 | 69 |
| **all** | **73** | **33** | **239** | **345** |

### Text is passed through as shipped

The pinned revision also repairs corrupted escapes of the earlier one (4 questions and 2 choice sets
in 202511–202602, per the manifest). The adapter does not clean text itself, and not everything is
repaired: **2 records still carry control characters where LaTeX belongs**, both in val:
202603:89, with the terminal bold codes `\x1b[1mE\x1b[0m`, and 202606:15.

### Groups and failure modes

- **`Item.group` is `paper_link`.** Each of the 345 papers contributes exactly one item, so no paper
  is in two splits and there is nothing within a paper for a universal image to memorize.
- **Items are returned in manifest order**, and `benchmark.limit` takes a prefix of each split.
- The load **fails loudly** on a manifest id missing from the files, an id shipped twice, a manifest
  field (`month`, `no`, `paper_link`, `source_file`) that disagrees with the record, an entry whose
  `source_file` is not in `source_files`, option or meta labels other than exactly A–E, and any
  non-empty `benchmark.options` (there are none).

---

## 3. How the choices are presented

**The source ships the correct choice as A in every record**, with four distractors as B–E.
Shown as shipped, a universal image that learned "answer A" would score 100%.

So every item is shown in its own fixed order. Each option is ranked by a sha256 of
`f"{PERMUTATION_SEED}:{item_id}:{source_label}"` (`base._unit_hash`), shown in that order and
relabelled A–E. The gold and the meta labels (`weaker_true_label`, `false_labels`,
`wildcard_false_label`) are remapped to the shown labels. The order depends on nothing but the
seed, the id and the label: it is the same across processes, runs, splits and the image / no-image
conditions. **`PERMUTATION_SEED = 0` is protocol**: changing it reorders every item and breaks
comparability with earlier runs. The tests pin the first five train orders.

Gold labels after the permutation:

| split | A | B | C | D | E |
|---|---|---|---|---|---|
| train | 11 | 13 | 14 | 22 | 13 |
| val | 6 | 9 | 5 | 6 | 7 |
| test | 47 | 46 | 46 | 54 | 46 |
| all | 64 | 68 | 65 | 82 | 66 |

---

## 4. Token budgets

Measured with the `Qwen/Qwen3.5-9B` tokenizer (nearest-rank percentiles):

| | median | p95 | p99 | max |
|---|---|---|---|---|
| user turn, characters | 1,746 | 3,792 | 6,017 | 7,403 |
| user turn, tokens | 593 | 1,483 | 2,623 | **3,361** |
| target, tokens | 6 | 6 | 6 | **6** |
| chat-templated system + user + target, tokens | 717 | 1,607 | 2,747 | **3,485** |

System prompt: **100** tokens. The worst-case training sequence is 3,485 tokens plus the image
tokens (64 for Qwen3.5 at 256x256, 256 for Gemma 4 and Nemotron 3).

---

## 5. Ceilings and shortcuts

- **Chance is 20%.** The best constant label scores 22.6% on test (D, 54 of 239), 27.3% on val
  (B, 9 of 33) and 30.1% on train (D, 22 of 73). Train's most frequent label is also test's, so an
  image that taught the model "answer D" from train scores 22.6% on test without any mathematics.
  Val's most frequent label scores 19.2% there (B, 46 of 239).
- **The stronger-result option is filtered out.** In the source it is the gold in every record that
  carries it, so it could be picked without any mathematics; those records are not in the split
  (§2).
- **Val has 33 items**, so one item moves the val score by 3.0 points, and `best.png` is selected in
  steps of that size. Val also leans towards B (9 of 33) and away from C (5 of 33): a checkpoint
  can gain on val from a label preference that does not carry over to test.

---

## 6. The generation budget

The target is 6 tokens, so nothing but `max_new_tokens` bounds how long the model thinks first, and
a trace that does not converge is cut off.

**A cut-off trace scores 0.** Without a complete `<answer>...</answer>` pair there is no answer, and
there is deliberately no last-line fallback: a trace cut at `max_new_tokens` can end on a line that
reads `A` (the article), which a fallback would count as a pick. A trace that quotes the system
prompt's `<answer>B</answer>` example and is then cut off still scores as B, as in SearchQA; report
the share of outputs without a tag beside the headline.

---

## 7. Scoring

`score()` returns accuracy (the base `metric_name`): 1.0 when the extracted label is the gold one.

| step | behavior |
|---|---|
| extraction | text after the last `</think>`, then its **last** `<answer>` tag, case-insensitive; a later tag overrides a draft one, and a tag never spans a second opener |
| accepted | one label A–E, with optional surrounding whitespace, parentheses, `**` bold and a trailing period |
| scores 0 | several letters, prose, an empty tag, a lowercase or unknown letter, no complete tag |

`Item.ground_truth` is the bare shown label and the target wraps it, so rescoring the per-item JSONL
offline from its `target` column is exact.

---

## 8. Running it

```bash
# baseline, then the noise control — same seed, same split, same eval.limit
uv run skillpng eval -c configs/livemathematicianbench.yaml --set eval.condition=none
uv run skillpng eval -c configs/livemathematicianbench.yaml --set eval.condition=noise

# optimize on train, selecting best.png on the whole val split (train.eval_items: 33)
uv run skillpng train -c configs/livemathematicianbench.yaml

# the optimized image on test
uv run skillpng eval -c configs/livemathematicianbench.yaml \
  --set eval.condition=skillpng \
  --set eval.image_path=runs/livemathematicianbench/livemathematicianbench/best.png

# read the runs: one table over every runs/<benchmark>/<model>/results.jsonl
uv run python analysis/analyze_runs.py
```

The image path is `<train.run_dir>/<train.run_name>/best.png` for the shipped config; set
`model.model_id` first. For a smoke run use `--set eval.limit=32`.

There is **no build step**: the split ships as ids and the eight files come from the Hub on the
first `items()` call. Point `HF_HOME` at shared storage on a cluster so compute nodes reuse one copy.

---

## 9. Quick reference

```
source            LiveMathematicianBench/LiveMathematicianBench @ 6f53c5ff, 8 files 202511-202606,
                  downloaded with hf_hub_download, never load_dataset at main
splits            73 / 33 / 239, fixed ids (seed 42, 2:1:7), filtered of the stronger-result
                  option; unfiltered 121 / 60 / 427 in livemathematicianbench_full_id_split;
                  extends the SkillOpt 35/17/125 split and is not comparable with it
group             paper_link; 345 papers, one item each
prompt            system + "## Question\n{q}\n\n## Choices\nA. ...\nE. ...", image in the user turn;
                  theorem and sketch never shown
choices           gold always A in the source; shown in a per-item sha256 order, PERMUTATION_SEED 0
budgets           full text sequence max 3,485 tok, target 6 tok: base.yaml defaults are safe
chance            20%; best constant label 22.6% on test (D, also train's), 27.3% on val
headline          accuracy on a lone label in the last <answer> tag after </think>; no fallback
val               33 items, 3.0 points per item
watch for         2 val records with control characters, passed through verbatim
```
