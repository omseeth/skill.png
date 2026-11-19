# TRAJECT-Bench

Code: [`traject.py`](traject.py) (adapter),
[`traject_build.py`](traject_build.py) (the two shipped artifacts),
[`traject_report.py`](traject_report.py) (offline reading).
Config: [`configs/traject.yaml`](../../../configs/traject.yaml). Clone (parity fixture only, gitignored):
`baselines/traject_bench/code/`.

---

## 1. What the task is

One prompt, one answer, nothing executed. The model sees a pool of candidate tools and a query,
and emits every call the query needs at once as a JSON list. This is the parallel-simple subset of 
TRAJECT-Bench recast as a supervised sweep, used both as the cross-entropy target and as a 
judge-free scorer.

```
system: <benchmark.system_prompt>  # task instruction + output schema
user:   ## Tools
        {k gold tools + 10 distractors, one block each, order fixed per row}

        ## Query
        {query verbatim}
```

The image goes into the **user** turn at `injection.site` (chat templates do not expand an image
block in a system message). `pre_query` puts it ahead of the tool block; `post_query` after the
query.

Target and answer shape:

```json
<answer>[{"tool name": "Wayfair: reviews/list",
          "required parameters": [{"name": "sku", "value": "W004939121"}],
          "optional parameters": [{"name": "page", "value": "1"},
                                  {"name": "sort_order", "value": "HELPFUL"}]}]</answer>
```

Calls sorted by tool name, parameters by name, deterministic separators. The same canonicalization
runs over model output before comparison, so permutations are free. Extraction drops everything up
to the last `</think>`, then takes the **last** `<answer>` block (never one spanning a second opener), then a fenced block, then the last balanced JSON value; `{"tool list": [...]}`
(the benchmark's own answer shape) and `required_parameters` (underscored) are both accepted.

---

## 2. Token budgets

Measured over the 875 shipped items with the `Qwen/Qwen3.5-9B` tokenizer, with the shipped render
(descriptions whole, defaults and `code` absent):

| | median | p95 | p99 | max |
|---|---|---|---|---|
| user turn (tools + query) | 2,651 | 4,220 | 4,930 | **5,743** |
| target | 213 | 468 | 566 | **867** |

System prompt: **258** tokens. Worst-case sequence ≈ 5,743 + 258 + 867 = **6,868**, plus image
tokens.

These are lower than the pre-filter figures (max 6,364 / 872 over all 1800 rows) because the
reachability filter of §4 removes parameter-heavy rows preferentially. The pins below did not
change: they were sized for the larger corpus and remain safe headroom for this one.

`configs/traject.yaml` therefore pins:

```yaml
model:
  max_target_tokens: 1024     # max observed 872
  max_sequence_tokens: 12288  # room for resolution 768 image tokens
```

**Do not inherit the defaults**, and re-measure if you change `n_distractors`,
`description_chars`, `render_output_info`, or the tokenizer family. `description_chars: null`
(whole descriptions) is what these numbers describe; capping at 300 would only save ~9%, so it is
not worth spending on by default.

---

## 3. Corpus and splits

Nine `parallel_*_simple` Hub configs, 200 rows each, **1800 total**, asserted at build time. The
paper's tenth parallel domain (Mapping) has no Hub config; `_hard` and `sequential_*` are out of
scope.

**925 of those 1800 rows are then dropped as unanswerable, leaving 875.** The rule and its cost are
§4; the short version is that a row whose gold requires a value the query never mentions cannot be
answered from any prompt we are willing to build, so it is removed rather than scored. Filtering
happens *before* the split, so the fractions below are over the 875 that remain.

| split | items | share |
|---|---|---|
| train | 175 | 20.0% |
| val | 89 | 10.2% |
| test | 611 | 69.8% |

Per-domain, train / val / test:

```
eCommerce 31/10/86   Education 27/8/75    Email      27/31/106
Finance    3/0/5     Gaming    27/12/110  Music       7/5/34
News_Media 28/11/98  Travel    15/5/47    Weather    10/7/50
```

**The filter does not fall evenly on the domains, and this is the one thing to know before
stratifying by domain.** Kept, out of 200: Email 164, Gaming 149, News_Media 137, eCommerce 127,
Education 110, Travel 67, Weather 67, Music 46, **Finance 8**. Finance is essentially gone — its
gold leans on `symbol`, `interval` and date ranges that the queries state in words — and with 8
rows total it has 3 in train, 0 in val and 5 in test. Report Finance as a footnote or not at all;
do not read a per-domain Finance number as an estimate of anything.

**Row id is `{config}:{row_index}`** — the manifest key, the per-instance seed, and `Item.id`.

### Grouping: `(domain, sorted set of gold provider names)`

743 groups over the 875 kept rows, none straddling a split, seed 109 (chosen from the first 200 as
closest to 20/10/70). The seed moved when the filter went in: it is searched over the rows that
survive, so the old seed 133 no longer names the closest partition.

The reason this is not `domain` and not `(domain, task_name)` is worth knowing, because it bounds
what a leakage claim can say:

- **Within every one of the nine domains, the gold-tool co-occurrence graph is a single connected
  component.** So "no gold tool appears in two splits" is *equivalent to* splitting on domain.
  There is no finer unit that blocks tool-level leakage. This is a property of the corpus, not a
  tuning choice.
- `domain` has 9 values: quantized at 11.1% steps, and whole domains land in test with zero train
  exposure.
- `(domain, task_name)` has 39 groups but Education and Music have one `task_name` each covering
  199/200 and 197/200 rows — two ~11% atoms that no seed can place well.

### `has_unseen_values` — thin, show the n

True when none of the row's gold parameter values (casefolded, whitespace-collapsed) occurs
anywhere in train. **50 of 611 test items (8.2%)**, 10 of 89 val, 0 of train by construction. The
share roughly doubled when the filter went in, for the mechanical reason that train shrank from 363
rows to 175 and so covers fewer values — the subset is not more novel, the reference set is
smaller. Usable as a subset, but it will give a wide interval — always print the n beside it.

---

## 4. The reachability filter

`full_em` is conjunctive: `tool_em == 1` **and** every required argument matching gold. One wrong
required value zeroes the item.

| source | available to the model? |
|---|---|
| **the query** | Yes, when the value appears in it. This is the ability we mean to measure. |
| **the schema `default`** | **No — stripped at index build (§5).** The synthesis pipeline seeded gold from defaults, so leaving it in turns extraction into copying. |
| **the model's prior** | Sometimes. `units='metric'`, `page=1`, coordinates for a city the query names in words. Convention and world knowledge, not extraction. |

### The rule

**A row ships only if every required gold value occurs in its query under normalization** —
casefold, drop punctuation, collapse whitespace, then substring (`recoverable_flags`,
`filter_reachable`). 925 rows fail and are dropped; the survivors satisfy
`n_required == n_required_recoverable` by construction, and the adapter refuses any item that does
not. Every dropped id is listed with its offending parameters in
`data/traject_unreachable.json`.

Optional parameters are **not** filtered on.

The dropped values are dominated by a few parameters: `interval` (486 occurrences), `date` (184),
`lon`/`lat` (201 together), `range` (88), `start`/`end` (160), `symbol` (78), plus API-key
placeholders like `ed10b4d4498256508a2deeoamdco23lam`.

---

## 5. Leak control, and its residuals

The model must infer argument *values* from the query, never from the tool schema. Three fields are
dropped when the **index is built**, so they cannot reach a prompt even by mistake:

| dropped | why |
|---|---|
| `default` on every parameter | The synthesis pipeline seeded gold from defaults. `Wayfair: reviews/list` has `sku` default `W004939121` and `page` default `"1"` — both exactly the gold values. 30% of required and 51% of optional gold values equal their default. |
| `code` | Contains the querystring dict, e.g. `{"page": page, "sku": sku, "sort_order": sort_order}`, revealing precisely which optional parameters gold expects. Easy to overlook — it looks like inert documentation. |
| `connected tools` / `connect params` | Gold tools appear in each other's connected lists (274 of 1800 rows). Names only are kept server-side, for distractor sampling; never rendered. |

`output_info` is excluded by default, behind `benchmark.options.render_output_info`.

**The structural guarantee:** `render_tool` takes only an index entry and has no argument saying
whether the tool is gold. It cannot render the two roles differently, and a test asserts every
endpoint renders identically whichever role it plays across the whole split.

### Two residuals

1. **Enum documentation.** A `sort_order` documented as `RELEVANCE|HELPFUL|DATE_ASCENDING|...` does
   name the gold `HELPFUL` among six alternatives. Removing enum lists would make the parameter
   unanswerable rather than harder, and would diverge from any realistic tool-calling setting, so
   they stay.
2. **Measured visibility.** Over the test split, **23.6%** of gold values appear somewhere in their
   own tool's rendered block vs **16.2%** in a distractor's. The gap is almost entirely values of
   ≤4 characters (55% vs 45% — near chance); at 9+ characters it is 5.7% vs 1.9%, and the hits are
   enum members.

### Comparability

**Our numbers are not comparable to the paper's.** The reference builder
(`evaluation/tool_evaluation_model.py:418`) copies `required_parameters` and `optional_parameters`
whole, defaults included — it is *looser* than ours on defaults and *tighter* on
`parent tool description` (it drops it). A `needs_clone` test pins that fact so it fails if the
reference changes. `usage_em` is computed with the benchmark's own normalization (literal-eval →
whitespace → country codes → date → boolean → numeric), verified against
`utils/metrics.py` compiled straight out of the clone, so when you write "Usage" it means what they
mean.

---

## 6. The candidate pool

`tool_count` gold tools + `n_distractors` (default 10) same-domain distractors.

- **`tool_count` runs 2–11, not 2–10** (one row has 11), so pools are **12–21**.
- Distribution: `{2: 90, 3: 405, 4: 261, 5: 181, 6: 168, 7: 164, 8: 170, 9: 183, 10: 177, 11: 1}`.
- The gold-to-distractor ratio therefore runs 2:10 to 11:10. **Effective difficulty varies by item
  — always stratify by `tool_count`.**

**Determinism.** Distractor choice and tool order are seeded per instance from the row id via
`base._unit_hash`, never from module random state (which the training loop reseeds). The pool is
identical across runs, across splits, and across the image / no-image conditions. Order is fixed
but uncorrelated with gold membership, so position bias over a 20-tool list cannot masquerade as an
effect.

**Pool digests.** The manifest carries a per-item `pool_digest`, verified on load — a silent
sampler drift is an error, not a quiet change of experiment. The check applies only under the
manifest's `pool_defaults` (`n_distractors: 10, hard_negatives: 0`); any other pool is a
deliberately different experiment and the check steps aside **with a warning in the log**. If you
change the pool, say so in the write-up: results are not comparable across pool settings.

**`hard_negatives: M`** draws M distractors from the gold providers' own sibling endpoints
(same-domain, non-gold), then `connected tools`, then uniform fill.

> Phase-0 finding worth remembering: drawing hard negatives from `connected tools` alone — the
> obvious reading of the benchmark — yields **zero candidates for 1600 of 1800 rows (89%)**,
> because the connection graph is deliberately cross-domain. Siblings work on 99.7% of rows
> (1506 have ≥10, only 6 have none). That is why siblings are tier one.

**Domain pool sizes** (index, unique endpoints): Finance 265, Music 262, Weather 134, Education
117, News_Media 103, Travel 103, Gaming 72, eCommerce 53, Email 51. Smallest still leaves ≥40
non-gold, so 10 distractors are always available.

---

## 7. Metrics: which one to report

`score()` returns one number, selected by `benchmark.options.metric` (`full_em` default). Every
component rides in `Result.meta` → `PredictionRow.meta` → the per-item JSONL.

| component | definition |
|---|---|
| `parse_success` | Did the output parse as JSON matching the schema? |
| `tool_em` | Set equality of predicted vs gold tool names. |
| `tool_inclusion` | \|predicted ∩ gold\| / \|gold\|. |
| `required_em` | Over the shared tools, exact match on the required `{name: value}` mapping. |
| `optional_em` | Same for optional. **Never folded into the headline.** |
| `usage_em` | The reference metric: required **and** optional both matching, over shared tools. |
| **`full_em`** | **`tool_em == 1` and `required_em == 1`. The headline.** |

Stratification keys also in `meta`: `domain`, `tool_count`, `has_unseen_values`, `pool_size`.

Value comparison drops empty-valued parameters and normalizes exactly as the reference does, so
`"1"` == `1`, `" term "` == `"term"`, `"US"` in a comma-separated location == `"United States"`.

---

## 8. Running it

```bash
# artifacts, built once, committed, digest-verified on load
python -m skillpng.benchmarks.trajectbench.traject_build index
python -m skillpng.benchmarks.trajectbench.traject_build split

# baseline, then the image — same seed, same split, same eval.limit
uv run skillpng eval -c configs/traject.yaml --set eval.condition=none
uv run skillpng eval -c configs/traject.yaml \
  --set eval.condition=skillpng --set eval.image_path=runs/traject/.../best.png

# optimize
uv run skillpng train -c configs/traject.yaml

# read the run
python -m skillpng.benchmarks.trajectbench.traject_report \
  runs/traject/<model>/<run>/eval/test-none.jsonl \
  --against runs/traject/<model>/<run>/eval/test-skillpng.jsonl
```

The report prints parse success separately and the headline conditioned on it, the breakdown by
`tool_count` and by domain, the `has_unseen_values` subset, and the **paired** difference joined on
`item_id` with a bootstrap 95% CI. Use the paired delta — the claim is a paired difference, not two
independent intervals.

Cluster: `slurm/traject/qwen3.5-9B/`, one env var per knob, including `METRIC`, `N_DISTRACTORS`,
`HARD_NEGATIVES`, `DESCRIPTION_CHARS`, `RENDER_OUTPUT_INFO`.

`eval.condition=noise` is the control for "does any image at all move this" — worth running here,
because a long structured prompt gives an image more surface to disrupt than SearchQA does.

---

## 9. Known limitations

**No scaffolding mask.** The loss is CE over `item.target` as a *string*. `VLM.loss_on_target`
tokenizes prompt and target separately and takes CE over the whole target span; there is no seam
through which JSON scaffolding (`{"tool name": ...`, brackets, quotes) could be masked. A share of
the CE mass lands on punctuation the model already produces correctly. Fixing it means changing the
shared `VLM` contract. Mitigated, not solved, by trimming the target to the three scored keys:
`tool description` (300–600 chars, already in the prompt) and `parent tool name` / `API name` /
`domain name` (derivable from `tool name`) are excluded.

**Tools straddle splits.** See §3 — unavoidable above domain granularity.

**`has_unseen_values` is 8.2% of test (50 items).** Show the n.

**The filter is lexical, in both directions.** §4 — it drops rows a good model could have inferred
(coordinates, `1day`), and it keeps a tail of rows that matched on a 1–2 character value by
accident. A boundary-matched rule would trade 296 more rows for that tail.

**Finance is 8 rows.** The filter took 192 of its 200. Three splits, 0 of them in val. Not a
domain you can report on.

**The GitHub clone's tool index is stale.** It carries 674 unique endpoints and is missing 139 of
the 752 gold tools, affecting 60% of rows. The index is therefore built from the **Hub**
(`tools/all_tools.json`, 1,281 unique → 1,279 after normalization merges two double-space pairs),
where the join rate is **100%**. The clone is a parity fixture only; the adapter never imports it
and the parity tests skip when it is absent.

**Index duplicate resolution.** 95 of 1,281 names have duplicate entries disagreeing on
descriptions and, for four of them, on parameter lists. First occurrence in file order wins; the
shipped digest pins the result.

---

## 10. Quick reference

```
corpus            1800 rows, 9 configs × 200, parallel-simple only
filter            875 kept, 925 dropped as unanswerable (a required value not in the query)
splits            175 / 89 / 611 of the 875, grouped on (domain, gold provider set), seed 109,
                  743 groups
tool index        1279 endpoints, digest-verified, no default / code / connect params
pool              tool_count (2–11) gold + 10 distractors → 12–21 tools, seeded from row id
budgets           max_target_tokens 1024, max_sequence_tokens 12288  (worst case 6,868)
headline          full_em = tool_em ∧ required_em, every shipped row answerable from its prompt
report beside it  parse_success, and full_em | parsed
always stratify   tool_count, domain, has_unseen_values (n=50); Finance is 8 rows, skip it
not comparable    to the paper: unfiltered corpus there, and its prompt ships parameter defaults
```
