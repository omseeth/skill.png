# SkillOpt baseline

One Markdown skill per benchmark and target model, trained with SkillOpt
(Yang et al., "SkillOpt: Executive Strategy for Self-Evolving Agent Skills",
arXiv:2605.23904, [github.com/microsoft/SkillOpt](https://github.com/microsoft/SkillOpt)).
An optimizer LLM reads the target's scored train rollouts and proposes edits
to the skill, and an edit is kept only if it raises the val score.
SkillOpt's optimizer loop runs unchanged from a vendored copy. Only the
target side is this repository's:

| | Upstream SkillOpt | Here |
|---|---|---|
| Loop | rollouts, reflection, merging, ranking, update, val gate, slow update and meta skill at epoch ends | the same code, byte-identical |
| Recipe | 4 epochs, 40 rollouts per step, at most 4 edits per step decaying to 2 (cosine), patch updates | upstream's defaults |
| Target | its own env adapters and model backends | the benchmark's prompts, splits and scoring, served by vLLM, with the skill in the system turn |
| Decoding | per backend | that of the non-reasoning test evals (temperature 0.7, seed 42), thinking off |
| Optimizer | GPT-5.5 over Azure OpenAI by default, medium effort | GPT-6 Luna over OpenRouter, or Claude Sonnet 5.5 over Anthropic's API, both at extra-high effort |
| Test split | scored at the end of training | never read during training; scored by `skillpng eval` |

The skill is composed into the system turn with `injection.skill_template`,
exactly as `eval.condition=skillopt` composes it, so a skill is trained
under the messages it is scored with. The val gate selects on the
benchmark's reported metric.

## Files

- `train.py`: the entry point. It resolves the target config, checks that
  no id occurs in two splits, runs upstream's trainer and freezes the
  selected skill.
- `adapter.py`: `BenchmarkAdapter`, upstream's `EnvAdapter` over any
  registered benchmark. Train rollouts leave their trajectories, with the
  gold target, for reflection. Val rollouts leave scores only.
- `openrouter.py`: the optimizer over OpenRouter, through upstream's own
  request path, and the call log.
- `anthropic_api.py`: the optimizer over Anthropic's Messages API.
- `upstream/`: the vendored SkillOpt code at commit `32deaeb`; see
  [upstream/VENDORED.md](upstream/VENDORED.md).
- `seeds/<benchmark>.md`: the blank skill each run starts from.
- `skills/<benchmark>/<model>/<name>/`: the frozen skills. `skill.md` is
  the selected skill, and `skill.meta.json` records the recipe, the val
  scores, the accepted and rejected steps, and the optimizer's token usage
  and cost.
- `slurm/<benchmark>/<model>/<benchmark>_train_skillopt.sh` and
  `..._train_skillopt_sonnet.sh` (Nemotron 3 only): the train scripts
  (submit with `OPENROUTER_API_KEY=... sbatch`, or `ANTHROPIC_API_KEY=...`
  for Sonnet). The compute node needs internet access to reach the
  optimizer.
- `tests/test_skillopt.py`: checks the adapter, the optimizer route, a
  training run and the eval condition offline, and every vendored file
  against its upstream sha256.

```bash
export OPENROUTER_API_KEY=...            # or --api-key-file
MODEL=/path/to/qwen3.5-9b make serve     # in one shell

uv run --extra skillopt python baselines/skillopt/train.py --benchmark searchqa \
    --optimizer openai/gpt-6-luna --reasoning-effort xhigh \
    --set model.model_id=/path/to/qwen3.5-9b \
    --set train.run_dir=runs/searchqa/qwen3.5-9B
```

The target is set through the usual `--set` keys. Artifacts go to
`<train.run_dir>/skillopt_<name>/`, where `optimizer_calls.jsonl` holds one
line per optimizer call. The selected skill is frozen in
`skills/<benchmark>/<model>/<name>/`, and an existing directory there is
never overwritten. The run's own settings are flags of `train.py`:

| Flag | Default | |
|---|---|---|
| `--provider` | openrouter | where the optimizer is reached; `anthropic` reads `ANTHROPIC_API_KEY` |
| `--optimizer` | openai/gpt-5.5 | the optimizer's model id at the provider |
| `--reasoning-effort` | medium | the optimizer's reasoning effort |
| `--seed` | 42 | SkillOpt's train seed |
| `--init-skill` | the blank seed | skill to start from |
| `--limit` | all items | cap on the train and val items, for smoke runs; freezes nothing |
| `--workers` | 32 | target rollouts run at once |
| `--exec-workers` | 4 | T1 code executor processes run at once |
| `--resume` | off | continue a killed run from its last completed step |
| `--dry-run` | off | print the first train item's messages and the SkillOpt config, call nothing |
| `--probe-optimizer` | off | send one tiny optimizer request and print its stats |

Score a frozen skill with `eval.condition=skillopt` and
`eval.skill_path=<its skill.md>`.

## Deviations from the reference

1. Upstream's env adapters, loaders and rollouts are replaced by
   `adapter.py`, and every path into upstream's own target backend raises.
2. The splits are this repository's train and val splits, not ones built
   by upstream's loaders.
3. The test split is not scored at the end of training
   (`eval_test: false`).
4. SearchQA and LiveMathematicianBench use upstream's analyst prompts for
   these benchmarks. T1 and TRAJECT-Bench have none upstream and fall back
   to its generic prompts.
5. Only `searchqa.md` is upstream's blank skill. The other seeds are its
   placeholder under a retitled heading.
6. For Sonnet 5.5, upstream's `chat_optimizer` is replaced by a streamed
   Messages request with adaptive thinking. The prompts are unchanged.
