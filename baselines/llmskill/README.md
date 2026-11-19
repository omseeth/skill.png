# LLM-Skill baseline

One Markdown skill per benchmark, written in a single pass by GPT-5.5 at
medium reasoning effort over OpenRouter. The generator sees the benchmark's
description, its system prompt, how answers are scored and five training
examples with their gold answers. It never sees a target model, a rollout or
a score, and nothing it writes is revised, so one skill serves every model.

| | Skill.png | LLM-Skill |
|---|---|---|
| Obtained | gradient descent on the target model's CE | one generator call, no feedback |
| Per | benchmark and model | benchmark; every model gets the same skill |
| Injected | an image in the user turn | Markdown under its own `## Skill` section, after the system prompt |
| Selection | best.png on val | none; the first skill written is frozen |

The four skills run 516 to 1,136 words and cost $0.05 to $0.16 each.

## Files

- `generate.py`: builds the prompt, calls the generator and freezes the
  skill. Examples come from the train split, and an id that is also in val
  or test stops the run. A reply cut off before it ends writes nothing.
- `prompts/<benchmark>.md`: the prompt templates, with slots for the system
  prompt, the examples and, for T1 only, the task prompt.
- `skills/<benchmark>/<model>_<effort>_k<k>_seed<seed>/`: the frozen skills.
  `skill.md` is the skill, and `skill.meta.json` records the full prompt,
  the example ids, the raw reply, the token usage and cost, and how many
  word 8-grams the skill shares with the examples. An existing directory is
  never overwritten.
- `slurm/<benchmark>/<model>/<benchmark>_test_{nothink,think}_llmskill.sh`
  and `..._skillpng_llmskill.sh`: score the skill alone and beside the
  Skill.png image.
- `tests/test_llmskill.py`: the skill's place in the system turn, the
  conditions that read it and the `llmskill` result rows.

```bash
export OPENROUTER_API_KEY=...            # or --api-key-file
uv run python baselines/llmskill/generate.py --benchmark searchqa \
    --model openai/gpt-5.5 --reasoning-effort medium --k 5 --seed 42
```

The skill is scored with `eval.condition=llmskill`, or with
`skillpng_llmskill` together with an image:

```bash
uv run skillpng eval -c configs/searchqa.yaml --set eval.condition=llmskill \
    --set eval.skill_path=baselines/llmskill/skills/searchqa/gpt-5.5_medium_k5_seed42/skill.md
```

| Flag | Default | |
|---|---|---|
| `--k`, `--seed` | 5, 42 | examples shown (0 is zero-shot) and the seed they are drawn with |
| `--reasoning-effort` | unset | minimal, low, medium or high; sent only when given |
| `--temperature`, `--max-tokens` | unset | sent only when given |
| `--provider` | any | pins one OpenRouter provider, without fallbacks |
| `--lint-n` | 8 | n-gram length of the overlap count |
| `--dry-run` | off | prints the prompt and calls nothing |
