# LoRA baseline

One LoRA adapter per benchmark and model, fitted under the objective the
Skill.png image is optimized with, thinking off. It is a gradient-based
reference that changes the weights instead of the input, so it yields no
skill that could be carried to another model.

| | Skill.png | LoRA |
|---|---|---|
| Trained | one image, fed with every prompt | an adapter on the language model's `q/k/v/o_proj` and `gate/up/down_proj`; routed MoE experts, embeddings and towers stay frozen |
| Loss | CE on the target tokens | the same, plus the marker that closes the assistant turn (read off the chat template), so the adapter learns to stop |
| Recipe | Adam lr 0.03 cosine, β1 0.5, batch 12, 80 epochs | rank 16, alpha 32, dropout 0.05, AdamW lr 2e-4 with 5% warmup and cosine decay, grad clip 1.0, batch 8, 10 epochs |
| Selection | best.png on val | val once per epoch; step 0, the base model, competes too |

A fit that never beats the base model freezes a no-op adapter, which the
analysis reports like a failed image: the No Skill score with an asterisk.

## Files

- `finetune.py`: fits, selects and freezes the adapter. It goes to
  `<model_id>-lora/<benchmark>/<name>/` (or `--adapter-root`) beside its
  `lora.meta.json` and is never overwritten. history.jsonl and summary.json
  go to `<train.run_dir>/lora_<name>/`.
- `text_only.py`: vLLM serves LoRA on Nemotron 3 Nano Omni's language model
  class but not on the omni class around it. That adapter is therefore fitted
  on `language_model` alone and served on the text-only checkpoint this
  script writes (~59 GiB; the test scripts write it on first use).
- `slurm/<benchmark>/<model>/<benchmark>_train_lora.sh` and
  `<benchmark>_test_{nothink,think}_lora.sh`: fit and score.
- `tests/test_lora.py`: the targeted modules, the end-of-turn marker, the
  selection, the text-only route and the `lora` result rows.

```bash
uv run --extra lora python baselines/lora/finetune.py --benchmark searchqa \
    --set model.model_id=/path/to/qwen3_5-4B --set train.run_dir=runs/searchqa/qwen3.5-4B
```

The adapter is scored on the request the `none` baseline sends, with vLLM
applying it:

```bash
vllm serve /path/to/qwen3_5-4B --served-model-name /path/to/qwen3_5-4B \
    --enable-lora --max-lora-rank 16 --lora-modules lora=/path/to/qwen3_5-4B-lora/searchqa/<name>
uv run skillpng eval -c configs/searchqa.yaml \
    --set model.name=vllm --set model.model_id=/path/to/qwen3_5-4B \
    --set model.vllm.served_model_name=lora --set eval.condition=lora
```

| Flag | Default | |
|---|---|---|
| `--rank`, `--alpha`, `--dropout` | 16, 32, 0.05 | rank 8, 16, 32, 64 or 128, as vLLM accepts |
| `--lr`, `--epochs`, `--batch-size` | 2e-4, 10, 8 | items are accumulated one sequence at a time |
| `--warmup`, `--grad-clip` | 0.05, 1.0 | 0 turns clipping off |
| `--select-on` | val_metric | val_loss selects on teacher-forced CE, far cheaper |
| `--eval-every` | once per epoch | steps between val checks |
| `--target-modules` | the seven above | other targets add `_targets<sha8>` to the name |
| `--limit` | none | caps train and val for a smoke run |
| `--dry-run` | off | prints the first train item and its target, loads no weights |
