# UltraBreak baseline

One image optimized on Qwen3.5-4B with the method of UltraBreak (Cui et al.,
"Toward Universal and Transferable Jailbreak Attacks on Vision-Language
Models", ICLR 2026, [github.com/kaiyuanCui/UltraBreak](https://github.com/kaiyuanCui/UltraBreak)),
for injection into the other models. It runs on the Skill.png trainer
(`src/skillpng/train.py`). Only the view each item's loss sees and the loss
itself change; the Skill.png trainer has no rotation, zoom or TV penalty of its 
own:

| | Skill.png | UltraBreak |
|---|---|---|
| View | the image | the image as a patch at 2/3 of the canvas side, rotated ±15°, zoomed 0.8–1.2, at a random whole-pixel offset on white, one draw per item |
| Loss | CE on the target tokens | attention-mode semantic loss (τ 0.5), over the input embeddings, plus a total variation penalty on the bare image (weight 0.5) |
| Recipe | Adam lr 0.03 cosine, β1 0.5, gray init | Adam lr 0.01 constant, β1 0.9, uniform random init |

Validation and the saved image are the bare image, as UltraBreak feeds its
patch at test. best.png and latest.png can therefore be scored like any
Skill.png image. UltraBreak itself keeps the image of the last step
(latest.png). The train scripts validate only every 400 steps and at the
end.

## Files

- `semantic_loss.py`: port of `semantic_similarity_loss(mode="attention")`.
- `patch_view.py`: the random patch view. `sample_view` draws the angle and
  the zoom, `rotate_scale` applies both in one resample, and `place_patch`
  shifts the result over the canvas.
- `tv_loss.py`: the total variation penalty.
- `optimize.py`: `UltraBreakTrainer`, a `Trainer` subclass, and the entry
  point. history.jsonl adds the CE of the same forward (`ce`), the total
  variation (`tv`) and its weighted penalty (`prior_loss`), and each item's
  view (`rotation_deg`, `scale`, `offset_y`, `offset_x`). The sidecar gets
  an `ultrabreak` block with every setting below.
- `slurm/<benchmark>/qwen3.5-4B/<benchmark>_train_ultrabreak.sh`: the four
  train scripts.
- `tests/test_ultrabreak.py`: checks the loss against a CPU port of the
  reference, the view, the TV penalty, and the trainer.

```bash
uv run python baselines/ultrabreak/optimize.py -c configs/searchqa.yaml \
    --set model.model_id=/path/to/qwen3_5-4B --set train.run_dir=runs/searchqa/qwen3.5-4B \
    --tv-weight 0.5 --rotation-range -15 15 --scale-range 0.8 1.2
```

The optimizer is set through the usual `--set train.*` and `--set param.*`
keys. UltraBreak's own settings are flags of `optimize.py`, and all of them
default to the reference's values:

| Flag | Default | |
|---|---|---|
| `--tv-weight` | 0.5 | weight of the TV penalty; 0 leaves it out |
| `--rotation-range LOW HIGH` | -15 15 | angle in degrees, drawn per item |
| `--scale-range LOW HIGH` | 0.8 1.2 | zoom factor, drawn per item |
| `--patch-fraction` | 2/3 | side of the unzoomed patch relative to the canvas |
| `--background` | 1.0 | canvas value around the patch (white) |
| `--tau` | 0.5 | temperature of the semantic loss's attention |
| `--pe-alpha` | 0.01 | scale of its positional encoding |
| `--noise` | 1e-4 | std of the noise on the expected embeddings |

## Deviations from the reference

1. The reference splices the raw [0,1] patch into the processor's
   normalized pixel values (`patch_only=False`), while its white background
   is normalized. Here the canvas is composed in [0,1] and normalized as a
   whole by the model's own preprocessing, as at test.
2. Edges blend bilinearly into the background. The reference instead treats
   every non-black pixel as patch.
3. The canvas has the image's size (256 px), with the patch at 2/3 of its
   side, instead of a 224 px patch on a 336 px canvas. Training thus sees
   the patch at about 2/3 of the scale that val and test show it at.
4. The targets are the benchmarks' gold answers. There is no
   "You must start your response with" suffix and no shared target prefix.
5. The step budget and batch are Skill.png's (80 epochs, batch 12), not
   5000 full-batch steps. best.png is also selected on val.
6. The positional encoding is computed in float32. The reference uses
   bfloat16, which rounds positions above 256.
