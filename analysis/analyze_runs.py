"""Transfer table over the recorded eval runs.

Reads every runs/<benchmark>/<model>/results.jsonl and prints two tables,
one for non-reasoning and one for reasoning runs. Each has one block per
model: the no-skill baseline, then the LLM-Skill, the SkillOpt, the LoRA and
the Skill.png score, each with its delta against the baseline. The reasoning
table instead takes every delta, the baseline's too, against the same score
in non-reasoning mode. SkillOpt there is the skill optimized by GPT-6 Luna.

Two more tables repeat the baseline and Skill.png next to the combined
Skill.png + LLM-Skill score, whose delta is taken against Skill.png instead.
A non-reasoning table puts a random-noise image of the Skill.png size, the
control for Skill.png, next to Skill.png, both with their delta against the
baseline, and one more compares the SkillOpt skills of every optimizer LLM
with Skill.png on non-reasoning Nemotron, and with a Skill.png optimized
under the Claude Sonnet 5.5 skill and evaluated with it.

A last table puts the non-reasoning baseline of every model next to the GPT-6
Luna SkillOpt skill and the Skill.png image of Qwen3.5-4B injected into it,
on the benchmarks with such transfer or ensemble runs. Its Ensemble row
injects the Skill.png optimized on Qwen3.5-4B and Gemma-4-E4B-it at once,
with unit-norm gradients. Its UltraBreak row injects the baseline image
that baselines/ultrabreak optimized on Qwen3.5-4B.

One table per source in BEST_TRANSFERS puts the GPT-6 Luna or Claude
Sonnet 5.5 SkillOpt skill and the Skill.png image of that source cell into
every model. The source model's block shows its own run of the same skill
and image, read from its old/results.jsonl when the main file has none.

A resolution table puts the non-reasoning Qwen3.5-4B baseline next to its
Skill.png images trained and evaluated at each ablation size, whatever
--resolution says. Images below 256x256 are trained and evaluated upscaled
to it, the smallest size Qwen3.5 feeds its tower. Where images of several
training seeds were tested at one size, its cell shows their mean and sample
standard deviation, and a list below it each seed's score; a seed whose image
failed counts with its stand-in score. Images of a seed other than 42 enter
no other table.

A beta1 table puts the non-reasoning Qwen3.5-9B baseline on LiveMath next to
its Skill.png images trained with each Adam beta1 of the ablation. Every
other table reports the beta1 0.5 image there; the other values enter no
other table.

Each benchmark with ensemble runs gets one more table, a column per model:
the non-reasoning baseline, the model's own Skill.png and the transferred
Qwen3.5-4B one next to the Skill.png optimized on Qwen3.5-4B and
Gemma-4-E4B-it at once. The header marks which models were in that
ensemble and which were held out. Like the transfer table, it shows no
stand-ins.

Transfer and ensemble runs enter only the transfer and ensemble tables.
TRAJECT-Bench columns only use runs with --distractors distractor tools, 10 by default. Qwen  Skill.png 
columns only use images trained and evaluated at --resolution, 256 by  default. Gemma-4-12B-it runs are left out. A Skill.png image whose best  validation 
score in training never beat the no-skill validation score is not  reported, 
nor is a SkillOpt skill whose optimization never beat its seed skill, nor a
LoRA adapter selected at step 0, before any update. Their cells show the score
of the same request without them, marked with an asterisk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Only these are reported; anything else on disk is ignored.
BENCHMARKS = {
    "searchqa": "SearchQA",
    "traject": "TRAJECT",
    "livemathematicianbench": "LiveMath",
    "t1": "T1",
}

# Directory name to display name. Models missing here still render, under
# their directory name, so a new model does not silently drop out. Tables
# list the models in this order, by size.
MODELS = {
    "qwen3.5-4B": "Qwen3.5-4B",
    "gemma-4-E4B-it": "Gemma-4-E4B-it",
    "qwen3.5-9B": "Qwen3.5-9B",
    "nemotron-3-nano-omni-30B-A3B-reasoning": "Nemotron-3-Nano-Omni-30B-A3B",
}
# Never reported. Gemma-4-12B-it needs its own training recipe, so its
# Skill.png runs are not comparable to the other models.
EXCLUDED_MODELS = {"gemma-4-12B-it"}

BASELINE_CONDITION = "none"
INJECTED_CONDITION = "skillpng"
LLMSKILL_CONDITION = "llmskill"
# The skill trained by baselines/skillopt, also in the system turn. Run
# names carry the optimizer LLM right after this segment.
SKILLOPT_CONDITION = "skillopt"
# Optimizer LLM, as run names spell it, to display name.
OPTIMIZERS = {
    "gpt-6-luna": "GPT-6 Luna",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
}
# The optimizer whose skills the main tables report.
MAIN_OPTIMIZER = "gpt-6-luna"
# The model the optimizer comparison is reported for, non-reasoning only.
OPTIMIZER_MODEL = "nemotron-3-nano-omni-30B-A3B-reasoning"
# The optimizer whose skill the optimizer comparison's skill-conditioned
# image was trained and evaluated with.
CONDITIONED_OPTIMIZER = "claude-sonnet-5-5"
# The request of the baseline, answered with a LoRA adapter trained by
# baselines/lora. Run names carry the adapter's name after this segment.
LORA_CONDITION = "lora"
LORA_NAME = re.compile(r"_lora_(.+?)(?:_reasoning)?_temperature")
# The optimized image and the LLM-Skill in the same request.
COMBINED_CONDITION = "skillpng_llmskill"
# An image optimized with a skill in the system turn, evaluated with that
# skill. Under a SkillOpt skill, run names carry the optimizer LLM after the
# SkillOpt segment.
CONDITIONED_CONDITION = "skillpng_conditioned"
# A random-noise image injected where Skill.png goes.
NOISE_CONDITION = "noise"
SEED = 42
# Training seed of every reported image. Images of other seeds are
# replicates, whose rows load_rows gives conditions of their own, suffixed
# like this, so no table but the resolution one reads them.
TRAIN_SEED = 42
SEED_SUFFIX = re.compile(r"_seed(\d+)$")

# Runs whose image or skill was optimized on another model carry this in
# their name. They are never reported as a model's own Skill.png or SkillOpt.
TRANSFER_TAG = "TRANSFER"
# The model the transferred images and skills were optimized on. Its own run
# of the same image or skill fills its row of the transfer table, its
# main-table run where nothing of that condition was transferred yet.
TRANSFER_SOURCE = "qwen3.5-4B"
# Condition the transfer table keys the transferred image under.
TRANSFER_CONDITION = "transfer"
# Condition it keys the transferred SkillOpt skill under.
TRANSFER_SKILLOPT_CONDITION = "transfer_skillopt"
# Condition it keys the transferred image optimized for transfer under. No
# row shows it; it only keeps that image out of the plain image's cell.
TRANSFER_AUGMENT_CONDITION = "transfer_augment"
# Run-name segment of such images, the TV weight and the rotation and zoom
# ranges, e.g. _tv0.003_rot-10to10_scale0.9to1.1_.
AUGMENT_PATTERN = re.compile(r"_tv[^_]+_rot[^_]+_scale[^_]+_")
# Condition it keys the transferred UltraBreak image under, and the run-name
# segment of such runs.
TRANSFER_ULTRABREAK_CONDITION = "transfer_ultrabreak"
ULTRABREAK_PATTERN = re.compile(r"_ultrabreak_")
# Transfer-table condition of each transferred row, by the row's condition.
# Only the main optimizer's skills are reported.
TRANSFER_CONDITIONS = {
    INJECTED_CONDITION: TRANSFER_CONDITION,
    f"{SKILLOPT_CONDITION}_{MAIN_OPTIMIZER}": TRANSFER_SKILLOPT_CONDITION,
}

# Runs of an image optimized against several models at once carry this in
# their name. Without TRANSFER in it, so main() alone decides which tables
# they enter.
ENSEMBLE_TAG = "_ENSEMBLE_"
# The models the ensemble images were optimized against, the primary first;
# its directory holds their training runs.
ENSEMBLE_MEMBERS = ("qwen3.5-4B", "gemma-4-E4B-it")
# Columns of the ensemble tables, the members first, then the held-out ones.
ENSEMBLE_MODELS = (*ENSEMBLE_MEMBERS, "qwen3.5-9B",
                   "nemotron-3-nano-omni-30B-A3B-reasoning")
# Condition the ensemble tables key the ensemble image under.
ENSEMBLE_CONDITION = "ensemble"
# Run-name segment of ensemble images trained with unit-norm gradients.
GRAD_NORM_PATTERN = re.compile(r"_gradnorm_")
# Ensemble-table condition of the ensemble image the transfer table reports.
TRANSFER_ENSEMBLE_CONDITION = f"{ENSEMBLE_CONDITION}_gradnorm"

# A run whose optimization found nothing better than no skill shows the score
# of the same request without the optimized part: condition to its stand-in,
# No Skill unless listed. An image trained under a SkillOpt skill falls back
# to that skill alone.
STAND_INS = {COMBINED_CONDITION: LLMSKILL_CONDITION,
             **{f"{CONDITIONED_CONDITION}_{optimizer}":
                f"{SKILLOPT_CONDITION}_{optimizer}"
                for optimizer in OPTIMIZERS}}
# Marks a cell that shows a stand-in score.
STAND_IN_MARK = "*"
# SkillOpt summaries name a best skill that no edit beat like this.
SEED_ORIGIN = "initial_skill"

# TRAJECT-Bench's shipped pool size. Rows of benchmarks without distractors
# carry no n_distractors and are always reported.
DISTRACTORS = 10

# Image side length the Skill.png runs of models with this prefix are
# reported at. Other models keep the one size they were run at.
RESOLUTION = 256
RESOLUTION_PREFIX = "qwen"
# Image side lengths of the resolution ablation, and the model it ran on.
# Its table shows every benchmark with runs at more than one of them.
RESOLUTIONS = (64, 128, 256, 320, 512, 768)
RESOLUTION_MODEL = "qwen3.5-4B"
# Smallest side the models with RESOLUTION_PREFIX feed their tower. Smaller
# images are trained and evaluated upscaled to it.
MIN_FED = 256
# Adam beta1 of the reported image of the beta1 ablation's model and
# benchmark. Rows of its images of other values get conditions of their
# own from load_rows, suffixed like this, so only the beta1 table reads them.
REPORTED_BETA1 = 0.5
BETA1_SUFFIX = re.compile(r"_b1(\d+(?:\.\d+)?)$")
# Torch's default, which training run names leave out. They name any other
# value in a segment like this, e.g. _b10.5_.
TORCH_BETA1 = 0.9
BETA1_SEGMENT = re.compile(r"_b1(\d+(?:\.\d+)?)_")
# Values of the beta1 ablation, and the model and benchmark it ran on.
BETA1S = (0.1, 0.5, 0.9)
BETA1_MODEL = "qwen3.5-9B"
BETA1_BENCHMARK = "livemathematicianbench"
# Image digest read for rows that inject no image, which carry none.
NO_IMAGE = "none"
# Skill digest of rows without a skill, and of rows older than the field.
NO_SKILL = "none"
# Height and width of an image, and such sizes keyed by benchmark, model and
# image digest.
Size = tuple[int, ...]
Sizes = dict[tuple[str, str, str], Size]
# Rows keyed by benchmark, model, reasoning and condition.
Keyed = dict[tuple[str, str, bool, str], dict[str, Any]]
# Score and stand-in flag of each training seed, keyed by benchmark and the
# cell key of an image size.
SeedScores = dict[tuple[str, str], dict[int, tuple[float, bool]]]

# Cell key and display name of each row in a model's block.
ROWS = (("baseline", "No Skill"), ("llmskill", "LLM-Skill"),
        ("skillopt", "SkillOpt"), ("lora", "LoRA"), ("injected", "Skill.png"))
# Rows of the non-reasoning random-image control table.
NOISE_ROWS = (("noise", "Random Image"), ("injected", "Skill.png"))
# Rows of the separate tables for the combined experiment.
COMBINED_ROWS = (("baseline", "No Skill"), ("injected", "Skill.png"),
                 ("combined", "Skill.png + LLM-Skill"))
# Cell key each row's delta is taken against, No Skill unless listed. The
# combined row is compared with Skill.png, to show what the LLM-Skill adds.
DELTA_REFERENCES = {"combined": "injected"}
# Rows of the optimizer comparison, keyed by optimizer, then Skill.png and
# the image trained and evaluated with the CONDITIONED_OPTIMIZER skill.
OPTIMIZER_ROWS = (("baseline", "No Skill"),
                  *((optimizer, f"SkillOpt ({name})")
                    for optimizer, name in OPTIMIZERS.items()),
                  ("injected", "Skill.png"),
                  ("conditioned", "SkillOpt + Skill.png"))
# Cells whose SkillOpt skill and Skill.png image are transferred to every
# other model: benchmark, source model and the SkillOpt optimizer.
BEST_TRANSFERS = (
    ("t1", "nemotron-3-nano-omni-30B-A3B-reasoning", "claude-sonnet-5-5"),
    ("t1", "qwen3.5-9B", "gpt-6-luna"),
    ("searchqa", "gemma-4-E4B-it", "gpt-6-luna"),
    ("livemathematicianbench", "nemotron-3-nano-omni-30B-A3B-reasoning",
     "claude-sonnet-5-5"),
)
# Rows of each such table.
BEST_TRANSFER_ROWS = (("baseline", "No Skill"),
                      ("transfer_skillopt", "SkillOpt"),
                      ("transfer", "Skill.png"))
# Rows of the transfer table.
TRANSFER_ROWS = (("baseline", "No Skill"),
                 ("transfer_skillopt",
                  f"SkillOpt ({MODELS[TRANSFER_SOURCE]})"),
                 ("transfer_ultrabreak",
                  f"UltraBreak ({MODELS[TRANSFER_SOURCE]})"),
                 ("transfer", f"Skill.png ({MODELS[TRANSFER_SOURCE]})"),
                 (TRANSFER_ENSEMBLE_CONDITION, "Ensemble"))
# Row name of each ablation size, with the fed size of upscaled images.
RESOLUTION_LABELS = {
    size: f"Skill.png {size}x{size}"
    + (f" (fed {MIN_FED}x{MIN_FED})" if size < MIN_FED else "")
    for size in RESOLUTIONS}
# Rows of the resolution ablation, one per image size.
RESOLUTION_ROWS = (("baseline", "No Skill"),
                   *((f"res{size}", label)
                     for size, label in RESOLUTION_LABELS.items()))
# Row name of each beta1 of the ablation, the reported one marked.
BETA1_LABELS = {
    beta1: f"Skill.png, beta1 {beta1}"
    + (" (reported)" if beta1 == REPORTED_BETA1 else "")
    for beta1 in BETA1S}
# Rows of the beta1 ablation, one per value.
BETA1_ROWS = (("baseline", "No Skill"),
              *((f"b1{beta1}", label)
                for beta1, label in BETA1_LABELS.items()))
# Rows of the ensemble tables. Each variant of the ensemble image has its
# own row; the starred ones were trained under the transfer regularizers.
ENSEMBLE_ROWS = (("baseline", "No Skill"),
                 ("injected", "Skill.png (own)"),
                 ("transfer", f"Skill.png ({MODELS[TRANSFER_SOURCE]})"),
                 (ENSEMBLE_CONDITION, "Skill.png (ensemble)"),
                 (f"{ENSEMBLE_CONDITION}_gradnorm",
                  "Skill.png (ensemble, unit-norm grads)"),
                 (f"{ENSEMBLE_CONDITION}_augment", "Skill.png* (ensemble)"),
                 (f"{ENSEMBLE_CONDITION}_augment_gradnorm",
                  "Skill.png* (ensemble, unit-norm grads)"))

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def is_reasoning(run_name: str) -> bool:
    """Tell whether a run was evaluated in reasoning mode.

    Args:
        run_name: Run name recorded on the result row.
    Returns:
        True when the run is a reasoning-mode run.
    """
    # Reasoning is not its own field, only a run_name segment. Matching the
    # whole _reasoning_temperature segment keeps the nemotron model name,
    # which ends in -reasoning, from reading as a reasoning run.
    return "_reasoning_temperature" in run_name


def is_transfer(row: dict) -> bool:
    """Tell whether a row evaluated an image or skill from another model.

    Args:
        row: Result row.
    Returns:
        True when the run name carries the transfer tag.
    """
    return TRANSFER_TAG in row["run_name"]


def is_ensemble(row: dict) -> bool:
    """Tell whether a row evaluated an image optimized on several models.

    Args:
        row: Result row.
    Returns:
        True when the run name carries the ensemble tag.
    """
    return ENSEMBLE_TAG in row["run_name"]


def ensemble_condition(row: dict) -> str:
    """Name the ensemble-table condition of an ensemble image's row.

    Args:
        row: Result row of an ensemble image.
    Returns:
        The condition, suffixed per training variant.
    """
    condition = ENSEMBLE_CONDITION
    if AUGMENT_PATTERN.search(row["run_name"]):
        condition += "_augment"
    if GRAD_NORM_PATTERN.search(row["run_name"]):
        condition += "_gradnorm"
    return condition


def transfer_tag(model: str) -> str:
    """Spell a model as transfer run names tag it.

    Args:
        model: Model directory name.
    Returns:
        Upper-case tag, e.g. QWEN3.5_4B.
    """
    return model.upper().replace("-", "_")


def transfer_from(row: dict) -> str | None:
    """Name the model a transfer row's image or skill was optimized on.

    Args:
        row: Result row.
    Returns:
        Model directory name, or None when the tag names no known model.
    """
    tag = row["run_name"].rpartition(f"_{TRANSFER_TAG}_")[2]
    return {transfer_tag(model): model for model in MODELS}.get(tag)


def transfer_source(row: dict) -> tuple[str, str, str, str]:
    """Identify what a row injected, to pair transfer runs with the source's.

    Args:
        row: Result row.
    Returns:
        Benchmark, condition, image digest and skill digest.
    """
    return (row["benchmark"], row["condition"],
            row.get("image_digest", NO_IMAGE),
            row.get("skill_digest", NO_SKILL))


def transfer_condition(row: dict) -> str:
    """Name the transfer-table condition a transferred image or skill fills.

    Args:
        row: Result row of a condition in TRANSFER_CONDITIONS.
    Returns:
        Transfer-table condition, its own one for an UltraBreak or
        augmented image.
    """
    if row["condition"] == INJECTED_CONDITION:
        if ULTRABREAK_PATTERN.search(row["run_name"]):
            return TRANSFER_ULTRABREAK_CONDITION
        if AUGMENT_PATTERN.search(row["run_name"]):
            return TRANSFER_AUGMENT_CONDITION
    return TRANSFER_CONDITIONS[row["condition"]]


def skillopt_condition(optimizer: str) -> str:
    """Name the condition of the SkillOpt skill from one optimizer LLM.

    Args:
        optimizer: Optimizer LLM as run names spell it.
    Returns:
        Condition name that keeps this optimizer's runs apart.
    """
    return f"{SKILLOPT_CONDITION}_{optimizer}"


def conditioned_condition(optimizer: str) -> str:
    """Name the condition of an image trained under one optimizer's skill.

    Args:
        optimizer: Optimizer LLM as run names spell it.
    Returns:
        Condition name that keeps this optimizer's runs apart.
    """
    return f"{CONDITIONED_CONDITION}_{optimizer}"


def resolution_condition(resolution: int) -> str:
    """Name the condition of the Skill.png image of one ablation size.

    Args:
        resolution: Image side length.
    Returns:
        Condition name that keeps this size's runs apart.
    """
    return f"{INJECTED_CONDITION}_res{resolution}"


def fed_side(resolution: int) -> int:
    """Find the side length an image is fed to the tower at.

    Args:
        resolution: Image side length.
    Returns:
        The side length, raised to MIN_FED for smaller images.
    """
    return max(resolution, MIN_FED)


def seed_condition(condition: str, seed: int) -> str:
    """Name the condition of a row whose image was trained with one seed.

    Args:
        condition: Condition of the row.
        seed: Training seed of the row's image.
    Returns:
        The condition, suffixed with any seed other than TRAIN_SEED.
    """
    return condition if seed == TRAIN_SEED else f"{condition}_seed{seed}"


def split_seed(condition: str) -> tuple[str, int]:
    """Undo seed_condition.

    Args:
        condition: Condition, suffixed with a training seed or not.
    Returns:
        The condition without the suffix, and the training seed.
    """
    match = SEED_SUFFIX.search(condition)
    if not match:
        return condition, TRAIN_SEED
    return condition[:match.start()], int(match.group(1))


def beta1_condition(condition: str, beta1: float) -> str:
    """Name the condition of a row whose image was trained with one beta1.

    Args:
        condition: Condition of the row.
        beta1: Adam beta1 the row's image was trained with.
    Returns:
        The condition, suffixed with any beta1 other than REPORTED_BETA1.
    """
    if beta1 == REPORTED_BETA1:
        return condition
    return f"{condition}_b1{beta1}"


def split_beta1(condition: str) -> tuple[str, float]:
    """Undo beta1_condition.

    Args:
        condition: Condition, suffixed with a beta1 or not.
    Returns:
        The condition without the suffix, and the beta1.
    """
    match = BETA1_SUFFIX.search(condition)
    if not match:
        return condition, REPORTED_BETA1
    return condition[:match.start()], float(match.group(1))


def load_image_beta1s(runs_dir: Path) -> dict[tuple[str, str, str], float]:
    """Read the Adam beta1 every image was trained with.

    Args:
        runs_dir: Directory holding benchmark/model/run training trees.
    Returns:
        Beta1 of each image, keyed by benchmark, model and digest.
    """
    beta1s = {}
    for name in ("best.json", "latest.json"):
        for path in runs_dir.glob(f"*/*/*/{name}"):
            sidecar = json.loads(path.read_text())
            run = path.parent
            key = (run.parent.parent.name, run.parent.name, sidecar["digest"])
            # Sidecars older than the field leave it to the run name.
            beta1 = sidecar.get("beta1")
            if beta1 is None:
                match = BETA1_SEGMENT.search(run.name)
                beta1 = float(match.group(1)) if match else TORCH_BETA1
            beta1s[key] = beta1
    return beta1s


def load_image_seeds(runs_dir: Path) -> dict[tuple[str, str, str], int]:
    """Read the training seed of every image from its sidecar.

    Args:
        runs_dir: Directory holding benchmark/model/run training trees.
    Returns:
        Training seed of each image, keyed by benchmark, model and digest.
    """
    seeds = {}
    for name in ("best.json", "latest.json"):
        for path in runs_dir.glob(f"*/*/*/{name}"):
            sidecar = json.loads(path.read_text())
            run = path.parent
            key = (run.parent.parent.name, run.parent.name, sidecar["digest"])
            # Sidecars older than the seed field all come from TRAIN_SEED.
            seeds[key] = sidecar.get("seed", TRAIN_SEED)
    return seeds


def load_rows(runs_dir: Path, seed: int,
              pattern: str = "*/*/results.jsonl") -> list[dict]:
    """Read every result row for one seed.

    Args:
        runs_dir: Directory holding benchmark/model/results.jsonl files.
        seed: Eval seed to keep; rows from other seeds are dropped.
        pattern: Glob of the results files below runs_dir.
    Returns:
        Result rows with benchmark, model, reasoning and train_seed filled in.
    """
    seeds = load_image_seeds(runs_dir)
    beta1s = load_image_beta1s(runs_dir)
    rows = []
    for path in sorted(runs_dir.glob(pattern)):
        benchmark, model = path.relative_to(runs_dir).parts[:2]
        if benchmark not in BENCHMARKS or model in EXCLUDED_MODELS:
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("seed") != seed:
                continue
            row["benchmark"] = benchmark
            row["model"] = model
            row["reasoning"] = is_reasoning(row.get("run_name", ""))
            # Split SkillOpt, and images trained under a SkillOpt skill, per
            # optimizer, so the newest run of one optimizer never replaces
            # another's.
            segment = f"_{SKILLOPT_CONDITION}_"
            if row["condition"] == SKILLOPT_CONDITION:
                optimizer = row["run_name"].split(segment)[1].split("_")[0]
                row["condition"] = skillopt_condition(optimizer)
            elif (row["condition"] == CONDITIONED_CONDITION
                  and segment in row["run_name"]):
                optimizer = row["run_name"].split(segment)[1].split("_")[0]
                row["condition"] = conditioned_condition(optimizer)
            # Split the beta1 ablation's images per beta1, and all images
            # per training seed, so neither replaces the reported image as
            # the newest run.
            image = (benchmark, model, row.get("image_digest", NO_IMAGE))
            if (benchmark, model) == (BETA1_BENCHMARK, BETA1_MODEL):
                row["condition"] = beta1_condition(
                    row["condition"], beta1s.get(image, REPORTED_BETA1))
            row["train_seed"] = seeds.get(image, TRAIN_SEED)
            row["condition"] = seed_condition(row["condition"],
                                              row["train_seed"])
            rows.append(row)
    return rows


def load_failed_images(runs_dir: Path) -> dict[tuple, str]:
    """Find the trained images that never beat the no-skill validation score.

    Args:
        runs_dir: Directory holding benchmark/model/run training trees.
    Returns:
        Why each such image failed, keyed by benchmark, model and digest.
    """
    failed = {}
    for name in ("best.json", "latest.json"):
        for path in runs_dir.glob(f"*/*/*/{name}"):
            sidecar = json.loads(path.read_text())
            clean = sidecar.get("val_clean_score")
            # Under val_loss selection best_score is a negated loss, which no
            # clean score compares against.
            if sidecar.get("select_on") != "val_metric" or clean is None:
                continue
            if sidecar["best_score"] <= clean:
                run = path.parent
                key = (run.parent.parent.name, run.parent.name,
                       sidecar["digest"])
                # Images trained under a skill compare with that skill alone.
                baseline = ("skill-only" if sidecar.get("skill_digest")
                            else "No Skill")
                failed[key] = (f"best val {sidecar['best_score'] * 100:.2f}"
                               f" <= {baseline} val {clean * 100:.2f} in"
                               f" {run.name}")
    return failed


def load_image_sizes(runs_dir: Path) -> Sizes:
    """Read the trained size of every image from its sidecar.

    Args:
        runs_dir: Directory holding benchmark/model/run training trees.
    Returns:
        Height and width of each image, keyed by benchmark, model and digest.
    """
    sizes: Sizes = {}
    for name in ("best.json", "latest.json"):
        for path in runs_dir.glob(f"*/*/*/{name}"):
            sidecar = json.loads(path.read_text())
            run = path.parent
            key = (run.parent.parent.name, run.parent.name, sidecar["digest"])
            sizes[key] = tuple(sidecar["shape"][-2:])
    return sizes


def image_sizes(row: dict[str, Any], sizes: Sizes) -> tuple[Size, Size]:
    """Find the trained and the evaluated size of a row's image.

    Args:
        row: Result row that injects an image.
        sizes: Trained image sizes from load_image_sizes.
    Returns:
        Trained and evaluated height and width.
    """
    # vLLM resizes the image to the eval resolution, so a row alone cannot
    # tell a 320 image evaluated at 256 from a 256 one. Images no training
    # run on disk has count as trained at the eval size.
    fed = tuple(row["resolution"])
    key = (row["benchmark"], row["model"], row["image_digest"])
    return sizes.get(key, fed), fed


def at_resolution(row: dict[str, Any], sizes: Sizes,
                  resolution: int) -> bool:
    """Tell whether a row may be reported at the requested image size.

    Args:
        row: Result row.
        sizes: Trained image sizes from load_image_sizes.
        resolution: Side length reported for models with RESOLUTION_PREFIX.
    Returns:
        False only for such a model's image trained or fed at another size.
    """
    if (not row["model"].startswith(RESOLUTION_PREFIX)
            or row.get("image_digest", NO_IMAGE) == NO_IMAGE):
        return True
    trained, fed = image_sizes(row, sizes)
    side = fed_side(resolution)
    return trained == (resolution, resolution) and fed == (side, side)


def load_failed_skills(runs_dir: Path) -> dict[tuple, str]:
    """Find the SkillOpt skills whose optimization never beat the seed skill.

    Args:
        runs_dir: Directory holding benchmark/model/run SkillOpt trees.
    Returns:
        Why each such skill failed, keyed by benchmark, model and digest.
    """
    failed = {}
    pattern = f"*/*/{SKILLOPT_CONDITION}_*/summary.json"
    for path in runs_dir.glob(pattern):
        summary = json.loads(path.read_text())
        if summary.get("best_origin") != SEED_ORIGIN:
            continue
        run = path.parent
        # Eval rows hash the stripped skill text.
        skill = (run / "best_skill.md").read_text(encoding="utf-8").strip()
        key = (run.parent.parent.name, run.parent.name,
               hashlib.sha256(skill.encode()).hexdigest())
        failed[key] = (f"no edit beat the seed skill, val"
                       f" {summary['best_selection_hard'] * 100:.2f}, in"
                       f" {run.name}")
    return failed


def lora_name(row: dict) -> str | None:
    """Read the adapter's name off a LoRA row's run name.

    Args:
        row: Result row of the LoRA condition.
    Returns:
        The adapter's name, or None when the run name does not carry it.
    """
    match = LORA_NAME.search(row.get("run_name", ""))
    return match.group(1) if match else None


def load_failed_loras(runs_dir: Path) -> dict[tuple, str]:
    """Find the LoRA adapters whose selection never moved past step 0.

    Args:
        runs_dir: Directory holding benchmark/model/run LoRA trees.
    Returns:
        Why each such adapter failed, keyed by benchmark, model and name.
    """
    failed = {}
    for path in runs_dir.glob(f"*/*/{LORA_CONDITION}_*/summary.json"):
        summary = json.loads(path.read_text())
        selection = summary.get("selection", {})
        if selection.get("best_step") != 0:
            continue
        run = path.parent
        key = (run.parent.parent.name, run.parent.name, summary["name"])
        failed[key] = (f"no step beat the base model, val"
                       f" {selection['best_score']:.4f}"
                       f" ({selection['select_on']}), in {run.name}")
    return failed


def find_failed(selected: dict, images: dict, skills: dict,
                 loras: dict | None = None) -> dict:
    """Pick the runs whose optimization found nothing better than no skill.

    Args:
        selected: Rows keyed by benchmark, model, reasoning and condition.
        images: Reasons of failed images keyed by benchmark, model, digest.
        skills: Reasons of failed SkillOpt skills, keyed the same way.
        loras: Reasons of failed LoRA adapters, keyed by adapter name.
    Returns:
        Display name and reason of each failed run, keyed like selected.
    """
    skillopts = {skillopt_condition(optimizer): name
                 for optimizer, name in OPTIMIZERS.items()}
    # Images other than the main Skill.png, by condition.
    sizes = {resolution_condition(size): label
             for size, label in RESOLUTION_LABELS.items()}
    labels = dict(sizes)
    labels.update({conditioned_condition(optimizer):
                   f"SkillOpt ({name}) + Skill.png"
                   for optimizer, name in OPTIMIZERS.items()})
    labels.update({beta1_condition(INJECTED_CONDITION, beta1): label
                   for beta1, label in BETA1_LABELS.items()
                   if beta1 != REPORTED_BETA1})
    # Replicates of an ablation size, named after their training seed.
    for *_, condition in selected:
        size, seed = split_seed(condition)
        if size in sizes and seed != TRAIN_SEED:
            labels[condition] = f"{sizes[size]}, seed {seed}"
    failed = {}
    for key, row in selected.items():
        benchmark, model, reasoning, condition = key
        if condition == INJECTED_CONDITION:
            reason = images.get((benchmark, model, row["image_digest"]))
            if not reason:
                continue
            failed[key] = ("Skill.png", reason)
            # The Skill.png row decides for the combined runs too. They inject
            # the same best.png, but some carry a digest no training run on
            # disk has.
            combined = (benchmark, model, reasoning, COMBINED_CONDITION)
            if combined in selected:
                failed[combined] = ("Skill.png + LLM-Skill", reason)
        elif condition in skillopts:
            reason = skills.get((benchmark, model, row.get("skill_digest")))
            if reason:
                failed[key] = (f"SkillOpt ({skillopts[condition]})", reason)
        elif condition in labels:
            reason = images.get((benchmark, model, row["image_digest"]))
            if reason:
                failed[key] = (labels[condition], reason)
        elif condition == LORA_CONDITION:
            reason = (loras or {}).get((benchmark, model, lora_name(row)))
            if reason:
                failed[key] = ("LoRA", reason)
    return failed


def stand_in_row(selected: dict, key: tuple) -> dict | None:
    """Look up the row whose score stands in for a failed run.

    Args:
        selected: Rows keyed by benchmark, model, reasoning and condition.
        key: Key of the failed run.
    Returns:
        The stand-in row, or None when it is missing.
    """
    benchmark, model, reasoning, condition = key
    return selected.get((benchmark, model, reasoning,
                         STAND_INS.get(condition, BASELINE_CONDITION)))


def select_latest(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Keep the newest row per benchmark, model, reasoning mode and condition.

    Args:
        rows: Result rows to reduce.
    Returns:
        Tuple of the selected rows keyed by that tuple and the superseded rows.
    """
    by_key: dict = {}
    for row in rows:
        key = (row["benchmark"], row["model"], row["reasoning"],
               row["condition"])
        by_key.setdefault(key, []).append(row)

    selected = {}
    superseded = []
    for key, group in by_key.items():
        group.sort(key=lambda row: row["timestamp"])
        selected[key] = group[-1]
        superseded.extend(group[:-1])
    return selected, superseded


def select_transfer(rows: list[dict],
                    selected: dict) -> tuple[dict, list[str]]:
    """Key the non-reasoning transfer runs and their baselines like selected.

    Args:
        rows: Result rows, transfer runs included.
        selected: Main-table rows keyed by benchmark, model, mode, condition.
    Returns:
        Baseline and transferred image and skill rows, and their benchmarks.
    """
    transfer = [row for row in rows
                if is_transfer(row) and not row["reasoning"]
                and transfer_from(row) == TRANSFER_SOURCE
                and row["condition"] in TRANSFER_CONDITIONS]
    sources = {transfer_source(row) for row in transfer}
    # The source model's own run of each transferred image or skill.
    own = [row for row in rows
           if row["model"] == TRANSFER_SOURCE and not row["reasoning"]
           and not is_transfer(row) and transfer_source(row) in sources]
    # Keyed by transfer-table condition, so the augmented image never
    # replaces the plain one.
    latest, _ = select_latest([{**row, "condition": transfer_condition(row)}
                               for row in transfer + own])
    benchmarks = [benchmark for benchmark in BENCHMARKS
                  if any(row["benchmark"] == benchmark for row in transfer)]
    # A condition not transferred yet shows the source model's main-table run,
    # also on benchmarks that only the ensemble image brings into the table.
    transferred = {(row["benchmark"], transfer_condition(row))
                   for row in transfer}
    for benchmark in BENCHMARKS:
        for condition, cell in TRANSFER_CONDITIONS.items():
            key = (benchmark, TRANSFER_SOURCE, False, condition)
            if (benchmark, cell) not in transferred and key in selected:
                latest[(benchmark, TRANSFER_SOURCE, False, cell)] = (
                    selected[key])

    picked = dict(latest)
    for benchmark, model, reasoning, _ in latest:
        baseline = (benchmark, model, reasoning, BASELINE_CONDITION)
        if baseline in selected:
            picked[baseline] = selected[baseline]
    return picked, benchmarks


def select_best_transfer(rows: list[dict], archived: list[dict],
                         selected: dict, benchmark: str, source: str,
                         optimizer: str) -> dict:
    """Key one source cell's transfer runs and their baselines like selected.

    Args:
        rows: Result rows, transfer runs included.
        archived: Rows of the old/results.jsonl files.
        selected: Main-table rows keyed by benchmark, model, mode, condition.
        benchmark: Benchmark of the source cell.
        source: Model the skill and image were optimized on.
        optimizer: Optimizer LLM of the SkillOpt skill.
    Returns:
        Baseline and transferred skill and image rows of every model.
    """
    cells = {INJECTED_CONDITION: TRANSFER_CONDITION,
             skillopt_condition(optimizer): TRANSFER_SKILLOPT_CONDITION}
    transfer = [row for row in rows
                if is_transfer(row) and not row["reasoning"]
                and row["benchmark"] == benchmark
                and transfer_from(row) == source
                and row["condition"] in cells
                and not AUGMENT_PATTERN.search(row["run_name"])
                and not ULTRABREAK_PATTERN.search(row["run_name"])]
    sources = {transfer_source(row) for row in transfer}
    # The source model's own run of the same skill or image, which may
    # predate its current results file.
    own = [row for row in rows + archived
           if row["model"] == source and row["benchmark"] == benchmark
           and not row["reasoning"] and not is_transfer(row)
           and transfer_source(row) in sources]
    picked, _ = select_latest([{**row, "condition": cells[row["condition"]]}
                               for row in transfer + own])
    for _, model, reasoning, _ in list(picked):
        baseline = (benchmark, model, reasoning, BASELINE_CONDITION)
        if baseline in selected:
            picked[baseline] = selected[baseline]
    return picked


def select_resolutions(
        rows: list[dict[str, Any]], sizes: Sizes,
) -> tuple[dict[tuple[str, str, bool, str], dict[str, Any]], list[str]]:
    """Key the ablation model's non-reasoning runs at every size like selected.

    Each training seed of a size keys its newest row under its own condition,
    from seed_condition.

    Args:
        rows: Result rows of every image size.
        sizes: Trained image sizes from load_image_sizes.
    Returns:
        Baseline and per-size Skill.png rows, and their benchmarks.
    """
    own = [row for row in rows
           if row["model"] == RESOLUTION_MODEL and not row["reasoning"]]
    picked = {}
    for size in RESOLUTIONS:
        images = [{**row, "condition": seed_condition(
                      resolution_condition(size), row["train_seed"])}
                  for row in own
                  if row["condition"] == seed_condition(INJECTED_CONDITION,
                                                        row["train_seed"])
                  and at_resolution(row, sizes, size)]
        picked.update(select_latest(images)[0])
    # A benchmark needs more than one size, not more than one seed.
    counts = Counter(benchmark for benchmark, _ in
                     {(key[0], split_seed(key[3])[0]) for key in picked})
    benchmarks = [benchmark for benchmark in BENCHMARKS
                  if counts[benchmark] > 1]
    baselines = [row for row in own if row["condition"] == BASELINE_CONDITION]
    picked.update(select_latest(baselines)[0])
    picked = {key: row for key, row in picked.items() if key[0] in benchmarks}
    return picked, benchmarks


def select_beta1(rows: list[dict[str, Any]], selected: Keyed) -> Keyed:
    """Key the beta1 ablation's non-reasoning runs like selected.

    Args:
        rows: Result rows of every beta1.
        selected: Main-table rows keyed by benchmark, model, mode, condition.
    Returns:
        Baseline and per-beta1 Skill.png rows, empty without another beta1.
    """
    varied = {beta1_condition(INJECTED_CONDITION, beta1)
              for beta1 in BETA1S if beta1 != REPORTED_BETA1}
    images = [row for row in rows
              if row["benchmark"] == BETA1_BENCHMARK
              and row["model"] == BETA1_MODEL and not row["reasoning"]
              and row["condition"] in varied]
    picked, _ = select_latest(images)
    if not picked:
        return {}
    # The reported beta1 is the main tables' Skill.png.
    for condition in (BASELINE_CONDITION, INJECTED_CONDITION):
        key = (BETA1_BENCHMARK, BETA1_MODEL, False, condition)
        if key in selected:
            picked[key] = selected[key]
    return picked


def select_ensemble(rows: list[dict], selected: dict,
                    transfer: dict) -> tuple[dict, list[str]]:
    """Key the ensemble image runs and the rows they are compared with.

    Args:
        rows: Result rows of ensemble images.
        selected: Main-table rows keyed by benchmark, model, mode, condition.
        transfer: Transfer-table rows from select_transfer, keyed the same.
    Returns:
        Baseline, own, transferred and ensemble rows, and their benchmarks.
    """
    images = [{**row, "condition": ensemble_condition(row)} for row in rows
              if not row["reasoning"]
              and row["condition"] == INJECTED_CONDITION]
    picked, _ = select_latest(images)
    benchmarks = [benchmark for benchmark in BENCHMARKS
                  if any(row["benchmark"] == benchmark for row in images)]
    for benchmark in benchmarks:
        for model in ENSEMBLE_MODELS:
            for condition, source in ((BASELINE_CONDITION, selected),
                                      (INJECTED_CONDITION, selected),
                                      (TRANSFER_CONDITION, transfer)):
                key = (benchmark, model, False, condition)
                if key in source:
                    picked[key] = source[key]
    return picked, benchmarks


def build_table(selected: dict, reasoning: bool, failed: dict) -> list[dict]:
    """Turn the selected rows of one reasoning mode into one entry per model.

    Args:
        selected: Rows keyed by benchmark, model, reasoning and condition.
        reasoning: Which reasoning mode to report.
        failed: Failed runs from find_failed, keyed like selected.
    Returns:
        List of entries carrying a model, label and per-benchmark scores.
    """
    models = {key[1] for key in selected}
    order = [name for name in MODELS if name in models]
    order += sorted(models - set(order))

    # Cell key to condition. Each optimizer's skill also gets a cell under
    # its own name, for the optimizer comparison.
    conditions = {
        "baseline": BASELINE_CONDITION,
        "injected": INJECTED_CONDITION,
        "llmskill": LLMSKILL_CONDITION,
        "skillopt": skillopt_condition(MAIN_OPTIMIZER),
        "lora": LORA_CONDITION,
        "combined": COMBINED_CONDITION,
        "conditioned": conditioned_condition(CONDITIONED_OPTIMIZER),
        "noise": NOISE_CONDITION,
        "transfer": TRANSFER_CONDITION,
        "transfer_skillopt": TRANSFER_SKILLOPT_CONDITION,
        "transfer_augment": TRANSFER_AUGMENT_CONDITION,
        "transfer_ultrabreak": TRANSFER_ULTRABREAK_CONDITION,
    }
    conditions.update({optimizer: skillopt_condition(optimizer)
                       for optimizer in OPTIMIZERS})
    conditions.update({f"res{size}": resolution_condition(size)
                       for size in RESOLUTIONS})
    conditions.update({f"b1{beta1}":
                       beta1_condition(INJECTED_CONDITION, beta1)
                       for beta1 in BETA1S})
    conditions.update({key: key for key, _ in ENSEMBLE_ROWS
                       if key.startswith(ENSEMBLE_CONDITION)})

    table = []
    for model in order:
        # The cells showing a stand-in score, per benchmark.
        entry = {"model": model, "label": MODELS.get(model, model),
                 "cells": {}, "replaced": {}}
        for benchmark in BENCHMARKS:
            cell = {}
            for key, condition in conditions.items():
                lookup = (benchmark, model, reasoning, condition)
                row = selected.get(lookup)
                if lookup in failed:
                    row = stand_in_row(selected, lookup)
                    entry["replaced"].setdefault(benchmark, set()).add(key)
                cell[key] = row["score"] if row else None
            entry["cells"][benchmark] = cell
        # Drop a model never evaluated in this mode at all.
        if any(score is not None for cell in entry["cells"].values()
               for score in cell.values()):
            table.append(entry)
    return table


def resolution_seeds(
        ablation: Keyed,
        failed: dict[tuple[str, str, bool, str], tuple[str, str]],
) -> SeedScores:
    """Collect the score of every training seed in each resolution cell.

    Args:
        ablation: Rows from select_resolutions.
        failed: Failed runs from find_failed, keyed like ablation.
    Returns:
        Score and stand-in flag per seed, keyed by benchmark and cell key.
    """
    cells = {resolution_condition(size): f"res{size}" for size in RESOLUTIONS}
    seeds: SeedScores = {}
    for key, row in ablation.items():
        benchmark, _, _, condition = key
        size, seed = split_seed(condition)
        if size not in cells:
            continue
        # A failed seed counts with the score its own cell would show.
        stand_in = key in failed
        shown = stand_in_row(ablation, key) if stand_in else row
        if shown:
            seeds.setdefault((benchmark, cells[size]), {})[seed] = (
                shown["score"], stand_in)
    return seeds


def add_spread(table: list[dict[str, Any]], seeds: SeedScores) -> None:
    """Show the mean over seeds in each resolution cell, and their spread.

    Args:
        table: Entries of the resolution table, changed in place.
        seeds: Per-seed scores from resolution_seeds.
    """
    for entry in table:
        spread = entry.setdefault("spread", {})
        for (benchmark, cell), by_seed in seeds.items():
            scores = [score for score, _ in by_seed.values()]
            entry["cells"][benchmark][cell] = mean(scores)
            if len(scores) > 1:
                spread.setdefault(benchmark, {})[cell] = stdev(scores)
            if any(stand_in for _, stand_in in by_seed.values()):
                entry["replaced"].setdefault(benchmark, set()).add(cell)


def format_score(score: float | None) -> str:
    """Render a score fraction as a percentage.

    Args:
        score: Score fraction, or None when the run is missing.
    Returns:
        Percentage string, or a dash when the run is missing.
    """
    return "--" if score is None else f"{score * 100:.2f}"


def format_delta(baseline: float | None, injected: float | None) -> str:
    """Render the injected-minus-baseline delta in points.

    Args:
        baseline: Baseline score fraction, or None.
        injected: Injected score fraction, or None.
    Returns:
        Signed delta string, or an empty string when either side is missing.
    """
    if baseline is None or injected is None:
        return ""
    return f"{(injected - baseline) * 100:+.2f}"


def format_delta_cell(delta: str) -> str:
    """Wrap a signed delta for display beside its score.

    Args:
        delta: Signed delta string, empty when there is nothing to show.
    Returns:
        The parenthesised delta, or an empty string.
    """
    return f" ({delta})" if delta else ""


def colorize(text: str, delta: str, latex: bool) -> str:
    """Colour a delta green when it improves the score and red when it hurts.

    Args:
        text: Rendered delta text to colour.
        delta: Signed delta string used to pick the colour.
        latex: Whether to emit LaTeX colour commands instead of ANSI codes.
    Returns:
        The coloured text, unchanged when the delta is empty or zero.
    """
    # A tie, e.g. +0.00, is neither an improvement nor a regression.
    if not delta or float(delta) == 0:
        return text
    improved = float(delta) > 0
    if latex:
        name = "ForestGreen" if improved else "BrickRed"
        return f"\\textcolor{{{name}}}{{{text}}}"
    return f"{GREEN if improved else RED}{text}{RESET}"


def render_terminal(table: list[dict], rows: tuple,
                    reference: list[dict] | None = None,
                    benchmarks: list[str] | None = None) -> str:
    """Render the table as an aligned, colour-coded terminal table.

    Args:
        table: Entries from build_table, with any spread from add_spread.
        rows: Cell key and display name of each row in a model's block.
        reference: Entries whose same cell each delta is taken against.
        benchmarks: Benchmark columns to show, all of BENCHMARKS by default.
    Returns:
        The rendered table.
    """
    benchmarks = list(benchmarks or BENCHMARKS)
    against = {entry["model"]: entry for entry in reference or ()}
    # Lay out on the uncoloured text, so ANSI codes never skew the padding.
    plain = []
    for entry in table:
        other = against.get(entry["model"])
        for condition, name in rows:
            cells = []
            for benchmark in benchmarks:
                cell = entry["cells"][benchmark]
                delta = ""
                if reference is not None:
                    if other:
                        delta = format_delta(
                            other["cells"][benchmark][condition],
                            cell[condition])
                elif condition != "baseline":
                    key = DELTA_REFERENCES.get(condition, "baseline")
                    delta = format_delta(cell[key], cell[condition])
                mark = ""
                if condition in entry["replaced"].get(benchmark, ()):
                    mark = STAND_IN_MARK
                spread = entry.get("spread", {}).get(benchmark, {})
                std = (f" ± {spread[condition] * 100:.2f}"
                       if condition in spread else "")
                cells.append((format_score(cell[condition]), std, delta,
                              mark))
            plain.append((entry["label"] if condition == rows[0][0] else "",
                          name, cells))

    label_w = max([len(row[0]) for row in plain] + [5])
    cond_w = max(len(row[1]) for row in plain)
    # Score, spread and delta are padded separately, so a baseline score sits
    # directly above the scores of the rows below it instead of above their
    # spreads or deltas.
    score_w = [max(len(row[2][i][0]) for row in plain)
               for i in range(len(benchmarks))]
    std_w = [max(len(row[2][i][1]) for row in plain)
             for i in range(len(benchmarks))]
    delta_w = [max(len(format_delta_cell(row[2][i][2]) + row[2][i][3])
                   for row in plain)
               for i in range(len(benchmarks))]
    col_w = [max(score_w[i] + std_w[i] + delta_w[i], len(BENCHMARKS[key]))
             for i, key in enumerate(benchmarks)]

    lines = []
    header = " " * label_w + "  " + " " * cond_w
    for i, benchmark in enumerate(benchmarks):
        # Left-aligned, so the heading starts over the scores below it.
        header += "  " + BENCHMARKS[benchmark].ljust(col_w[i])
    lines.append(header.rstrip())
    lines.append("-" * (label_w + 2 + cond_w + sum(2 + w for w in col_w)))

    for index, (label, name, cells) in enumerate(plain):
        # A blank label marks a later row of a model, so separate blocks.
        if label and index:
            lines.append("")
        line = label.ljust(label_w) + "  " + name.ljust(cond_w)
        for i, (score, std, delta, mark) in enumerate(cells):
            text = format_delta_cell(delta)
            shown = colorize(text, delta, latex=False) if delta else text
            pad = col_w[i] - score_w[i] - std_w[i] - len(text) - len(mark)
            line += ("  " + score.rjust(score_w[i]) + std.ljust(std_w[i])
                     + shown + mark + " " * pad)
        lines.append(line.rstrip())
    return "\n".join(lines)


def render_by_model(table: list[dict], rows: tuple, benchmark: str) -> str:
    """Render one benchmark as a terminal table with a column per model.

    Args:
        table: Entries from build_table.
        rows: Cell key and display name of each row.
        benchmark: Benchmark whose cells are shown.
    Returns:
        The rendered table, without the rows that have no score at all.
    """
    entries = {entry["model"]: entry for entry in table}
    models = [model for model in ENSEMBLE_MODELS if model in entries]
    cells = {model: entries[model]["cells"][benchmark] for model in models}
    # Lay out on the uncoloured text, so ANSI codes never skew the padding.
    plain = []
    for key, name in rows:
        if key != "baseline" and all(cells[m][key] is None for m in models):
            continue
        shown = []
        for model in models:
            delta = ("" if key == "baseline" else
                     format_delta(cells[model]["baseline"], cells[model][key]))
            shown.append((format_score(cells[model][key]), delta))
        plain.append((name, shown))

    heads = [(MODELS.get(model, model),
              "in ensemble" if model in ENSEMBLE_MEMBERS else "held out")
             for model in models]
    name_w = max(len(name) for name, _ in plain)
    score_w = [max(len(row[1][i][0]) for row in plain)
               for i in range(len(models))]
    delta_w = [max(len(format_delta_cell(row[1][i][1])) for row in plain)
               for i in range(len(models))]
    col_w = [max(score_w[i] + delta_w[i], *map(len, heads[i]))
             for i in range(len(models))]

    lines = []
    for part in range(2):
        line = " " * name_w
        for i, head in enumerate(heads):
            line += "  " + head[part].ljust(col_w[i])
        lines.append(line.rstrip())
    lines.append("-" * (name_w + sum(2 + w for w in col_w)))
    for name, shown in plain:
        line = name.ljust(name_w)
        for i, (score, delta) in enumerate(shown):
            text = format_delta_cell(delta)
            colored = colorize(text, delta, latex=False) if delta else text
            pad = col_w[i] - score_w[i] - len(text)
            line += "  " + score.rjust(score_w[i]) + colored + " " * pad
        lines.append(line.rstrip())
    return "\n".join(lines)


def render_superseded(superseded: list[dict], selected: dict) -> str:
    """Describe the rows that a newer run replaced.

    Args:
        superseded: Rows that lost to a newer run.
        selected: Rows kept, keyed by benchmark, model, reasoning, condition.
    Returns:
        A report of the replaced rows, empty when nothing was replaced.
    """
    if not superseded:
        return ""
    lines = [f"Superseded by a later run ({len(superseded)}):"]
    for row in sorted(superseded, key=lambda r: r["timestamp"]):
        key = (row["benchmark"], row["model"], row["reasoning"],
               row["condition"])
        kept = selected[key]
        lines.append(f"  {row['run_name']}")
        lines.append(f"    {row['score'] * 100:.2f} at {row['timestamp'][:19]}"
                     f" -> kept {kept['score'] * 100:.2f} from"
                     f" {kept['run_name']}")
    return "\n".join(lines)


def render_replaced(failed: dict, selected: dict) -> str:
    """Describe the failed runs whose cells show a stand-in score.

    Args:
        failed: Failed runs from find_failed, keyed like selected.
        selected: Rows kept, keyed by benchmark, model, reasoning, condition.
    Returns:
        A report of the replaced cells, empty when nothing was replaced.
    """
    if not failed:
        return ""
    lines = [f"{STAND_IN_MARK} Optimization found nothing better than no"
             f" skill, score without it shown ({len(failed)}):"]
    for key, (name, reason) in sorted(failed.items()):
        benchmark, model, reasoning, _ = key
        stand_in = stand_in_row(selected, key)
        score = format_score(stand_in["score"] if stand_in else None)
        mode = "reasoning" if reasoning else "non-reasoning"
        lines.append(f"  {BENCHMARKS[benchmark]}, {MODELS.get(model, model)},"
                     f" {mode}: {name} {format_score(selected[key]['score'])}"
                     f" -> {score}")
        lines.append(f"    {reason}")
    return "\n".join(lines)


def render_hidden(hidden: list[dict]) -> str:
    """Point at the runs left out for their distractor count.

    Args:
        hidden: Rows whose distractor count was not requested.
    Returns:
        One line per left-out count, empty when nothing was left out.
    """
    counts = Counter((row["benchmark"], row["n_distractors"])
                     for row in hidden)
    return "\n".join(
        f"Hidden {n} {BENCHMARKS[benchmark]} runs with {count} distractors;"
        f" pass --distractors {count} to report them."
        for (benchmark, count), n in sorted(counts.items()))


def format_size(size: Size) -> str:
    """Render an image size as height x width.

    Args:
        size: Height and width.
    Returns:
        The size, e.g. 256x256.
    """
    return "x".join(str(side) for side in size)


def render_off_size(off_size: list[dict[str, Any]], sizes: Sizes) -> str:
    """Point at the image runs left out for their image size.

    Args:
        off_size: Rows whose image size was not requested.
        sizes: Trained image sizes from load_image_sizes.
    Returns:
        One line per left-out size, empty when nothing was left out.
    """
    counts = Counter((row["benchmark"], row["model"], *image_sizes(row, sizes))
                     for row in off_size)
    lines = []
    for (benchmark, model, trained, fed), n in sorted(counts.items()):
        where = f"{BENCHMARKS[benchmark]} {MODELS.get(model, model)}"
        expected = tuple(fed_side(side) for side in trained)
        if fed != expected:
            lines.append(f"Hidden {n} {where} Skill.png runs of a"
                         f" {format_size(trained)} image evaluated at"
                         f" {format_size(fed)}; set model.resolution to"
                         f" {format_size(expected)}.")
        else:
            lines.append(f"Hidden {n} {where} Skill.png runs at"
                         f" {format_size(trained)}; pass --resolution"
                         f" {trained[0]} to report them.")
    return "\n".join(lines)


def render_seeds(seeds: SeedScores) -> str:
    """List the score of each seed behind the cells that average several.

    Args:
        seeds: Per-seed scores from resolution_seeds.
    Returns:
        One line per such cell, empty when every cell has one seed.
    """
    lines = []
    for benchmark in BENCHMARKS:
        for size, label in RESOLUTION_LABELS.items():
            by_seed = seeds.get((benchmark, f"res{size}"), {})
            if len(by_seed) < 2:
                continue
            # The reported image first, then the replicates.
            order = sorted(by_seed, key=lambda seed: (seed != TRAIN_SEED,
                                                      seed))
            scores = ", ".join(
                f"{format_score(by_seed[seed][0])}"
                f"{STAND_IN_MARK if by_seed[seed][1] else ''} (seed {seed})"
                for seed in order)
            lines.append(f"  {BENCHMARKS[benchmark]}, {label}: {scores}")
    if not lines:
        return ""
    return "\n".join(["Training seeds behind each mean ± sample std:",
                      *lines])


def main() -> None:
    """Print the non-reasoning and reasoning tables for the recorded runs.

    Args:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="directory holding the benchmark run trees")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="only report runs with this seed")
    parser.add_argument("--distractors", type=int, default=DISTRACTORS,
                        help="only report TRAJECT-Bench runs with this many"
                             " distractor tools")
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="only report Qwen Skill.png runs whose image"
                             " is this many pixels square")
    args = parser.parse_args()

    loaded = load_rows(args.runs_dir, args.seed)
    # Other distractor counts stay on disk for ablations. Mixed in, the newest
    # run would win whatever pool it used.
    wanted = (None, args.distractors)
    hidden = [row for row in loaded if row.get("n_distractors") not in wanted
              and not is_ensemble(row) and not is_transfer(row)]
    loaded = [row for row in loaded if row.get("n_distractors") in wanted]
    # Ensemble images only ever appear in their own tables.
    ensembled = [row for row in loaded if is_ensemble(row)]
    loaded = [row for row in loaded if not is_ensemble(row)]
    rows = [row for row in loaded if not is_transfer(row)]
    # Other image sizes stay on disk for the resolution ablation, and would
    # otherwise win as the newest run.
    sizes = load_image_sizes(args.runs_dir)
    ablation, ablated = select_resolutions(rows, sizes)
    # Replicates of other training seeds only feed the resolution table's
    # spread, so they are neither reported nor pointed at below.
    rows = [row for row in rows if row["train_seed"] == TRAIN_SEED]
    off_size = [row for row in rows
                if not at_resolution(row, sizes, args.resolution)]
    rows = [row for row in rows
            if at_resolution(row, sizes, args.resolution)]
    # Other values of the beta1 ablation only enter its own table.
    varied = rows
    rows = [row for row in rows
            if split_beta1(row["condition"])[1] == REPORTED_BETA1]
    if not rows:
        raise SystemExit(f"no results with seed={args.seed} under "
                         f"{args.runs_dir}")
    selected, superseded = select_latest(rows)
    images = load_failed_images(args.runs_dir)
    skills = load_failed_skills(args.runs_dir)
    loras = load_failed_loras(args.runs_dir)
    failed = find_failed(selected, images, skills, loras)

    tables = {reasoning: build_table(selected, reasoning, failed)
              for reasoning in (False, True)}
    for rows, suffix in ((ROWS, ""),
                         (COMBINED_ROWS, ", Skill.png + LLM-Skill")):
        for reasoning, title in ((False, "Non-reasoning"),
                                 (True, "Reasoning")):
            table = tables[reasoning]
            # The main reasoning table shows what reasoning changes.
            reference = tables[False] if reasoning and rows is ROWS else None
            if reference is not None:
                title += ", delta against non-reasoning"
            print(title + suffix)
            print(render_terminal(table, rows, reference) if table
                  else "(no runs)")
            print()
    print("Non-reasoning, Random Image and Skill.png")
    print(render_terminal(tables[False], NOISE_ROWS) if tables[False]
          else "(no runs)")
    print()
    table = [entry for entry in tables[False]
             if entry["model"] == OPTIMIZER_MODEL]
    print("Non-reasoning, SkillOpt optimizers and Skill.png on "
          f"{MODELS.get(OPTIMIZER_MODEL, OPTIMIZER_MODEL)}")
    print(render_terminal(table, OPTIMIZER_ROWS) if table else "(no runs)")
    print()
    transfer, transferred = select_transfer(loaded, selected)
    ensemble, benchmarks = select_ensemble(ensembled, selected, transfer)
    # The transfer table shows the ensemble image next to the transferred one.
    picked = {**transfer,
              **{key: row for key, row in ensemble.items()
                 if key[3] in (BASELINE_CONDITION,
                               TRANSFER_ENSEMBLE_CONDITION)}}
    columns = [benchmark for benchmark in BENCHMARKS
               if benchmark in transferred or benchmark in benchmarks]
    table = build_table(picked, False, {}) if columns else []
    members = " and ".join(MODELS[model] for model in ENSEMBLE_MEMBERS)
    print(f"Non-reasoning, SkillOpt, UltraBreak and Skill.png of "
          f"{MODELS[TRANSFER_SOURCE]} and the Ensemble of {members} on every"
          " model")
    print(render_terminal(table, TRANSFER_ROWS, benchmarks=columns)
          if table else "(no runs)")
    print()
    table = build_table(ensemble, False, {}) if benchmarks else []
    for benchmark in benchmarks or [None]:
        name = BENCHMARKS[benchmark] if benchmark else "every benchmark"
        print(f"Non-reasoning, {name}: Skill.png optimized on {members} at"
              " once, on every model")
        print(render_by_model(table, ENSEMBLE_ROWS, benchmark) if benchmark
              else "(no runs)")
        # No stand-ins here, as in the transfer table; say so when the
        # weighted val score of an ensemble image never beat the clean one.
        notes = {}
        for (cell_benchmark, _, _, condition), row in ensemble.items():
            reason = images.get((cell_benchmark, ENSEMBLE_MEMBERS[0],
                                 row.get("image_digest", NO_IMAGE)))
            if (cell_benchmark == benchmark and reason
                    and condition.startswith(ENSEMBLE_CONDITION)):
                notes[condition] = reason
        labels = dict(ENSEMBLE_ROWS)
        for condition, reason in sorted(notes.items()):
            print(f"  {labels[condition]}: {reason}")
        print()
    archived = load_rows(args.runs_dir, args.seed, "*/*/old/results.jsonl")
    for benchmark, source, optimizer in BEST_TRANSFERS:
        best = select_best_transfer(loaded, archived, selected, benchmark,
                                    source, optimizer)
        table = build_table(best, False, {}) if best else []
        print(f"Non-reasoning, {OPTIMIZERS[optimizer]} SkillOpt and"
              f" Skill.png of {MODELS[source]} on {BENCHMARKS[benchmark]},"
              " on every model")
        print(render_terminal(table, BEST_TRANSFER_ROWS,
                              benchmarks=[benchmark])
              if table else "(no runs)")
        print()
    ablation_failed = find_failed(ablation, images, skills)
    table = build_table(ablation, False, ablation_failed) if ablated else []
    seeds = resolution_seeds(ablation, ablation_failed)
    add_spread(table, seeds)
    print("Non-reasoning, Skill.png image sizes on"
          f" {MODELS[RESOLUTION_MODEL]}, mean ± sample std over the training"
          " seeds where there are several")
    print(render_terminal(table, RESOLUTION_ROWS, benchmarks=ablated)
          if table else "(no runs)")
    report = render_seeds(seeds)
    if report:
        print(report)
    print()
    beta1 = select_beta1(varied, selected)
    beta1_failed = find_failed(beta1, images, skills)
    table = build_table(beta1, False, beta1_failed) if beta1 else []
    print(f"Non-reasoning, Skill.png Adam beta1 on {MODELS[BETA1_MODEL]}")
    print(render_terminal(table, BETA1_ROWS, benchmarks=[BETA1_BENCHMARK])
          if table else "(no runs)")
    print()
    report = render_replaced({**failed, **ablation_failed, **beta1_failed},
                             {**selected, **ablation, **beta1})
    if report:
        print(report)
    report = render_superseded(superseded, selected)
    if report:
        print(report)
    report = render_hidden(hidden)
    if report:
        print(report)
    report = render_off_size(off_size, sizes)
    if report:
        print(report)


if __name__ == "__main__":
    main()
