"""Evaluation: an image, an LLM-Skill or both against any (benchmark, model, split), in one call."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .benchmarks.base import Benchmark, Item, Result
from .benchmarks.registry import build_benchmark
from .config import BenchmarkCfg, Config
from .image.param import ImageParam, load_param, noise_param, read_sidecar
from .models.base import VLM
from .models.registry import build_model
from .results import PredictionLog, PredictionRow, ResultRow, append_row, git_sha

log = logging.getLogger(__name__)

# A skill in the system turn and no image: the LLM-Skill and the SkillOpt
# skill, told apart only by the condition on their rows.
TEXT_SKILL_CONDITIONS = ("llmskill", "skillopt")
# An image optimized with the skill already in the system turn
# (train.skill_path), scored with that same skill. skillpng_llmskill pairs
# an image and a skill that were obtained apart.
CONDITIONED_CONDITION = "skillpng_conditioned"
# The optimized image in the user turn, read from eval.image_path.
IMAGE_CONDITIONS = ("skillpng", "skillpng_llmskill", CONDITIONED_CONDITION)
# The skill in the system turn, read from eval.skill_path.
SKILL_CONDITIONS = (*TEXT_SKILL_CONDITIONS, "skillpng_llmskill", CONDITIONED_CONDITION)
# The request of none, answered with a LoRA adapter applied server-side.
ADAPTER_CONDITION = "lora"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch so a run is reproducible.

    Args:
        seed: Value handed to every random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_split(
    model: VLM,
    benchmark: Benchmark,
    image: Tensor | None,
    items: list[Item],
    site: str = "pre_query",
    predictions: PredictionLog | None = None,
) -> tuple[float, list[Result]]:
    """Roll out every item under one condition and average the scores.

    Real evaluation: generation plus the benchmark's own scoring, never a
    proxy for it. One condition per pass, so a no-image baseline is measured
    once and reported against every image that is scored on the same split.

    Args:
        model: Vision language model that generates the answers.
        benchmark: Benchmark supplying the rollout and scoring logic.
        image: Image shown to the model, or None for the clean condition.
        items: Items of the split to score, must be non-empty.
        site: Injection site passed through to the rollout.
        predictions: Log every item's row is written to, or None to keep none.

    Returns:
        The mean score and the per-item results.
    """
    if not items:
        raise ValueError("no items to evaluate")

    async def run() -> list[Result]:
        """Roll out every item of the split in order, logging as it goes.

        Args:
            None.

        Returns:
            One result per item, in item order.
        """
        results: list[Result] = []

        for item in items:
            # Sequential: the model is in-process and GPU-bound, so
            # concurrency buys nothing here. rollout is async so a
            # tool-using benchmark can await its own I/O.
            result = await benchmark.rollout(item, model, image, site=site)
            results.append(result)

            if predictions is not None:
                predictions.write(
                    PredictionRow(
                        item_id=item.id,
                        # Benchmarks that carry a question expose it here; the
                        # id alone is a hash, which is unreadable when the
                        # point is debugging.
                        question=str(item.meta.get("question", "")),
                        target=item.target,
                        output=result.output,
                        score=result.score,
                        meta=result.meta,
                    )
                )

        return results

    results = asyncio.run(run())

    return _mean(results), results


def condition_image(cfg: Config, device: str | torch.device = "cpu") -> ImageParam | None:
    """Turn eval.condition into an image parameter.

    Conditions are config values, not ad-hoc scripts, so each name maps to
    one construction path and anything else is an error.

    Args:
        cfg: Config whose eval section names the condition and its inputs.
        device: Device the constructed image parameter is placed on.

    Returns:
        The image parameter for the condition, or None for the clean one.
    """
    condition = cfg.eval.condition
    if condition in ("none", ADAPTER_CONDITION):
        return None
    
    if condition in TEXT_SKILL_CONDITIONS:
        # Text only: the skill rides in the system turn, not in an image.
        return None

    if condition == "noise":
        return noise_param(cfg.param, seed=cfg.seed, device=device)
    
    if condition in IMAGE_CONDITIONS:
        if not cfg.eval.image_path:
            raise ValueError(f"eval.condition={condition} requires eval.image_path")
        
        return load_param(cfg.eval.image_path, device=device)
    
    raise ValueError(f"unknown condition {condition!r}")


def condition_skill(cfg: Config) -> str | None:
    """Read the skill document that the conditions in SKILL_CONDITIONS ask for.

    Args:
        cfg: Config whose eval section names the condition and the skill.

    Returns:
        The skill text, stripped, or None for every other condition.
    """
    condition = cfg.eval.condition

    if condition not in SKILL_CONDITIONS:
        return None

    if not cfg.eval.skill_path:
        raise ValueError(f"eval.condition={condition} requires eval.skill_path")

    return read_skill(cfg.eval.skill_path, "eval.skill_path")


def read_skill(path: str, key: str) -> str:
    """Read a skill document the same way for training and evaluation.

    Args:
        path: Path to the skill document.
        key: Config key the path came from, named in the errors.

    Returns:
        The skill text, stripped.
    """
    file = Path(path)

    if not file.is_file():
        raise FileNotFoundError(f"{key} does not exist: {file}")

    skill = file.read_text(encoding="utf-8").strip()

    if not skill:
        raise ValueError(f"{key} is empty: {file}")

    return skill


def check_trained_under(image_path: str | None, skill: str) -> None:
    """Refuse an image that was not optimized under this exact skill.

    The sidecar written at training time records the skill's digest, so a
    pairing with any other skill, or with an image trained without one, is
    caught before the model is built.

    Args:
        image_path: The optimized image, with its sidecar beside it.
        skill: The skill text the image is about to be scored with.
    """
    if not image_path:
        raise ValueError(f"eval.condition={CONDITIONED_CONDITION} requires eval.image_path")

    sidecar = read_sidecar(Path(image_path).with_suffix(".png"))
    trained = sidecar.get("skill_digest")
    digest = hashlib.sha256(skill.encode()).hexdigest()

    if trained != digest:
        shown = str(trained)[:16] if trained else "none"
        raise ValueError(
            f"eval.condition={CONDITIONED_CONDITION}: {image_path} was optimized "
            f"under skill {shown} ({sidecar.get('skill_path')}), not under "
            f"eval.skill_path ({digest[:16]}). Score it with the skill its "
            "sidecar names, or use skillpng_llmskill for an image and a skill "
            "obtained apart."
        )


_SKILL_SLOTS = re.compile(r"\{system_prompt\}|\{skill\}")


def compose_system_prompt(system_prompt: str, template: str, skill: str) -> str:
    """Fill injection.skill_template with the benchmark prompt and an LLM-Skill.

    One pass over the template, so braces inside the prompt or the skill are
    never read as placeholders.

    Args:
        system_prompt: The benchmark's system prompt, unedited.
        template: Template holding each placeholder exactly once.
        skill: The skill text.

    Returns:
        The system prompt the benchmark is built with.
    """
    for slot in ("{system_prompt}", "{skill}"):
        count = template.count(slot)

        if count != 1:
            raise ValueError(
                f"injection.skill_template must contain {slot} exactly once, found {count}"
            )

    values = {"{system_prompt}": system_prompt, "{skill}": skill}

    return _SKILL_SLOTS.sub(lambda match: values[match.group(0)], template)


def with_skill(benchmark: BenchmarkCfg, template: str, skill: str) -> BenchmarkCfg:
    """Copy a benchmark config with the skill composed into its system prompt.

    Training and evaluation both go through here, so a skill shown during
    optimization sits exactly where it sits when the image is scored.

    Args:
        benchmark: Benchmark config carrying the unedited system prompt.
        template: Template holding each placeholder exactly once.
        skill: The skill text.

    Returns:
        The benchmark config whose system prompt carries the skill.
    """
    prompt = compose_system_prompt(benchmark.system_prompt, template, skill)

    return benchmark.model_copy(update={"system_prompt": prompt})


def run_eval(
    cfg: Config,
    model: VLM | None = None,
    benchmark: Benchmark | None = None,
    param: ImageParam | None = None,
    split: str | None = None,
    write: bool = True,
) -> ResultRow:
    """Score one condition on one split and append one result row.

    One condition, one pass: the no-image baseline is its own run
    (eval.condition=none) and is reported against every image scored on the
    same benchmark, model and split. Under llmskill, skillopt,
    skillpng_llmskill and skillpng_conditioned the benchmark is built with
    the skill in its system prompt, which every adapter sends verbatim. The
    last two also show the optimized image in the user turn, exactly as
    skillpng does; skillpng_conditioned takes only an image optimized under
    that same skill.

    Args:
        cfg: Config driving seeding, model, benchmark, condition and output.
        model: Prebuilt model, or None to build one from the config.
        benchmark: Prebuilt benchmark, or None to build one from the config.
        param: Prebuilt image parameter, or None to derive it from the condition.
        split: Split name to score, or None to take it from the config.
        write: Whether the row is appended to the results file.

    Returns:
        The row holding the score and the run's provenance fields.
    """
    set_seed(cfg.seed)
    # Before the model is built, so a bad skill fails without the wait on
    # a server.
    skill = condition_skill(cfg)
    benchmark_cfg = cfg.benchmark

    if skill is not None:
        benchmark_cfg = with_skill(cfg.benchmark, cfg.injection.skill_template, skill)

        if cfg.eval.condition == CONDITIONED_CONDITION:
            check_trained_under(cfg.eval.image_path, skill)

        if benchmark is not None and benchmark.cfg.system_prompt != benchmark_cfg.system_prompt:
            raise ValueError(
                f"eval.condition={cfg.eval.condition} got a prebuilt benchmark" 
                "whose system prompt does not carry the skill; build it from" "the composed prompt or pass none"
            )

        # skillpng_llmskill takes the image as skillpng does.
        if cfg.eval.condition in TEXT_SKILL_CONDITIONS and param is not None:
            raise ValueError(f"eval.condition={cfg.eval.condition} is text only; pass no image")

    model = model or build_model(cfg.model)
    benchmark = benchmark or build_benchmark(benchmark_cfg)
    split = split or cfg.eval.split

    items = benchmark.items(split)
    if cfg.eval.limit:
        items = items[: cfg.eval.limit]
    if param is None:
        param = condition_image(cfg, device=_device_of(model))

    image = param.tensor().detach() if param is not None else None
    path = _predictions_path(cfg, split)

    if write:
        log.info("per-item outputs stream to %s", path)

    with PredictionLog(path) if write else nullcontext(None) as predictions:
        score, _ = evaluate_split(
            model, benchmark, image, items, cfg.injection.site, predictions
        )

    described = model.describe()
    # A row without an image carries none of the image fields.
    row = ResultRow(
        image_digest=param.digest() if param is not None else None,
        skill_digest=hashlib.sha256(skill.encode()).hexdigest() if skill is not None else "none",
        param_kind=param.kind if param is not None else None,
        resolution=list(_resolution(model, param)) if param is not None else None,
        image_token_count=model.image_token_count if param is not None else None,
        benchmark=cfg.benchmark.name,
        n_distractors=benchmark.n_distractors,
        split=split,
        model=str(described.get("model", "unknown")),
        model_dtype=str(described.get("model_dtype", "unknown")),
        injection_site=cfg.injection.site if param is not None else None,
        condition=cfg.eval.condition,
        score=score,
        metric_name=benchmark.metric_name,
        n_items=len(items),
        seed=cfg.seed,
        git_sha=git_sha(),
        config_hash=cfg.hash(),
        run_name=cfg.train.run_name,
    )

    if write:
        append_row(cfg.results_path, row)

    log.info(
        "%s/%s [%s] %s=%.4f n=%d",
        cfg.benchmark.name,
        split,
        cfg.eval.condition,
        benchmark.metric_name,
        score,
        len(items),
    )

    return row


def _predictions_path(cfg: Config, split: str) -> Path:
    """Where the per-item outputs of this eval land.

    One file per (split, condition) inside the run directory, so scoring the
    same split under two conditions leaves both to compare.

    Args:
        cfg: Config naming the run directory, run name and condition.
        split: Split that was scored.

    Returns:
        Path to the per-item JSONL file for this eval.
    """
    return cfg.run_path / "eval" / f"{split}-{cfg.eval.condition}.jsonl"


def _mean(results: list[Result]) -> float:
    """Average the scores of one condition.

    Args:
        results: Results to average, must be non-empty.

    Returns:
        The mean score.
    """
    return float(sum(result.score for result in results) / len(results))


def _resolution(model: VLM, param: ImageParam) -> tuple[int, int]:
    """Prefer the resolution actually fed to the tower over the canvas size.

    Args:
        model: Model that may expose the size its vision tower consumes.
        param: Image parameter to fall back to.

    Returns:
        The height and width.
    """
    size: object = getattr(model, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])

    _, height, width = param.shape
    return height, width


def _device_of(model: VLM) -> str | torch.device:
    """Read the model's device, falling back to CPU when it has none.

    Args:
        model: Model that may carry a device attribute.

    Returns:
        The model's device, or cpu when it is missing or an odd type.
    """
    device: object = getattr(model, "device", "cpu")
    return device if isinstance(device, (str, torch.device)) else "cpu"
