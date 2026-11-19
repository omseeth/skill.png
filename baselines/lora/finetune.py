"""LoRA: one adapter fitted for one benchmark and one target model.

The adapter is fitted on the train split under the objective the Skill.png
image is optimized with, teacher-forced CE on the gold answer, plus the marker
that closes the assistant turn, so the model also learns to stop. It is
selected on val like best.png, and step 0, the base model itself, takes part
in that selection. Only the language model's attention and MLP projections
are adapted, the modules vLLM serves LoRA on.

    uv run --extra lora python baselines/lora/finetune.py --benchmark searchqa \\
        --set model.model_id=/path/to/qwen3_5-4B \\
        --set train.run_dir=runs/searchqa/qwen3.5-4B

The step history and summary.json go to <train.run_dir>/<train.run_name>/,
and the history to wandb as well under wandb.enabled. The selected adapter
goes to <adapter root>/<benchmark>/<name>/ beside its lora.meta.json, the
root defaulting to <model_id>-lora next to the checkpoint,
and an existing adapter there is never overwritten. It is scored under
eval.condition=lora by a vLLM server that loads it.

A family whose forward requires pixel_values (Nemotron 3 Nano Omni) gets its
adapter fitted on the language model alone: vLLM serves LoRA there only on
the text-only checkpoint that text_only.py writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import wandb
from torch import Tensor, nn

from skillpng.benchmarks.base import Benchmark, Item
from skillpng.benchmarks.registry import build_benchmark
from skillpng.config import Config, ModelCfg, load_config
from skillpng.evaluate import evaluate_split, set_seed
from skillpng.models.hf import HFVLM, _forward_takes
from skillpng.results import git_sha
from skillpng.train import open_wandb, prune_wandb_logs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# llmskill's generator holds the config lookup this script reuses.
sys.path.append(str(ROOT / "baselines" / "llmskill"))

from generate import resolve_config  # noqa: E402

log = logging.getLogger(__name__)

# The decoder's attention and MLP projections, which vLLM serves LoRA on in
# every family run here. vLLM packs Qwen3.5's linear attention projections
# differently from the checkpoint, so those are left out.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Routed MoE experts are left out: an adapter per expert would hold most of
# the parameters while each sees a few tokens. Shared experts stay in.
NOT_ROUTED = r"(?!.*\.experts\.\d)"
# The submodule that holds the decoder in every family run here.
LANGUAGE_MODEL = "language_model"
# The ranks vLLM's --max-lora-rank accepts, up to 128.
RANKS = (8, 16, 32, 64, 128)
# Skill.png images are optimized with thinking off; so is every adapter here.
THINKING_OFF = "model.chat_template_kwargs.enable_thinking=false"
# Stands in for the assistant's content when the chat template is rendered
# to read its end-of-turn marker.
SENTINEL = "SKILLPNG-LORA-SENTINEL"
META_NAME = "lora.meta.json"
WEIGHTS_NAME = "adapter_model.safetensors"


@dataclass
class Options:
    """Everything about a run that is not the target config."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 8
    warmup: float = 0.05
    grad_clip: float | None = 1.0
    eval_every: int | None = None
    select_on: str = "val_metric"
    targets: tuple[str, ...] = DEFAULT_TARGETS
    seed: int = 42
    limit: int | None = None
    dry_run: bool = False

    def name(self) -> str:
        """The adapter's directory name.

        Returns:
            E.g. r16_alpha32_lr0.0002_ep10_bs8_nothink_seed42, plus _valloss
            under val_loss selection and _targets<sha8> for other targets.
        """
        name = f"r{self.rank}_alpha{self.alpha}_lr{self.lr:g}_ep{self.epochs}_bs{self.batch_size}"

        if self.select_on == "val_loss":
            name += "_valloss"

        if self.targets != DEFAULT_TARGETS:
            name += f"_targets{_sha256(','.join(self.targets).encode())[:8]}"

        return f"{name}_nothink_seed{self.seed}"

    def run_name(self) -> str:
        """The default train.run_name.

        Returns:
            lora_<name>, plus _limit<N> for a capped run.
        """
        suffix = f"_limit{self.limit}" if self.limit else ""

        return f"lora_{self.name()}{suffix}"


class LoraVLM(HFVLM):
    """HFVLM with a text-only loss for the families whose forward needs pixels.

    Such a forward splices the image into the embeddings and hands them to
    its language model. Without an image there is nothing to splice, so the
    language model alone is the whole computation, and it carries the adapter.
    """

    def __init__(self, cfg: ModelCfg) -> None:
        """Load the checkpoint as HFVLM does and find the adapter's host.

        Args:
            cfg: Model configuration naming the checkpoint, dtype, and routes.
        """
        super().__init__(cfg)

        if self.text_only and not isinstance(getattr(self.model, LANGUAGE_MODEL, None), nn.Module):
            raise SystemExit(
                f"{cfg.model_id}: the forward requires pixel_values and the model has no "
                f"{LANGUAGE_MODEL} to take a text-only loss through"
            )

    @property
    def text_only(self) -> bool:
        """Whether the adapter lives on the language model alone.

        Returns:
            True when this family's forward cannot run without pixels.
        """
        return self._needs_pixels

    @property
    def host(self) -> nn.Module:
        """The module the adapter is injected into and saved from.

        Returns:
            The language model for a text-only family, else the whole model.
        """
        host = getattr(self.model, LANGUAGE_MODEL) if self.text_only else self.model

        return cast(nn.Module, host)

    def _forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None,
        extra: dict[str, Any],
        keep: int = 0,
    ) -> Tensor:
        """Take a prompt without pixels through the language model alone.

        Args:
            input_ids: Prompt and target ids concatenated.
            attention_mask: Mask over input_ids.
            pixel_values: Pixels for HFVLM's own route, or None.
            extra: Processor keys we do not own.
            keep: Trailing positions to want logits for, 0 for the whole sequence.

        Returns:
            The logits, over the whole sequence or over its last keep positions.
        """
        if pixel_values is not None or not self.text_only:
            return super()._forward(input_ids, attention_mask, pixel_values, extra, keep)

        language_model = self.host
        takes_keep = _forward_takes(language_model, "logits_to_keep")
        trailing: dict[str, Any] = {"logits_to_keep": keep} if keep and takes_keep else {}

        with self._proven_sdpa():
            out = language_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **trailing
            )

        return cast(Tensor, out.logits)


def _sha256(data: bytes) -> str:
    """Hex sha256 of some bytes.

    Args:
        data: Bytes to hash.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    """The current UTC time.

    Returns:
        An ISO 8601 timestamp.
    """
    return datetime.now(UTC).isoformat()


def _display(path: Path) -> str:
    """A path relative to the repository when it lies inside it.

    Args:
        path: Path to show.

    Returns:
        The relative path, or the path as given.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        The parsed flags.
    """
    defaults = Options()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark", required=True, help="the benchmark.name to fit on")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override of the benchmark's config, as skillpng eval takes it",
    )
    parser.add_argument(
        "--adapter-root", type=Path, help="where adapters go, by default <model_id>-lora"
    )
    parser.add_argument("--rank", type=int, default=defaults.rank, choices=RANKS)
    parser.add_argument("--alpha", type=int, default=defaults.alpha)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument(
        "--batch-size", type=int, default=defaults.batch_size, help="items accumulated per step"
    )
    parser.add_argument(
        "--warmup", type=float, default=defaults.warmup, help="fraction of the steps warmed up"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=defaults.grad_clip, help="0 turns clipping off"
    )
    parser.add_argument(
        "--eval-every", type=int, help="steps between val checks, by default once per epoch"
    )
    parser.add_argument(
        "--select-on", default=defaults.select_on, choices=("val_metric", "val_loss")
    )
    parser.add_argument(
        "--target-modules",
        default=",".join(defaults.targets),
        help="comma-separated module names adapted inside the language model",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--limit", type=int, help="cap the train and val items, for smoke runs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the first train item's messages and target, load no weights",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be at least 1")

    if args.eval_every is not None and args.eval_every < 1:
        parser.error("--eval-every must be at least 1")

    if not 0.0 <= args.warmup < 1.0 or not 0.0 <= args.dropout < 1.0:
        parser.error("--warmup and --dropout must lie in [0, 1)")

    if not [name for name in args.target_modules.split(",") if name.strip()]:
        parser.error("--target-modules names no module")

    return args


def options_from(args: argparse.Namespace) -> Options:
    """Collect the run's options from the parsed flags.

    Args:
        args: The flags parse_args returned.

    Returns:
        The options.
    """
    targets = tuple(name.strip() for name in args.target_modules.split(",") if name.strip())

    return Options(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup=args.warmup,
        grad_clip=args.grad_clip or None,
        eval_every=args.eval_every,
        select_on=args.select_on,
        targets=targets,
        seed=args.seed,
        limit=args.limit,
        dry_run=args.dry_run,
    )


def resolve_target(path: Path, overrides: list[str]) -> Config:
    """The target config: the overrides, then thinking off.

    Args:
        path: The benchmark's experiment config.
        overrides: Dotted key=value overrides.

    Returns:
        The validated config with enable_thinking false.
    """
    cfg = load_config(path, overrides)
    thinking = cfg.model.chat_template_kwargs.get("enable_thinking", False)

    if thinking is not False:
        raise SystemExit(
            f"model.chat_template_kwargs.enable_thinking={thinking!r}: adapters are fitted "
            "with thinking off on every model, as the Skill.png images are optimized"
        )

    if cfg.model.name != "hf":
        raise SystemExit(
            f"model.name={cfg.model.name}: the adapter is fitted through the local weights"
        )

    if cfg.ensemble.members:
        raise SystemExit("ensemble.members: an adapter is fitted for one model")

    if cfg.benchmark.limit is not None:
        raise SystemExit("benchmark.limit caps the held-out check too; use --limit instead")

    return load_config(path, [*overrides, THINKING_OFF])


def split_items(
    benchmark: Benchmark, limit: int | None, eval_items: int | None
) -> tuple[list[Item], list[Item]]:
    """The train and val items, after proving no split shares an id.

    Test ids are read for that check only; no test item reaches the fit.

    Args:
        benchmark: Benchmark built from the uncapped config.
        limit: Prefix of train and val to keep, or None for all of them.
        eval_items: Prefix of val the adapter is selected on, as train.eval_items.

    Returns:
        The train items and the val items.
    """
    train = benchmark.items("train")
    val = benchmark.items("val")
    test = {item.id for item in benchmark.items("test")}
    train_ids = {item.id for item in train}
    val_ids = {item.id for item in val}
    leaked = sorted((train_ids & (val_ids | test)) | (val_ids & test))

    # Not an assert: the check has to survive python -O.
    if leaked:
        raise SystemExit(f"LEAK: id(s) {leaked[:5]} occur in two splits; nothing trained")

    if limit:
        train, val = train[:limit], val[:limit]

    if eval_items:
        val = val[:eval_items]

    if not train or not val:
        raise SystemExit(f"{len(train)} train and {len(val)} val items; the fit needs both")

    return train, val


def end_of_turn(processor: Any, chat_template_kwargs: dict[str, object]) -> str:
    """Read the marker that closes an assistant turn off the chat template.

    The tokenizer's eos is not always it: Gemma 4's is <eos>, its turns end
    in <turn|>, and vLLM stops at the latter.

    Args:
        processor: The checkpoint's processor, or a tokenizer.
        chat_template_kwargs: The template switches the run uses.

    Returns:
        The first token-like run of text the template puts after the content.
    """
    messages = [
        {"role": "user", "content": "?"},
        # A plain string: some templates print block content as a repr.
        {"role": "assistant", "content": SENTINEL},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, **chat_template_kwargs)
    _, found, after = str(text).rpartition(SENTINEL)
    parts = after.split()

    if not found or not parts:
        raise SystemExit(
            "the chat template closes no assistant turn after its content; set "
            "model.target_suffix to the family's end-of-turn marker"
        )

    return parts[0]


def with_end_of_turn(cfg: Config, processor: Any) -> Config:
    """Append the end-of-turn marker to every target, unless one is configured.

    Args:
        cfg: The target config.
        processor: The checkpoint's processor, or a tokenizer.

    Returns:
        The config whose model.target_suffix closes the turn.
    """
    if cfg.model.target_suffix is not None or cfg.model.append_eos:
        return cfg

    marker = end_of_turn(processor, dict(cfg.model.chat_template_kwargs))
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = tokenizer(marker, add_special_tokens=False)["input_ids"]

    if len(ids) != 1:
        log.warning("end-of-turn marker %r is %d tokens, not one", marker, len(ids))

    log.info("targets end in %r, the marker that closes the turn", marker)
    model = cfg.model.model_copy(update={"target_suffix": marker})

    return cfg.model_copy(update={"model": model})


def load_processor(cfg: ModelCfg) -> Any:
    """Read the checkpoint's processor, which carries its chat template.

    Args:
        cfg: Model configuration naming the checkpoint.

    Returns:
        The processor.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)


def target_pattern(targets: Sequence[str], scoped: bool) -> str:
    """The regex PEFT matches module names against, in full.

    Args:
        targets: Module names to adapt, e.g. q_proj.
        scoped: Whether names must lie under the language model, which a host
            holding towers next to it needs.

    Returns:
        The pattern, which leaves routed experts out.
    """
    names = "|".join(re.escape(target) for target in targets)
    scope = rf"(?:.*\.)?{LANGUAGE_MODEL}\." if scoped else ""

    return rf"{NOT_ROUTED}{scope}.*\.(?:{names})"


def lora_layers(module: nn.Module) -> list[nn.Module]:
    """Every module the adapter was injected into.

    Args:
        module: The host, after injection.

    Returns:
        The LoRA layers, in module order.
    """
    from peft.tuners.lora import LoraLayer

    return [layer for layer in module.modules() if isinstance(layer, LoraLayer)]


def attach_lora(vlm: LoraVLM, options: Options) -> Any:
    """Inject the adapter into the host's projections, in place.

    Args:
        vlm: The loaded model, its own weights frozen.
        options: The run's options.

    Returns:
        The PeftModel around the host, which saves the adapter.
    """
    from peft import LoraConfig, get_peft_model

    host = vlm.host
    config = LoraConfig(
        r=options.rank,
        lora_alpha=options.alpha,
        lora_dropout=options.dropout,
        target_modules=target_pattern(options.targets, scoped=not vlm.text_only),
        bias="none",
    )
    peft_model = get_peft_model(host, config)
    peft_model.peft_config["default"].base_model_name_or_path = vlm.cfg.model_id
    # Injected modules come up in train mode; the dropout is switched on
    # per step instead, so generation never runs with it.
    host.eval()
    layers = lora_layers(host)
    trainable = sum(param.numel() for param in host.parameters() if param.requires_grad)
    log.info("adapter on %d modules, %d trainable parameters", len(layers), trainable)

    return peft_model


def set_dropout(layers: Sequence[nn.Module], active: bool) -> None:
    """Switch the adapter's dropout on for a step and off for everything else.

    Args:
        layers: The LoRA layers.
        active: Whether the dropout applies.
    """
    for layer in layers:
        for dropout in cast(nn.ModuleDict, layer.lora_dropout).values():
            dropout.train(active)


def warmup_cosine(steps: int, warmup: int) -> Callable[[int], float]:
    """The learning rate factor per step: linear warmup, then cosine decay.

    Args:
        steps: Total optimizer steps.
        warmup: Steps the rate rises over.

    Returns:
        The factor applied to the base rate at a given step.
    """

    def factor(step: int) -> float:
        """The factor at one step.

        Args:
            step: Steps taken so far.

        Returns:
            The multiplier of the base rate.
        """
        if step < warmup:
            return (step + 1) / warmup

        progress = (step - warmup) / max(steps - warmup, 1)

        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return factor


class Fit:
    """One adapter fitted on train, checked on val, its best kept on disk."""

    def __init__(
        self,
        cfg: Config,
        options: Options,
        vlm: LoraVLM,
        peft_model: Any,
        benchmark: Benchmark,
        items: tuple[list[Item], list[Item]],
        staging: Path,
        tracker: wandb.Run | None = None,
    ) -> None:
        """Set up the optimizer and schedule over the adapter's parameters.

        Args:
            cfg: The target config.
            options: The run's options.
            vlm: The model, the adapter already injected.
            peft_model: The PeftModel that saves the adapter.
            benchmark: Benchmark supplying the messages and the scoring.
            items: The train items and the val items.
            staging: Directory the best adapter is saved into.
            tracker: The wandb run each record is mirrored to, or None.
        """
        self.cfg = cfg
        self.options = options
        self.vlm = vlm
        self.peft_model = peft_model
        self.benchmark = benchmark
        self.train, self.val = items
        self.staging = staging
        self.tracker = tracker
        self.layers = lora_layers(vlm.host)
        self.params = [param for param in vlm.host.parameters() if param.requires_grad]

        if not self.params:
            raise SystemExit("the adapter has no trainable parameters")

        self.optimizer = torch.optim.AdamW(
            self.params, lr=options.lr, weight_decay=options.weight_decay
        )
        self.per_epoch = math.ceil(len(self.train) / options.batch_size)
        self.steps = self.per_epoch * options.epochs
        warmup = math.ceil(options.warmup * self.steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, warmup_cosine(self.steps, warmup)
        )
        self.eval_every = options.eval_every or self.per_epoch
        self.rng = random.Random(cfg.seed)
        self.history = cfg.run_path / "history.jsonl"
        self.step = 0
        self.best: float | None = None
        self.best_step = -1
        self.clean: float | None = None

    def run(self) -> dict[str, object]:
        """Fit for the configured epochs, saving each new best adapter.

        Returns:
            The selection: what it ran on, the best step, its score, the
            base model's score and the steps taken.
        """
        self.cfg.run_path.mkdir(parents=True, exist_ok=True)
        # A new job in this directory is a new run, not a continuation.
        self.history.unlink(missing_ok=True)
        log.info(
            "%d train items, %d val items, %d epochs of %d steps, %d steps in all, "
            "val checked every %d steps",
            len(self.train),
            len(self.val),
            self.options.epochs,
            self.per_epoch,
            self.steps,
            self.eval_every,
        )
        self._log({"step": 0, **self._evaluate()})
        order: list[Item] = []

        while self.step < self.steps:
            started = time.perf_counter()
            batch, order = self._next_batch(order)
            lr = float(self.optimizer.param_groups[0]["lr"])
            loss, grad_norm = self._step(batch)
            self.step += 1
            self.scheduler.step()
            record: dict[str, object] = {
                "step": self.step,
                "loss": loss,
                "grad_norm": grad_norm,
                "lr": lr,
                "seconds": time.perf_counter() - started,
                "batch": len(batch),
            }

            if self.step % self.eval_every == 0 or self.step == self.steps:
                record.update(self._evaluate())

            self._log(record)

        if self.best_step == 0:
            log.warning("no step beat the base model on val; the adapter selected is a no-op")

        return {
            "select_on": self.options.select_on,
            "metric_name": self.benchmark.metric_name,
            "best_step": self.best_step,
            "best_score": self.best,
            "val_clean": self.clean,
            "steps": self.steps,
            "steps_per_epoch": self.per_epoch,
        }

    def _next_batch(self, order: list[Item]) -> tuple[list[Item], list[Item]]:
        """Draw the next batch, reshuffling once the current epoch runs out.

        Args:
            order: Remaining items of the current epoch, may be empty.

        Returns:
            The batch and the leftover order after it was drawn.
        """
        batch: list[Item] = []

        while len(batch) < min(self.options.batch_size, len(self.train)):
            if not order:
                order = list(self.train)
                self.rng.shuffle(order)
            batch.append(order.pop())

        return batch, order

    def _messages(self, item: Item) -> list[dict[str, Any]]:
        """The item's prompt, without an image.

        Args:
            item: Item to build the prompt for.

        Returns:
            Its chat messages.
        """
        return self.benchmark.build_messages(item, with_image=False, site=self.cfg.injection.site)

    def _step(self, batch: list[Item]) -> tuple[float, float]:
        """One optimizer step, gradients accumulated one item at a time.

        Args:
            batch: Items whose target losses make up the step.

        Returns:
            The mean loss and the gradient norm before clipping.
        """
        self.optimizer.zero_grad(set_to_none=True)
        set_dropout(self.layers, True)
        total = 0.0

        try:
            for item in batch:
                loss = self.vlm.loss_on_target(self._messages(item), None, item.target)
                (loss / len(batch)).backward()
                total += float(loss.detach())
        finally:
            set_dropout(self.layers, False)

        clip = self.options.grad_clip or math.inf
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, clip))
        self.optimizer.step()

        return total / len(batch), norm

    def _evaluate(self) -> dict[str, object]:
        """Score the current adapter on val and save it when it is best.

        Returns:
            The val record, flagged with new_best on an improvement.
        """
        started = time.perf_counter()

        if self.options.select_on == "val_loss":
            with torch.no_grad():
                losses = [
                    float(self.vlm.loss_on_target(self._messages(item), None, item.target))
                    for item in self.val
                ]
            value = sum(losses) / len(losses)
            record: dict[str, object] = {"val_loss": value}
            # Selection is phrased as higher-is-better, so lower CE wins.
            selection = -value
        else:
            value, _ = evaluate_split(
                self.vlm, self.benchmark, None, self.val, self.cfg.injection.site
            )
            record = {"val_score": value, "metric_name": self.benchmark.metric_name}
            selection = value

        if self.clean is None:
            self.clean = value

        record["val_clean"] = self.clean

        if self.best is None or selection > self.best:
            self.best, self.best_step = selection, self.step
            # The embeddings are never adapted. PEFT's own check would read
            # the base config, which for a text-only host is the wrapper's,
            # and could take a vocab mismatch from it and save them anyway.
            self.peft_model.save_pretrained(str(self.staging), save_embedding_layers=False)
            record["new_best"] = True

        record["val_seconds"] = time.perf_counter() - started

        return record

    def _log(self, record: dict[str, object]) -> None:
        """Append the record to history.jsonl, wandb and a one-line summary.

        Args:
            record: Step record, carrying at least a step key.
        """
        with self.history.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        if self.tracker is not None:
            # The strings are constant over a run and live in the config.
            numbers = {
                key: value
                for key, value in record.items()
                if key != "step" and isinstance(value, (bool, int, float))
            }
            self.tracker.log(numbers, step=self.step)

        message = f"step {record['step']:>5}"

        if "loss" in record:
            message += f" loss {record['loss']:.4f} |g| {record['grad_norm']:.3e}"
            message += f" lr {record['lr']:.2e}"

        for key in ("val_score", "val_loss"):
            if key in record:
                message += f" {key} {record[key]:.4f} (clean {record['val_clean']:.4f})"

        if record.get("new_best"):
            message += " *best*"

        log.info(message)


def open_tracker(cfg: Config, options: Options) -> wandb.Run | None:
    """Open the fit's wandb run, as the image runs do under wandb.enabled.

    Args:
        cfg: The target config.
        options: The run's options, recorded beside the config.

    Returns:
        The wandb run, or None when tracking is disabled.
    """
    if not cfg.wandb.enabled:
        return None

    cfg.run_path.mkdir(parents=True, exist_ok=True)
    config = {**cfg.model_dump(mode="json"), "lora": asdict(options)}

    return open_wandb(cfg.train.run_name, cfg.run_path, config)


def adapter_dir(cfg: Config, options: Options, root: Path | None) -> Path:
    """Where the selected adapter goes.

    Args:
        cfg: The target config.
        options: The run's options.
        root: The adapter root, or None for <model_id>-lora.

    Returns:
        <root>/<benchmark>/<name>, suffixed _limit<N> for a capped run.
    """
    if root is None:
        checkpoint = Path(cfg.model.model_id.rstrip("/"))
        root = checkpoint.with_name(f"{checkpoint.name}-lora")

    name = options.name() + (f"_limit{options.limit}" if options.limit else "")

    return root / cfg.benchmark.name / name


def freeze(
    cfg: Config,
    options: Options,
    vlm: LoraVLM,
    selection: dict[str, object],
    counts: dict[str, int],
    staging: Path,
    dest: Path,
    started: str,
) -> Path:
    """Move the selected adapter into place beside how it was made.

    Args:
        cfg: The target config.
        options: The run's options.
        vlm: The model the adapter was fitted on.
        selection: What Fit.run returned.
        counts: Train and val item counts.
        staging: Directory holding the best adapter.
        dest: Directory it goes to; a frozen one is never overwritten.
        started: When the run started, ISO 8601.

    Returns:
        The directory holding the adapter and lora.meta.json.
    """
    pattern = target_pattern(options.targets, scoped=not vlm.text_only)
    meta: dict[str, object] = {
        "benchmark": cfg.benchmark.name,
        "model": Path(cfg.train.run_dir).name,
        "name": options.name(),
        "base_model": cfg.model.model_id,
        # Which checkpoint vLLM has to serve the adapter on.
        "served_on": "text_only" if vlm.text_only else "checkpoint",
        "adapter_sha256": _sha256((staging / WEIGHTS_NAME).read_bytes()),
        "lora_modules": len(lora_layers(vlm.host)),
        "trainable_parameters": sum(
            param.numel() for param in vlm.host.parameters() if param.requires_grad
        ),
        "hyperparameters": {
            "rank": options.rank,
            "alpha": options.alpha,
            "dropout": options.dropout,
            "lr": options.lr,
            "weight_decay": options.weight_decay,
            "epochs": options.epochs,
            "batch_size": options.batch_size,
            "warmup": options.warmup,
            "grad_clip": options.grad_clip,
            "eval_every": options.eval_every,
            "target_modules": list(options.targets),
            "target_pattern": pattern,
            "seed": options.seed,
        },
        "selection": selection,
        "target": {
            "model_id": cfg.model.model_id,
            "chat_template_kwargs": cfg.model.chat_template_kwargs,
            "target_suffix": cfg.model.target_suffix,
            "max_target_tokens": cfg.model.max_target_tokens,
            "max_sequence_tokens": cfg.model.max_sequence_tokens,
            "generate": cfg.model.generate,
            "max_new_tokens": cfg.model.max_new_tokens,
            "benchmark_options": cfg.benchmark.options,
            "system_prompt_sha256": _sha256(cfg.benchmark.system_prompt.encode()),
        },
        "splits": {**counts, "limit": options.limit},
        "run_dir": _display(cfg.run_path),
        "git_sha": git_sha(),
        "config_hash": cfg.hash(),
        "started_utc": started,
        "finished_utc": _now(),
    }
    (staging / META_NAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    if dest.exists():
        if not options.limit:
            raise SystemExit(f"{dest} exists; a frozen adapter is never overwritten")
        shutil.rmtree(dest)

    # One rename, so a kill never leaves a half-written adapter in place.
    staging.rename(dest)
    summary = {**meta, "adapter_dir": str(dest)}
    (cfg.run_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {dest}: best step {selection['best_step']}, val {selection['best_score']}")

    return dest


def show(
    cfg: Config,
    options: Options,
    benchmark: Benchmark,
    items: tuple[list[Item], list[Item]],
    dest: Path,
) -> None:
    """Print what a run would fit on and where its adapter would land.

    Args:
        cfg: The target config, its target_suffix resolved.
        options: The run's options.
        benchmark: Benchmark supplying the messages.
        items: The train items and the val items.
        dest: Where the selected adapter would land.
    """
    train, val = items
    item = train[0]
    messages = benchmark.build_messages(item, with_image=False, site=cfg.injection.site)
    per_epoch = math.ceil(len(train) / options.batch_size)

    print("===== target =====")
    print(f"benchmark: {cfg.benchmark.name} ({len(train)} train, {len(val)} val items)")
    print(f"model: {cfg.model.model_id}")
    print(f"chat_template_kwargs: {json.dumps(cfg.model.chat_template_kwargs)}")
    print(f"target suffix: {cfg.model.target_suffix!r}")
    print(f"target modules: {','.join(options.targets)} in the language model, routed experts out")
    print(f"steps: {per_epoch} per epoch, {per_epoch * options.epochs} in all")
    print(f"run dir: {cfg.run_path}")
    print(f"adapter goes to: {dest}")
    print(f"\n===== messages of the first train item, {item.id} =====")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print("\n===== its target =====")
    print(item.target + (cfg.model.target_suffix or ""))


def run(
    cfg: Config,
    options: Options,
    adapter_root: Path | None = None,
    build: Callable[[ModelCfg], LoraVLM] = LoraVLM,
    processor: Any | None = None,
) -> Path | None:
    """Fit, select and freeze one adapter, or with dry_run only show the setup.

    Args:
        cfg: The target config, thinking off.
        options: The run's options.
        adapter_root: Where adapters go, or None for <model_id>-lora.
        build: Loads the model from its config.
        processor: The checkpoint's processor, or None to read it from disk.

    Returns:
        The directory holding the selected adapter, or None for a dry run.
    """
    started = _now()
    set_seed(cfg.seed)
    benchmark = build_benchmark(cfg.benchmark)
    items = split_items(benchmark, options.limit, cfg.train.eval_items)
    dest = adapter_dir(cfg, options, adapter_root)

    # Before the weights load, which takes minutes.
    if not options.limit and dest.exists():
        raise SystemExit(f"{dest} exists; a frozen adapter is never overwritten")

    cfg = with_end_of_turn(cfg, processor if processor is not None else load_processor(cfg.model))

    if options.dry_run:
        show(cfg, options, benchmark, items, dest)
        return None

    vlm = build(cfg.model)
    # Again, so the adapter's initialization does not depend on what the
    # weight load drew.
    set_seed(cfg.seed)
    peft_model = attach_lora(vlm, options)
    staging = dest.with_name(dest.name + ".partial")

    # Whatever a killed run left behind.
    if staging.exists():
        shutil.rmtree(staging)

    staging.mkdir(parents=True)
    tracker = open_tracker(cfg, options)

    try:
        fit = Fit(cfg, options, vlm, peft_model, benchmark, items, staging, tracker)
        selection = fit.run()

        if tracker is not None:
            tracker.summary.update(
                {"best_step": selection["best_step"], "best_score": selection["best_score"]}
            )
    finally:
        if tracker is not None:
            tracker.finish()
            prune_wandb_logs(tracker)

    counts = {"train": len(items[0]), "val": len(items[1])}

    return freeze(cfg, options, vlm, selection, counts, staging, dest, started)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    options = options_from(args)
    config_path, base = resolve_config(args.benchmark)
    cfg = resolve_target(
        config_path, [f"seed={options.seed}", f"train.run_name={options.run_name()}", *args.set]
    )
    run_dir = Path(cfg.train.run_dir).resolve()

    if run_dir.parent != Path(base.train.run_dir).resolve():
        raise SystemExit(
            f"train.run_dir must be {base.train.run_dir}/<model>, as for the eval jobs, "
            f"e.g. --set train.run_dir={base.train.run_dir}/qwen3.5-4B; got {cfg.train.run_dir}"
        )

    run(cfg, options, adapter_root=args.adapter_root)


if __name__ == "__main__":
    main()
