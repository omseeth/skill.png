"""Training loop: one universal image per (benchmark, model, condition).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import torch
import wandb
from torch import Tensor

from .benchmarks.base import Benchmark, Item
from .benchmarks.registry import build_benchmark
from .config import BenchmarkCfg, Config, ModelCfg
from .evaluate import evaluate_split, read_skill, set_seed, with_skill
from .image.param import build_param
from .models.base import VLM
from .models.registry import build_model
from .results import git_sha

log = logging.getLogger(__name__)


@dataclass
class TrainState:
    step: int = 0
    # In an ensemble the scalars are the weighted means over its models, and
    # the dicts below hold each model's own value by label; empty otherwise.
    best_score: float | None = None
    best_step: int = -1
    val_clean_score: float | None = None
    val_clean_loss: float | None = None
    select_on: str = "val_metric"
    config_hash: str = ""
    meta: dict[str, object] = field(default_factory=dict)
    best_scores: dict[str, float] = field(default_factory=dict)
    val_clean_scores: dict[str, float] = field(default_factory=dict)
    val_clean_losses: dict[str, float] = field(default_factory=dict)


def model_labels(model_ids: list[str]) -> list[str]:
    """Name each model of a run by its checkpoint's basename.

    Args:
        model_ids: Checkpoint path or repo id of each model, primary first.

    Returns:
        One label per model, a repeated basename suffixed -2, -3 and so on.
    """
    labels: list[str] = []

    for model_id in model_ids:
        base = Path(model_id.rstrip("/")).name or "model"
        label, suffix = base, 2

        while label in labels:
            label, suffix = f"{base}-{suffix}", suffix + 1
        labels.append(label)

    return labels


def _norm(tensors: list[Tensor]) -> float:
    """Take the joint L2 norm of a list of tensors.

    Args:
        tensors: Gradient tensors, one per image leaf.

    Returns:
        The norm, 0.0 for an empty list.
    """
    squares = [(tensor.detach() ** 2).sum() for tensor in tensors]

    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def open_wandb(name: str, run_dir: Path, config: dict[str, object]) -> wandb.Run:
    """Open a new tracking run for one training job.

    Args:
        name: Run name shown in wandb.
        run_dir: Run directory the wandb files go under.
        config: Settings recorded as the run's config.

    Returns:
        The wandb run.
    """
    return wandb.init(
        name=name,
        dir=str(run_dir),
        config=config,
        settings=wandb.Settings(
            # Everything switched off here is either already on disk
            # (the config in the run directory) or of no use for a sweep
            # over configurations, and wandb would write a copy of it per
            # run. Console capture is on: the logging handler is bound to
            # the original stderr before wandb starts, so it takes a
            # file-descriptor redirect rather than "wrap" to catch it.
            # System metrics stay on to track GPU memory during training.
            console="redirect",
            save_code=False,
            x_save_requirements=False,
            x_disable_meta=False,
            x_disable_stats=False,
        ),
    )


def prune_wandb_logs(run: wandb.Run) -> None:
    """Delete the debug logs wandb leaves next to a finished run.

    The job's own output is captured under slurm/logs, and wandb sync
    uploads the transaction log rather than these. Call after finish, the
    writers hold them open until then.

    Args:
        run: The finished wandb run.
    """
    sync_dir = Path(str(run.dir)).parent
    logs_dir = sync_dir / "logs"
    # The core log is a symlink into the shared wandb cache, so the tree
    # below only removes the link; take the file it points at as well.
    core_log = (logs_dir / "debug-core.log").resolve()

    if core_log.is_file():
        core_log.unlink(missing_ok=True)
    shutil.rmtree(logs_dir, ignore_errors=True)

    for name in ("debug.log", "debug-internal.log"):
        (sync_dir.parent / name).unlink(missing_ok=True)


# Locked for the whole run. A POSIX lock, so it holds across the nodes of a
# shared file system and lapses with the job that took it.
RUN_LOCK = "train.lock"
# What a run leaves behind once it has saved an image.
RUN_IMAGES = ("best.png", "best.json", "latest.png", "latest.json")


def claim_run_dir(run_dir: Path, overwrite: bool) -> TextIO:
    """Lock the run directory, refusing one in use or holding an image.

    Args:
        run_dir: Directory the run writes into, created when missing.
        overwrite: Whether an earlier run's image there may be replaced.

    Returns:
        The open lock file; the lock lasts until it is closed.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    handle = (run_dir / RUN_LOCK).open("a")

    try:
        fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(
            f"{run_dir} is in use by another training job; give this run "
            "its own train.run_name"
        ) from None

    found = [name for name in RUN_IMAGES if (run_dir / name).exists()]

    if found and not overwrite:
        handle.close()
        raise FileExistsError(
            f"{run_dir} already holds {', '.join(found)} from an earlier "
            "run; give this run its own train.run_name, delete the "
            "directory, or set train.overwrite=true to replace them"
        )

    return handle


class Trainer:
    def __init__(
        self, cfg: Config, model: VLM | None = None, benchmark: Benchmark | None = None
    ) -> None:
        """Set up run directory, image parameter and optimizer for one run.

        Args:
            cfg: Full run configuration.
            model: Vision language model, built from cfg when None.
            benchmark: Benchmark supplying the items, built from cfg when None.
        """
        self.cfg = cfg
        set_seed(cfg.seed)
        self.run_dir = cfg.run_path
        # First of all, so a clash with another run costs nothing.
        self._run_lock = claim_run_dir(self.run_dir, cfg.train.overwrite)
        # Before the model is built, so a bad skill fails without the wait
        # on the weights.
        benchmark_cfg, self.skill_digest = self._benchmark_cfg(benchmark)
        self.model = model or build_model(cfg.model)
        self._refuse_untrainable(self.model, cfg.model)
        # The primary first. Each member is checked before the next one
        # loads, so a bad one fails without the wait on the rest.
        self.models = [self.model]

        for member_cfg in cfg.member_models:
            member = build_model(member_cfg)
            self._refuse_untrainable(member, member_cfg)
            self.models.append(member)
        self.weights = cfg.loss_weights
        self.labels = model_labels([c.model_id for c in (cfg.model, *cfg.member_models)])
        self.benchmark = benchmark or build_benchmark(benchmark_cfg)
        self.history_path = self.run_dir / "history.jsonl"
        self.state_path = self.run_dir / "state.json"
        # A new job in this directory is a new run, not a continuation.
        self.history_path.unlink(missing_ok=True)
        self.state = TrainState(config_hash=cfg.hash(), select_on=cfg.train.select_on)
        self.param = build_param(cfg.param, seed=cfg.seed, device=self._device())
        self.init_image = self.param.tensor().detach().clone()
        self.optimizer = torch.optim.Adam(
            self.param.leaves(), lr=cfg.train.lr, betas=(cfg.train.beta1, 0.999)
        )
        self.rng = random.Random(cfg.seed)
        # Built in run(), where the total step count is known.
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.wandb_run = self._init_wandb()

    def _benchmark_cfg(self, benchmark: Benchmark | None) -> tuple[BenchmarkCfg, str | None]:
        """Compose train.skill_path into the system prompt, when one is set.

        Args:
            benchmark: Prebuilt benchmark, which must already carry the skill.

        Returns:
            The benchmark config every item of this run is built from, and
            the skill's sha256, or None without a skill.
        """
        if not self.cfg.train.skill_path:
            return self.cfg.benchmark, None

        skill = read_skill(self.cfg.train.skill_path, "train.skill_path")
        composed = with_skill(self.cfg.benchmark, self.cfg.injection.skill_template, skill)

        if benchmark is not None and benchmark.cfg.system_prompt != composed.system_prompt:
            raise ValueError(
                "train.skill_path got a prebuilt benchmark whose system prompt "
                "does not carry the skill; build it from the composed prompt "
                "or pass none"
            )

        digest = hashlib.sha256(skill.encode()).hexdigest()
        log.info(
            "system turn carries the skill %s (%d chars, sha256 %s)",
            self.cfg.train.skill_path,
            len(skill),
            digest[:16],
        )

        return composed, digest

    @staticmethod
    def _refuse_untrainable(model: VLM, model_cfg: ModelCfg) -> None:
        """Refuse a backend that cannot carry gradient to the image.

        Refused here rather than at the first backward: under
        select_on=val_metric the clean baseline is measured first, and a
        reasoning model would spend that whole evaluation before finding out.

        Args:
            model: The loaded model.
            model_cfg: The model block it was built from.
        """
        if not model.trainable:
            raise ValueError(
                f"model.name={model_cfg.name!r} is an inference-only backend: "
                "nothing it returns carries gradient to the image, so there is "
                "nothing to optimize. Train with model.name=hf, then score the "
                "saved PNG with this backend."
            )

    def _build_scheduler(self, steps: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build the learning rate schedule for a run of the given length.

        Args:
            steps: Total optimizer steps the run takes.

        Returns:
            A cosine schedule decaying lr to zero over those steps, or None
            when the rate is held constant.
        """
        if self.cfg.train.lr_schedule == "constant":
            return None

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=steps, eta_min=0.0
        )

    def _lr(self) -> float:
        """Read the rate the next optimizer step will be taken with.

        Returns:
            The current learning rate of the image parameter group.
        """
        return float(self.optimizer.param_groups[0]["lr"])

    def _device(self) -> str | torch.device:
        """Read the model's device, falling back to CPU when it has none.

        Returns:
            The device the image parameter should live on.
        """
        device: object = getattr(self.model, "device", "cpu")
        return device if isinstance(device, (str, torch.device)) else "cpu"

    def _init_wandb(self) -> wandb.Run | None:
        """Open a new tracking run for this training job.

        Returns:
            The wandb run, or None when tracking is disabled.
        """
        if not self.cfg.wandb.enabled:
            return None

        return open_wandb(self.cfg.train.run_name, self.run_dir, self.cfg.model_dump(mode="json"))

    def run(self) -> TrainState:
        """Optimize the image for the configured number of epochs.

        Steps over shuffled train batches, evaluates on val at the configured
        interval and always writes a latest checkpoint, even on interrupt.

        Returns:
            The final training state, including the best score and step.
        """
        cfg = self.cfg.train
        train_items = self.benchmark.items("train")
        val_items = self.benchmark.items("val")

        if cfg.eval_items:
            val_items = val_items[: cfg.eval_items]

        if not train_items:
            raise ValueError("no training items")
        
        if not val_items:
            # best.png is defined as "the image that scored best on held-out 
            # benchmark data".
            raise ValueError(
                f"benchmark {self.cfg.benchmark.name!r} yielded no val items; checkpoint selection "
                "needs a held-out evaluation split (check benchmark.splits and train.eval_items)"
            )
        
        per_epoch = math.ceil(len(train_items) / max(cfg.batch_size, 1))
        steps = per_epoch * cfg.epochs
        log.info(
            "train=%d val=%d epochs=%d steps=%d",
            len(train_items),
            len(val_items),
            cfg.epochs,
            steps,
        )

        self.scheduler = self._build_scheduler(steps)
        self._clean_baseline(val_items)

        order: list[Item] = []
        evaluated_at = -1

        try:
            while self.state.step < steps:
                started = time.perf_counter()
                batch, order = self._next_batch(train_items, order)
                # Read before the schedule advances: this is the rate the
                # step just taken used.
                lr = self._lr()
                loss, grad_norm, extras = self._step(batch)
                self.state.step += 1
                record: dict[str, object] = {
                    "step": self.state.step,
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "lr": lr,
                    "seconds": time.perf_counter() - started,
                    "batch": len(batch),
                    "image_token_count": self.model.image_token_count,
                    **extras,
                    **self._image_deltas(),
                }

                if self.scheduler is not None:
                    self.scheduler.step()

                if self.state.step % cfg.eval_every == 0:
                    record.update(self._val_eval(val_items))
                    evaluated_at = self.state.step

                if self.state.step % max(cfg.log_every, 1) == 0 or "select_on" in record:
                    self._log(record)

            if evaluated_at != self.state.step:
                self._log(
                    {
                        "step": self.state.step,
                        "final": True,
                        **self._val_eval(val_items),
                        **self._image_deltas(),
                    }
                )
        except KeyboardInterrupt:
            log.warning("interrupted at step %d; saving latest", self.state.step)
        finally:
            self._save("latest")
            self._track_images()

            if self.wandb_run is not None:
                self.wandb_run.finish()
                self._prune_wandb_logs()

        return self.state

    def _clean_baseline(self, val_items: list[Item]) -> None:
        """Record the no-image val number this run is measured against.

        Measured on these exact items in this process, and only once per run.
        Under train.skill_path it is the skill alone, without the image. In an
        ensemble each model is measured, and the number is their weighted mean.

        Args:
            val_items: Held-out items scored without an image.
        """
        if self.cfg.train.select_on == "val_loss":

            if self.state.val_clean_loss is None:
                losses = [self._val_loss(model, val_items, image=None) for model in self.models]
                self.state.val_clean_loss = self._weighted(losses)
                self.state.val_clean_losses = self._by_model(losses)
                log.info(
                    "val clean loss = %.4f%s",
                    self.state.val_clean_loss,
                    self._models_text(self.state.val_clean_losses),
                )
            return

        if self.state.val_clean_score is None:
            scores = [
                evaluate_split(model, self.benchmark, None, val_items, self.cfg.injection.site)[0]
                for model in self.models
            ]
            self.state.val_clean_score = self._weighted(scores)
            self.state.val_clean_scores = self._by_model(scores)
            log.info(
                "val clean %s = %.4f%s",
                self.benchmark.metric_name,
                self.state.val_clean_score,
                self._models_text(self.state.val_clean_scores),
            )

    def _weighted(self, values: list[float]) -> float:
        """Average one number per model with the normalized loss weights.

        Args:
            values: One value per model, the primary first.

        Returns:
            The weighted mean, the value itself for a single model.
        """
        return sum(weight * value for weight, value in zip(self.weights, values, strict=True))

    def _by_model(self, values: list[float]) -> dict[str, float]:
        """Key one number per model by the model's label, in an ensemble.

        Args:
            values: One value per model, the primary first.

        Returns:
            The values by label, empty for a single model.
        """
        if len(self.models) == 1:
            return {}

        return dict(zip(self.labels, values, strict=True))

    @staticmethod
    def _models_text(values: dict[str, float]) -> str:
        """Spell out each model's value for a log line.

        Args:
            values: Values by model label, empty for a single model.

        Returns:
            The values in parentheses, or an empty string.
        """
        if not values:
            return ""

        return " (" + ", ".join(f"{label} {value:.4f}" for label, value in values.items()) + ")"

    def _next_batch(self, items: list[Item], order: list[Item]) -> tuple[list[Item], list[Item]]:
        """Draw the next batch, reshuffling once the current epoch runs out.

        Shuffle-cycle so every item is seen once per epoch.

        Args:
            items: All training items of the split.
            order: Remaining items of the current epoch, may be empty.

        Returns:
            The batch and the leftover order after it was drawn.
        """
        batch: list[Item] = []

        while len(batch) < min(self.cfg.train.batch_size, len(items)):

            if not order:
                order = list(items)
                self.rng.shuffle(order)
            batch.append(order.pop())

        return batch, order

    def _step(self, batch: list[Item]) -> tuple[float, float, dict[str, float | list[float]]]:
        """One optimizer step over a micro-batch.

        Args:
            batch: Items whose target losses are accumulated into one step.

        Returns:
            The weighted mean loss, the gradient norm, and what an ensemble
            logs per model beside them.
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss, extras = self._backward(batch)
        leaves = self.param.leaves()

        if self.cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(leaves, self.cfg.train.grad_clip)

        grad_norm = _norm([leaf.grad for leaf in leaves if leaf.grad is not None])
        self.optimizer.step()
        self.param.project_()

        return loss, grad_norm, extras

    def _backward(self, batch: list[Item]) -> tuple[float, dict[str, float | list[float]]]:
        """Leave the gradient of a micro-batch's loss on the image.

        Gradient accumulation rather than a padded batched forward: padding variable-length prompts
        across families with 4-D or packed attention masks is a well-known source of *silent*
        wrongness, and batching buys only wall time here.

        In an ensemble each model takes the whole batch in turn, and its
        image gradient is set aside before the next one starts. The image is
        then left with the weighted sum of those gradients.

        Args:
            batch: Items whose target losses are accumulated.

        Returns:
            The weighted mean loss, and in an ensemble each model's loss,
            gradient norm, time and image token count.
        """
        prompts = [
            self.benchmark.build_messages(item, with_image=True, site=self.cfg.injection.site)
            for item in batch
        ]
        leaves = self.param.leaves()
        losses: list[float] = []
        grads: list[list[Tensor]] = []
        seconds: list[float] = []

        for model in self.models:
            started = time.perf_counter()
            total = 0.0
            # One sequence at a time, gradients accumulated into the image
            # param; peak memory is one item's graph, not the batch's.
            for item, messages in zip(batch, prompts, strict=True):
                loss = self._item_loss(model, messages, self.param.tensor(), item)
                (loss / len(batch)).backward()
                total += float(loss.detach())

            losses.append(total / len(batch))
            grads.append([self._take_grad(leaf) for leaf in leaves])
            seconds.append(time.perf_counter() - started)

        norms = [_norm(grad) for grad in grads]

        for leaf, grad in zip(leaves, self._combine(grads, norms), strict=True):
            leaf.grad = grad

        extras: dict[str, float | list[float]] = {}

        for name, values in (("loss", losses), ("grad_norm", norms), ("seconds", seconds)):

            for label, value in self._by_model(values).items():
                extras[f"{name}/{label}"] = value

        if len(self.models) > 1:
            extras.update(self._model_extras())

        return self._weighted(losses), extras

    def _item_loss(
        self, model: VLM, messages: list[dict[str, Any]], image: Tensor, item: Item
    ) -> Tensor:
        """Take the training loss of one item through one model.

        Args:
            model: Model the loss is taken through.
            messages: The item's chat messages with an image slot.
            image: The [0,1] (3,H,W) image being optimized.
            item: The item, whose target the loss is taken on.

        Returns:
            The teacher-forced CE on the target, differentiable w.r.t. image.
        """
        return model.loss_on_target(messages, image, item.target)

    def _extra_meta(self) -> dict[str, object]:
        """Name the fields a variant of this trainer adds to every sidecar.

        Returns:
            Extra provenance fields, none for a plain run.
        """
        return {}

    @staticmethod
    def _take_grad(leaf: Tensor) -> Tensor:
        """Take the gradient accumulated on an image leaf off it.

        Args:
            leaf: Image leaf whose gradient one model's items accumulated.

        Returns:
            That gradient, zeros when none reached the leaf.
        """
        grad = leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
        leaf.grad = None

        return grad

    def _combine(self, grads: list[list[Tensor]], norms: list[float]) -> list[Tensor]:
        """Sum the models' image gradients with their loss weights.

        Under ensemble.normalize_grads each gradient is first scaled to
        unit norm, so every model pulls on the image with its weight alone.

        Args:
            grads: Each model's gradient, one tensor per image leaf.
            norms: Each model's gradient norm.

        Returns:
            The combined gradient, one tensor per image leaf.
        """
        scales = list(self.weights)

        if self.cfg.ensemble.normalize_grads:
            # A model whose gradient vanished adds nothing, not a NaN.
            scales = [
                weight / norm if norm > 0 else 0.0
                for weight, norm in zip(scales, norms, strict=True)
            ]

        combined: list[Tensor] = []

        for per_model in zip(*grads, strict=True):
            total = scales[0] * per_model[0]

            for scale, grad in zip(scales[1:], per_model[1:], strict=True):
                total = total + scale * grad
            combined.append(total)

        return combined

    def _model_extras(self) -> dict[str, float | list[float]]:
        """Collect what an ensemble logs per model beside its losses.

        Returns:
            Each model's image token count, and the peak memory reserved on
            each GPU a model's inputs sit on, in GB.
        """
        record: dict[str, float | list[float]] = {}

        for label, model in zip(self.labels, self.models, strict=True):
            count = model.image_token_count

            if count is not None:
                record[f"image_token_count/{label}"] = count

        devices = {
            torch.device(device)
            for device in (getattr(model, "device", "cpu") for model in self.models)
        }

        for device in devices:

            if device.type == "cuda":
                index = device.index if device.index is not None else torch.cuda.current_device()
                peak = torch.cuda.max_memory_reserved(index) / 1e9
                record[f"gpu_peak_gb/cuda:{index}"] = peak

        return record

    def _val_loss(self, model: VLM, items: list[Item], image: Tensor | None) -> float:
        """Teacher-forced CE on held-out items.

        The same objective as training, on data not trained on.

        Args:
            model: Model the loss is taken through.
            items: Held-out items to score.
            image: Image shown to the model, or None for the clean condition.

        Returns:
            The mean cross-entropy over the items.
        """
        total = 0.0

        with torch.no_grad():
            for item in items:
                messages = self.benchmark.build_messages(
                    item, with_image=image is not None, site=self.cfg.injection.site
                )
                total += float(model.loss_on_target(messages, image, item.target))

        return total / len(items)

    def _val_eval(self, val_items: list[Item]) -> dict[str, object]:
        """Score the current image on val and checkpoint it when it is best.

        An ensemble scores the image on each of its models and selects on
        the weighted mean of their numbers.

        Args:
            val_items: Held-out items the current image is scored on.

        Returns:
            The val record to log, flagged with new_best on an improvement.
        """
        image: Tensor = self.param.tensor().detach()
        record: dict[str, object] = {"n_val": len(val_items), "select_on": self.cfg.train.select_on}

        if self.cfg.train.select_on == "val_loss":
            losses = [self._val_loss(model, val_items, image) for model in self.models]
            val_loss = self._weighted(losses)
            record["val_loss"] = val_loss
            record["val_clean_loss"] = self.state.val_clean_loss
            values, cleans, name = losses, self.state.val_clean_losses, "loss"
            # Selection is phrased as higher-is-better everywhere, so lower CE
            # is a higher score.
            selections = [-loss for loss in losses]
            selection = -val_loss
        else:
            scores = [
                evaluate_split(model, self.benchmark, image, val_items, self.cfg.injection.site)[0]
                for model in self.models
            ]
            score = self._weighted(scores)
            record["val_score"] = score
            record["val_clean_score"] = self.state.val_clean_score
            record["metric_name"] = self.benchmark.metric_name
            values, cleans, name = scores, self.state.val_clean_scores, "score"
            selections = scores
            selection = score

        for label, value in self._by_model(values).items():
            record[f"val_{name}/{label}"] = value
            record[f"val_clean_{name}/{label}"] = cleans[label]

        if self.state.best_score is None or selection > self.state.best_score:
            self.state.best_score = selection
            self.state.best_scores = self._by_model(selections)
            self.state.best_step = self.state.step
            self._save("best")
            record["new_best"] = True
        self._save("latest")

        return record

    def _image_deltas(self) -> dict[str, float]:
        """Measure how far the image has drifted from its initialization.

        Returns:
            The L-infinity and L2 distances from the initial image.
        """
        with torch.no_grad():
            delta = self.param.tensor().detach() - self.init_image
            return {
                "linf_from_init": float(delta.abs().max()),
                "l2_from_init": float(delta.norm()),
            }

    def _log(self, record: dict[str, object]) -> None:
        """Append the record to history.jsonl and log a one-line summary.

        Args:
            record: Step record, must carry at least a step key.
        """
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._track(record)
        message = f"step {record['step']:>5}"

        if "loss" in record:
            message += f" loss {record['loss']:.4f} |g| {record['grad_norm']:.3e}"
            message += self._models_text(self._from_record(record, "loss"))

        if "val_score" in record or "val_loss" in record:

            if "val_loss" in record:
                value, clean, name = record["val_loss"], record.get("val_clean_loss"), "val loss"
                key = "val_loss"
            else:
                value, clean, name = record["val_score"], record.get("val_clean_score"), "val"
                key = "val_score"
            message += f" {name} {value:.4f}"

            if clean is not None:
                message += f" (clean {clean:.4f})"
            message += self._models_text(self._from_record(record, key))

            if record.get("new_best"):
                message += " *best*"

        log.info(message)

    def _from_record(self, record: dict[str, object], name: str) -> dict[str, float]:
        """Read each model's value of one quantity back out of a record.

        Args:
            record: Step record as written to history.jsonl.
            name: Quantity logged per model, e.g. loss.

        Returns:
            The values by model label, empty when the record has none.
        """
        values: dict[str, float] = {}

        for label in self.labels:
            value = record.get(f"{name}/{label}")

            if isinstance(value, (int, float)):
                values[label] = float(value)

        return values

    def _track(self, record: dict[str, object]) -> None:
        """Mirror the numbers of one history record to wandb.

        Only numbers are sent; the string fields of a record are constant over
        a run and already live in the wandb config.

        Args:
            record: Step record as written to history.jsonl.
        """
        if self.wandb_run is None:
            return

        payload: dict[str, object] = {
            key: value
            for key, value in record.items()
            if key != "step" and isinstance(value, (bool, int, float))
        }

        self.wandb_run.log(payload, step=self.state.step)

    def _track_images(self) -> None:
        """Send the best and the latest checkpoint to wandb, once per run.

        A frame per eval step left hundreds of PNGs per run behind, in the
        wandb media directory and on the server; the two checkpoints are the
        images that actually get used.
        """
        if self.wandb_run is None:
            return

        payload: dict[str, object] = {}

        for name in ("best", "latest"):
            path = self.run_dir / f"{name}.png"

            if path.exists():
                payload[name] = wandb.Image(str(path))

        if payload:
            self.wandb_run.log(payload, step=self.state.step)

    def _prune_wandb_logs(self) -> None:
        """Delete the debug logs wandb leaves next to a finished run."""
        if self.wandb_run is not None:
            prune_wandb_logs(self.wandb_run)

    def _save(self, name: str) -> None:
        """Write the image plus its provenance metadata and the run state.

        Args:
            name: Checkpoint stem written under the run directory.
        """
        ensemble: dict[str, object] = {}

        if len(self.models) > 1:
            # best_score and the clean numbers stay the weighted means, so
            # they compare with each other as in a single-model run.
            ensemble = {
                "models": [
                    {
                        "label": label,
                        "model_id": model_cfg.model_id,
                        # What the tower is fed, which the processor may
                        # have rounded from the configured resolution.
                        "resolution": list(
                            getattr(model, "size", None) or model_cfg.resolution or ()
                        ),
                        "weight": weight,
                    }
                    for label, model, model_cfg, weight in zip(
                        self.labels,
                        self.models,
                        (self.cfg.model, *self.cfg.member_models),
                        self.weights,
                        strict=True,
                    )
                ],
                "normalize_grads": self.cfg.ensemble.normalize_grads,
                "best_scores": self.state.best_scores,
                "val_clean_scores": self.state.val_clean_scores,
                "val_clean_losses": self.state.val_clean_losses,
            }

        self.param.save(
            self.run_dir / name,
            meta={
                **ensemble,
                **self._extra_meta(),
                "step": self.state.step,
                "run_name": self.cfg.train.run_name,
                "seed": self.cfg.seed,
                "beta1": self.cfg.train.beta1,
                "benchmark": self.cfg.benchmark.name,
                "model": self.cfg.model.model_id,
                "injection_site": self.cfg.injection.site,
                "skill_path": self.cfg.train.skill_path,
                "skill_digest": self.skill_digest,
                "best_score": self.state.best_score,
                "best_step": self.state.best_step,
                "select_on": self.cfg.train.select_on,
                "val_clean_score": self.state.val_clean_score,
                "val_clean_loss": self.state.val_clean_loss,
                "metric_name": self.benchmark.metric_name,
                "config_hash": self.cfg.hash(),
                "git_sha": git_sha(),
            },
        )
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2, sort_keys=True) + "\n")


def train(cfg: Config) -> TrainState:
    """Build a trainer from the config and run it to completion.

    Args:
        cfg: Full run configuration.

    Returns:
        The final training state.
    """
    return Trainer(cfg).run()
