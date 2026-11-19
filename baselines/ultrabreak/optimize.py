"""UltraBreak: one image optimized with patch views and the semantic loss.

The Skill.png trainer with UltraBreak's three changes (Cui et al., ICLR
2026, github.com/kaiyuanCui/UltraBreak). Each item's loss sees the image as
a randomly rotated, zoomed and placed patch on a white canvas, the loss is
the semantic one in place of CE, and a total variation penalty on the image
is added to it. The optimizer is set through ordinary config keys.
Validation and the saved image stay the bare image, as UltraBreak feeds its
patch at test, so best.png and latest.png are scored like a Skill.png image.

    uv run python baselines/ultrabreak/optimize.py -c configs/searchqa.yaml \\
        --set model.model_id=/path/to/qwen3_5-4B \\
        --set train.run_dir=runs/searchqa/qwen3.5-4B \\
        --tv-weight 0.5 --rotation-range -15 15 --scale-range 0.8 1.2

history.jsonl logs the semantic loss as loss, the CE of the same forward
as ce, the total variation as tv, and each item's view as rotation_deg,
scale, offset_y and offset_x.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast, runtime_checkable

import torch
from patch_view import (
    PATCH_FRACTION,
    ROTATION_RANGE,
    SCALE_RANGE,
    WHITE,
    max_offset,
    place_patch,
    sample_offset,
    sample_view,
)
from semantic_loss import NOISE, PE_ALPHA, TAU, semantic_loss
from torch import Tensor
from tv_loss import TV_WEIGHT, total_variation

from skillpng.benchmarks.base import Benchmark, Item
from skillpng.config import Config, load_config
from skillpng.models.base import VLM, token_ce
from skillpng.train import Trainer

log = logging.getLogger(__name__)

# Angles tried across the rotation range when looking for the largest box.
ANGLE_STEPS = 720


@runtime_checkable
class LogitsVLM(Protocol):
    """A model that hands back the logits its target loss is taken on."""

    def target_logits(
        self, messages: list[dict[str, Any]], image: Tensor | None, target: str
    ) -> tuple[Tensor, Tensor, tuple[int, int]]:
        """Run the teacher-forced forward the target loss is taken on.

        Args:
            messages: Chat messages in the processor's template format.
            image: A [0,1] image of shape (3,H,W), or None.
            target: The gold continuation the loss is taken on.

        Returns:
            The logits, the prompt and target ids, and the target's span.
        """
        ...


@dataclass(frozen=True)
class Options:
    """UltraBreak's own settings, the ones the config has no key for."""

    tau: float = TAU
    pe_alpha: float = PE_ALPHA
    noise: float = NOISE
    patch_fraction: float = PATCH_FRACTION
    background: float = WHITE
    # Weight of the total variation penalty; 0 leaves it out.
    tv_weight: float = TV_WEIGHT
    # Every item draws one angle in degrees and one zoom factor, uniformly.
    rotation_range: tuple[float, float] = ROTATION_RANGE
    scale_range: tuple[float, float] = SCALE_RANGE

    def __post_init__(self) -> None:
        """Refuse settings that leave the loss or the view undefined."""
        if self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau}")

        if self.pe_alpha < 0.0 or self.noise < 0.0:
            raise ValueError("pe_alpha and noise must not be negative")

        if not 0.0 < self.patch_fraction <= 1.0:
            raise ValueError(f"patch_fraction must be in (0, 1], got {self.patch_fraction}")

        if not 0.0 <= self.background <= 1.0:
            raise ValueError(f"background must be in [0, 1], got {self.background}")

        if self.tv_weight < 0.0:
            raise ValueError(f"tv_weight must not be negative, got {self.tv_weight}")

        for name, bounds in (
            ("rotation_range", self.rotation_range),
            ("scale_range", self.scale_range),
        ):

            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} {list(bounds)} must be [low, high]")

        if self.scale_range[0] <= 0.0:
            raise ValueError(f"scale_range {list(self.scale_range)} must be positive")


def check_fit(cfg: Config, options: Options) -> None:
    """Refuse a view range whose largest patch would not fit the canvas.

    Args:
        cfg: Full run configuration, read for the image size.
        options: UltraBreak's settings, read for the patch fraction and ranges.
    """
    low, high = options.rotation_range
    angles = [low + (high - low) * step / ANGLE_STEPS for step in range(ANGLE_STEPS + 1)]
    zoom = options.patch_fraction * max(options.scale_range)

    for degrees in angles:
        max_offset(cfg.param.height, cfg.param.width, degrees, zoom)


def input_embeddings(model: VLM) -> Tensor:
    """Read the input embedding matrix the semantic loss projects onto.

    Args:
        model: A loaded model whose network sits under its model attribute.

    Returns:
        The (V,D) embedding weight, detached.
    """
    network = getattr(model, "model", None)

    if network is None or not isinstance(model, LogitsVLM):
        raise TypeError(
            f"{type(model).__name__} exposes no logits or embedding matrix; "
            "UltraBreak trains through model.name=hf"
        )

    table = cast(Tensor, network.get_input_embeddings().weight).detach()
    head = network.get_output_embeddings()

    if head is not None and head.weight.shape[0] != table.shape[0]:
        raise ValueError(
            f"the head scores {head.weight.shape[0]} tokens but the input "
            f"embedding matrix has {table.shape[0]} rows"
        )

    return table


class UltraBreakTrainer(Trainer):
    def __init__(
        self,
        cfg: Config,
        options: Options,
        model: VLM | None = None,
        benchmark: Benchmark | None = None,
    ) -> None:
        """Check the views fit, then set up the run as the Trainer does.

        Args:
            cfg: Full run configuration.
            options: UltraBreak's own settings.
            model: Vision language model, built from cfg when None.
            benchmark: Benchmark supplying the items, built from cfg when None.
        """
        if cfg.member_models:
            raise ValueError("UltraBreak optimizes against one model; leave ensemble.members empty")

        # Before the weights load, so a bad range costs nothing.
        check_fit(cfg, options)
        self.options = options
        # Streams of their own, so the views leave the batch order of a seed
        # unchanged, and the offsets its rotation and zoom draws.
        self.view_rng = random.Random(cfg.seed + 1)
        self.offset_rng = random.Random(cfg.seed + 2)
        self._ces: list[float] = []
        self._views: list[tuple[float, float]] = []
        self._offsets: list[tuple[int, int]] = []
        super().__init__(cfg, model, benchmark)
        self.table = input_embeddings(self.model)

        if self.wandb_run is not None:
            self.wandb_run.config.update({"ultrabreak": asdict(options)})
        log.info("UltraBreak: %s", asdict(options))

    def _backward(self, batch: list[Item]) -> tuple[float, dict[str, float | list[float]]]:
        """Backward the semantic loss on patch views, then the TV penalty.

        Args:
            batch: Items whose losses are accumulated.

        Returns:
            The mean semantic loss, and the step's CE, TV term and views.
        """
        self._ces, self._views, self._offsets = [], [], []
        loss, extras = super()._backward(batch)
        extras.update(self._tv_penalty())
        extras["ce"] = sum(self._ces) / len(self._ces)
        extras["rotation_deg"] = [degrees for degrees, _ in self._views]
        extras["scale"] = [scale for _, scale in self._views]
        extras["offset_y"] = [float(dy) for dy, _ in self._offsets]
        extras["offset_x"] = [float(dx) for _, dx in self._offsets]

        return loss, extras

    def _tv_penalty(self) -> dict[str, float]:
        """Backward the weighted total variation of the bare image.

        Taken on the untransformed image, the one that gets saved.

        Returns:
            The raw total variation and its weighted penalty, empty when the
            weight is zero.
        """
        if not self.options.tv_weight:
            return {}

        tv = total_variation(self.param.tensor())
        penalty = self.options.tv_weight * tv
        penalty.backward()

        return {"tv": float(tv.detach()), "prior_loss": float(penalty.detach())}

    def _patch_view(self, image: Tensor) -> Tensor:
        """Paste the image as a patch at a random spot of a plain canvas.

        Every call draws its own angle, zoom and offset, so each item of a
        step sees a view of its own.

        Args:
            image: The [0,1] (3,H,W) image being optimized.

        Returns:
            The canvas one item's loss sees.
        """
        options = self.options
        degrees, scale = sample_view(self.view_rng, options.rotation_range, options.scale_range)
        height, width = image.shape[-2:]
        zoom = options.patch_fraction * scale
        offset = sample_offset(self.offset_rng, height, width, degrees, zoom)
        self._views.append((degrees, scale))
        self._offsets.append(offset)

        return place_patch(
            image,
            degrees,
            scale,
            offset,
            fraction=options.patch_fraction,
            background=options.background,
        )

    def _item_loss(
        self, model: VLM, messages: list[dict[str, Any]], image: Tensor, item: Item
    ) -> Tensor:
        """Take the semantic loss of one item on a patch view of the image.

        The CE of the same forward is set aside for the log.

        Args:
            model: Model the loss is taken through.
            messages: The item's chat messages with an image slot.
            image: The [0,1] (3,H,W) image being optimized.
            item: The item, whose target the loss is taken on.

        Returns:
            The semantic loss on the target, differentiable w.r.t. image.
        """
        canvas = self._patch_view(image)
        logits, input_ids, span = cast(LogitsVLM, model).target_logits(
            messages, canvas, item.target
        )

        if self.table.device != logits.device:
            self.table = self.table.to(logits.device)

        with torch.no_grad():
            self._ces.append(float(token_ce(logits, input_ids, span)))

        return semantic_loss(
            logits,
            input_ids,
            span,
            self.table,
            tau=self.options.tau,
            pe_alpha=self.options.pe_alpha,
            noise=self.options.noise,
        )

    def _extra_meta(self) -> dict[str, object]:
        """Record UltraBreak's settings in every sidecar.

        Returns:
            The loss and the options under an ultrabreak key.
        """
        return {"ultrabreak": {"loss": "semantic_attention", **asdict(self.options)}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the config path, the overrides and UltraBreak's own settings.

    Args:
        argv: Command line arguments, sys.argv when None.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Optimize one image the UltraBreak way.")
    parser.add_argument("--config", "-c", required=True, help="YAML config path")
    parser.add_argument(
        "--set",
        "-s",
        dest="overrides",
        action="append",
        default=[],
        help="override a config key: --set train.lr=0.01 (repeatable)",
    )
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--pe-alpha", type=float, default=PE_ALPHA)
    parser.add_argument("--noise", type=float, default=NOISE)
    parser.add_argument("--patch-fraction", type=float, default=PATCH_FRACTION)
    parser.add_argument("--background", type=float, default=WHITE)
    parser.add_argument("--tv-weight", type=float, default=TV_WEIGHT)
    parser.add_argument(
        "--rotation-range",
        type=float,
        nargs=2,
        default=ROTATION_RANGE,
        metavar=("LOW", "HIGH"),
        help="rotation angle in degrees, drawn per item",
    )
    parser.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        default=SCALE_RANGE,
        metavar=("LOW", "HIGH"),
        help="zoom factor, drawn per item",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    options = Options(
        tau=args.tau,
        pe_alpha=args.pe_alpha,
        noise=args.noise,
        patch_fraction=args.patch_fraction,
        background=args.background,
        tv_weight=args.tv_weight,
        rotation_range=tuple(args.rotation_range),
        scale_range=tuple(args.scale_range),
    )
    cfg = load_config(args.config, args.overrides)
    state = UltraBreakTrainer(cfg, options).run()
    summary = {
        "steps": state.step,
        "best_step": state.best_step,
        "best_score": state.best_score,
        "val_clean_score": state.val_clean_score,
        "run_dir": str(cfg.run_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
