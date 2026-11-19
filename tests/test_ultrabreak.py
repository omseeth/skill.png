"""The UltraBreak baseline: its losses, its patch view and its trainer.

Offline and without weights. The loss is checked against a CPU port of the
reference implementation, the trainer runs on test_loop's tiny model with
a logits route and an embedding matrix added.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from skillpng.config import Config
from skillpng.models.base import IGNORE_INDEX, masked_labels, target_span
from skillpng.models.registry import MODELS
from test_loop import VOCAB, TinyVLM, _config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "ultrabreak"))

import optimize as ultrabreak  # noqa: E402
from patch_view import (  # noqa: E402
    max_offset,
    place_patch,
    rotate_scale,
    sample_offset,
    sample_view,
    shift,
)
from semantic_loss import semantic_loss, sinusoidal_encoding  # noqa: E402
from tv_loss import total_variation  # noqa: E402

GRAY = 0.5


def reference_loss(logits: Tensor, labels: Tensor, table: Tensor, epsilon: float) -> Tensor:
    """Port semantic_similarity_loss(mode="attention") of the reference to CPU.

    Args:
        logits: Full-sequence logits (B,T,V).
        labels: Target ids, IGNORE_INDEX outside the target.
        table: Embedding matrix (V,D).
        epsilon: Std of the noise on the expected embeddings.

    Returns:
        The reference's scalar loss.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    batch, length, _ = logits.size()
    dim = table.size(1)
    probs = torch.softmax(logits, dim=-1)
    expected = probs @ table
    target = table[labels.clamp(min=0)]
    mask = (labels != IGNORE_INDEX).float()
    expected = expected + torch.randn_like(expected) * epsilon
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    rates = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim)
    encoding[:, 0::2] = torch.sin(position * rates)
    encoding[:, 1::2] = torch.cos(position * rates)
    encoding = encoding.unsqueeze(0)
    pred = F.normalize(expected + 0.01 * encoding, dim=-1)
    tgt = F.normalize(target + 0.01 * encoding, dim=-1)
    scores = torch.bmm(pred, tgt.transpose(1, 2)) / 0.5
    causal = torch.triu(torch.ones((length, length))).bool().unsqueeze(0).expand(batch, -1, -1)
    keep = causal & mask.bool().unsqueeze(1).expand(-1, length, -1)
    scores = scores.masked_fill(~keep, float("-inf"))
    scores[~(keep.sum(dim=-1) > 0)] = 0.0
    attended = torch.bmm(torch.softmax(scores, dim=-1), target)
    sim = F.cosine_similarity(expected, attended, dim=-1) * mask

    return ((1 - sim) * mask).sum() / mask.sum()


def _sequence(
    length: int = 12, target: int = 4, vocab: int = 20, dim: int = 8
) -> tuple[Tensor, Tensor, tuple[int, int], Tensor]:
    """Draw random logits, ids, a target span at the end, and embeddings.

    Args:
        length: Sequence length.
        target: Target length.
        vocab: Vocabulary size.
        dim: Embedding width.

    Returns:
        Logits, ids, the target span and the embedding matrix.
    """
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(1, length, vocab, generator=generator)
    ids = torch.randint(0, vocab, (1, length), generator=generator)
    table = torch.randn(vocab, dim, generator=generator)

    return logits, ids, (length - target, length), table


def test_the_loss_matches_the_reference_implementation() -> None:
    logits, ids, span, table = _sequence()
    labels = masked_labels(ids, span)

    ours = semantic_loss(logits, ids, span, table, noise=0.0)
    theirs = reference_loss(logits, labels, table, epsilon=0.0)

    assert torch.allclose(ours, theirs, atol=1e-6)


def test_the_loss_reads_trimmed_logits_as_the_last_positions() -> None:
    logits, ids, span, table = _sequence()
    keep = span[1] - span[0] + 1

    full = semantic_loss(logits, ids, span, table, noise=0.0)
    trimmed = semantic_loss(logits[:, -keep:], ids, span, table, noise=0.0)

    assert torch.allclose(full, trimmed)


def test_the_loss_refuses_logits_that_miss_the_first_target_token() -> None:
    logits, ids, span, table = _sequence()
    keep = span[1] - span[0]

    with pytest.raises(ValueError, match="do not cover"):
        semantic_loss(logits[:, -keep:], ids, span, table)


def test_the_loss_falls_as_the_target_gets_likelier() -> None:
    logits, ids, span, table = _sequence()
    logits.requires_grad_(True)

    loss = semantic_loss(logits, ids, span, table, noise=0.0)
    loss.backward()

    assert logits.grad is not None and logits.grad[:, span[0] - 1 : span[1] - 1].abs().sum() > 0
    assert logits.grad[:, : span[0] - 1].abs().sum() == 0

    sure = logits.detach().clone()
    for position in range(*span):
        sure[0, position - 1, ids[0, position]] += 20.0

    assert semantic_loss(sure, ids, span, table, noise=0.0) < loss


def test_the_encoding_takes_an_odd_width() -> None:
    encoding = sinusoidal_encoding(torch.arange(3), 5)

    assert encoding.shape == (3, 5)
    assert torch.allclose(encoding[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))


def test_total_variation_is_zero_when_flat_and_largest_on_a_checkerboard() -> None:
    """Every neighbour differs by one on a checkerboard, so both means are one."""
    board = ((torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2).float()

    assert float(total_variation(torch.full((3, 8, 8), GRAY))) == 0.0
    assert float(total_variation(board.expand(3, 8, 8))) == pytest.approx(2.0)


def test_no_rotation_and_no_zoom_return_the_image_itself() -> None:
    """No resample happens for the identity."""
    image = torch.rand(3, 8, 8)

    assert rotate_scale(image, 0.0, 1.0) is image


def test_a_quarter_turn_stays_rigid_on_a_non_square_image() -> None:
    """A centered 2x4 bar turns into a 4x2 bar, not a squashed one."""
    image = torch.full((3, 8, 16), GRAY)
    image[:, 3:5, 6:10] = 1.0
    expected = torch.full_like(image, GRAY)
    expected[:, 2:6, 7:9] = 1.0

    assert torch.allclose(rotate_scale(image, 90.0, 1.0, fill=GRAY), expected, atol=1e-5)


def test_zooming_out_fills_the_border_and_keeps_the_center() -> None:
    """At half size the content covers the central quarter, the fill around it."""
    view = rotate_scale(torch.zeros(3, 16, 16), 0.0, 0.5, fill=GRAY)

    assert torch.allclose(view[:, 4:12, 4:12], torch.zeros(3, 8, 8))
    assert torch.allclose(view[:, :4], torch.full((3, 4, 16), GRAY))


def test_the_resample_carries_gradient_back_to_the_image() -> None:
    image = torch.rand(3, 8, 8, requires_grad=True)
    rotate_scale(image, 10.0, 1.1).sum().backward()

    assert image.grad is not None
    assert image.grad.abs().sum() > 0


def test_views_are_drawn_inside_their_ranges() -> None:
    """Draws land in the given ranges and are not all the same."""
    rng = random.Random(0)
    draws = [sample_view(rng, (-15.0, 15.0), (0.8, 1.2)) for _ in range(100)]

    assert all(-15.0 <= degrees <= 15.0 and 0.8 <= scale <= 1.2 for degrees, scale in draws)
    assert len(set(draws)) == len(draws)
    assert sample_view(rng, (0.0, 0.0), (1.0, 1.0)) == (0.0, 1.0)


def test_shift_moves_the_content_and_fills_what_it_uncovers() -> None:
    image = torch.arange(3 * 6 * 7, dtype=torch.float32).reshape(3, 6, 7)

    moved = shift(image, 2, -3, fill=-1.0)

    assert moved.shape == image.shape
    assert torch.equal(moved[:, 2:, :4], image[:, :4, 3:])
    assert (moved[:, :2] == -1.0).all()
    assert (moved[:, :, 4:] == -1.0).all()


def test_the_identity_view_is_the_image_itself() -> None:
    image = torch.rand(3, 16, 16)

    assert torch.equal(place_patch(image, 0.0, 1.0, (0, 0), fraction=1.0), image)


@pytest.mark.parametrize("corner", [(1, 1), (-1, -1), (1, -1)])
def test_the_largest_patch_at_the_farthest_offset_stays_on_the_canvas(
    corner: tuple[int, int],
) -> None:
    """A black patch keeps its whole area of ink: nothing was cut off."""
    side, fraction, scale, degrees = 96, 2.0 / 3.0, 1.0, 15.0
    room_y, room_x = max_offset(side, side, degrees, fraction * scale)
    offset = (corner[0] * room_y, corner[1] * room_x)

    canvas = place_patch(torch.zeros(3, side, side), degrees, scale, offset, fraction=fraction)
    ink = float((1.0 - canvas).sum()) / 3

    assert room_y > 0 and room_x > 0
    assert ink == pytest.approx((fraction * scale * side) ** 2, rel=0.02)
    assert float(canvas.min()) >= 0.0 and float(canvas.max()) <= 1.0


def test_the_reference_ranges_just_fit_the_canvas() -> None:
    """Its largest patch, zoomed 1.2 and turned 15 degrees, has no room left."""
    assert max_offset(256, 256, 15.0, 2.0 / 3.0 * 1.2) == (2, 2)
    assert max_offset(96, 96, -15.0, 2.0 / 3.0 * 1.2) == (0, 0)


def test_a_patch_larger_than_the_canvas_is_refused() -> None:
    with pytest.raises(ValueError, match="more than"):
        max_offset(64, 64, 15.0, 1.2)


def test_offsets_cover_every_spot_that_fits_and_no_other() -> None:
    rng = random.Random(0)
    room = max_offset(32, 32, 10.0, 0.6)
    draws = {sample_offset(rng, 32, 32, 10.0, 0.6) for _ in range(4000)}

    assert {dy for dy, _ in draws} == set(range(-room[0], room[0] + 1))
    assert {dx for _, dx in draws} == set(range(-room[1], room[1] + 1))


def test_the_view_carries_gradient_to_the_patch() -> None:
    image = torch.rand(3, 32, 32, requires_grad=True)

    place_patch(image, 12.0, 0.9, (3, -2)).sum().backward()

    assert image.grad is not None and image.grad.abs().sum() > 0


class _Network:
    """The two embedding accessors of an HF model, over TinyVLM's layers."""

    def __init__(self, net: nn.ModuleDict) -> None:
        self.net = net

    def get_input_embeddings(self) -> nn.Module:
        return self.net["embed"]

    def get_output_embeddings(self) -> nn.Module:
        return self.net["head"]


@MODELS.register("_test_ultrabreak")
class LogitsVLM(TinyVLM):
    """TinyVLM with HFVLM's logits route, recording what it is shown."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.model = _Network(self.net)
        self.trained_on: list[Tensor] = []
        self.generated_on: list[Tensor] = []

    def target_logits(
        self, messages: list[dict[str, Any]], image: Tensor | None, target: str
    ) -> tuple[Tensor, Tensor, tuple[int, int]]:
        if image is not None and torch.is_grad_enabled():
            self.trained_on.append(image.detach().clone())
        prompt = self._ids(messages)
        appended = torch.tensor([[abs(hash(word)) % VOCAB for word in target.split()]])
        ids = torch.cat([prompt, appended], dim=1)
        return self._logits(ids, image), ids, target_span(prompt.shape[1], appended.shape[1])

    def generate(self, messages: list[dict[str, Any]], image: Tensor | None, **kw: Any) -> str:
        if image is not None:
            self.generated_on.append(image.detach().clone())
        return super().generate(messages, image, **kw)


def _ultrabreak_config(tmp_path: Path, **train: Any) -> Config:
    """Build test_loop's config with the UltraBreak recipe on the logits model.

    Args:
        tmp_path: Directory the run writes into.
        **train: Train keys set on top of the recipe.

    Returns:
        The run configuration.
    """
    data = _config(tmp_path).model_dump()
    data["model"]["name"] = "_test_ultrabreak"
    data["param"]["init"] = "random"
    data["train"].update(
        {
            "run_name": "ub",
            "batch_size": 3,
            "lr": 0.001,
            "beta1": 0.9,
            **train,
        }
    )

    return Config.model_validate(data)


def test_training_sees_patches_and_validation_the_bare_image(tmp_path: Path) -> None:
    trainer = ultrabreak.UltraBreakTrainer(_ultrabreak_config(tmp_path), ultrabreak.Options())
    trainer.run()
    model = trainer.model
    assert isinstance(model, LogitsVLM)

    assert model.trained_on and model.generated_on
    assert all(float((view == 1.0).float().mean()) > 0.2 for view in model.trained_on)
    assert all(float((image == 1.0).float().mean()) < 0.05 for image in model.generated_on)

    run = tmp_path / "runs" / "ub"
    steps = [
        record
        for record in map(json.loads, (run / "history.jsonl").read_text().splitlines())
        if "loss" in record
    ]
    assert steps
    for record in steps:
        assert 0.0 <= record["loss"] <= 2.0
        assert record["ce"] > 0.0
        assert len(record["offset_y"]) == len(record["offset_x"]) == record["batch"]
        # Every item drew a view of its own.
        assert len(set(record["rotation_deg"])) == record["batch"]
        assert len(set(record["scale"])) == record["batch"]
        assert all(-15 <= degrees <= 15 for degrees in record["rotation_deg"])
        assert all(0.8 <= scale <= 1.2 for scale in record["scale"])
        assert record["prior_loss"] == pytest.approx(0.5 * record["tv"])
    assert any(record["batch"] > 1 for record in steps)

    sidecar = json.loads((run / "latest.json").read_text())
    assert sidecar["ultrabreak"]["loss"] == "semantic_attention"
    assert sidecar["ultrabreak"]["patch_fraction"] == pytest.approx(2.0 / 3.0)
    assert sidecar["ultrabreak"]["tv_weight"] == 0.5
    assert sidecar["ultrabreak"]["rotation_range"] == [-15, 15]
    assert sidecar["ultrabreak"]["scale_range"] == [0.8, 1.2]


def test_the_tv_penalty_pulls_an_outlier_toward_its_neighbours(tmp_path: Path) -> None:
    """A lone bright pixel is pushed down, its neighbours up."""
    options = ultrabreak.Options(tv_weight=1.0)
    trainer = ultrabreak.UltraBreakTrainer(_ultrabreak_config(tmp_path), options)
    leaf = trainer.param.leaves()[0]
    with torch.no_grad():
        leaf.fill_(0.5)
        leaf[:, 8, 8] = 0.9
    leaf.grad = None

    record = trainer._tv_penalty()

    assert set(record) == {"tv", "prior_loss"}
    assert leaf.grad is not None
    assert (leaf.grad[:, 8, 8] > 0).all()
    assert (leaf.grad[:, 7, 8] < 0).all()
    assert (leaf.grad[:, 8, 7] < 0).all()


def test_a_zero_tv_weight_leaves_the_penalty_out(tmp_path: Path) -> None:
    options = ultrabreak.Options(tv_weight=0.0)
    trainer = ultrabreak.UltraBreakTrainer(_ultrabreak_config(tmp_path), options)
    trainer.run()

    lines = (tmp_path / "runs" / "ub" / "history.jsonl").read_text().splitlines()
    assert not any({"tv", "prior_loss"} & set(json.loads(line)) for line in lines)


def test_the_recipe_arrives_as_command_line_flags() -> None:
    """The slurm scripts pass it this way; a negative angle is no flag."""
    args = ultrabreak.parse_args(
        ["-c", "x.yaml", "--tv-weight", "0.1", "--rotation-range", "-10", "10",
         "--scale-range", "0.9", "1.1"]
    )  # fmt: skip
    defaults = ultrabreak.parse_args(["-c", "x.yaml"])

    assert (args.tv_weight, args.rotation_range, args.scale_range) == (
        0.1, [-10.0, 10.0], [0.9, 1.1],
    )  # fmt: skip
    assert ultrabreak.Options(
        tv_weight=defaults.tv_weight,
        rotation_range=tuple(defaults.rotation_range),
        scale_range=tuple(defaults.scale_range),
    ) == ultrabreak.Options(tv_weight=0.5, rotation_range=(-15.0, 15.0), scale_range=(0.8, 1.2))


def test_a_range_whose_patch_overflows_is_refused_before_the_run_starts(tmp_path: Path) -> None:
    cfg = _ultrabreak_config(tmp_path)

    with pytest.raises(ValueError, match="more than"):
        ultrabreak.UltraBreakTrainer(cfg, ultrabreak.Options(patch_fraction=1.0))

    assert not (tmp_path / "runs" / "ub").exists()


def test_an_ensemble_is_refused(tmp_path: Path) -> None:
    cfg = _ultrabreak_config(tmp_path)
    data = cfg.model_dump()
    data["ensemble"]["members"] = [{"model": {"model_id": "other"}}]

    with pytest.raises(ValueError, match="one model"):
        ultrabreak.UltraBreakTrainer(Config.model_validate(data), ultrabreak.Options())


def test_a_model_without_a_logits_route_is_refused(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    with pytest.raises(TypeError, match="no logits"):
        ultrabreak.UltraBreakTrainer(cfg, ultrabreak.Options())


@pytest.mark.parametrize(
    "options",
    [
        {"tau": 0.0},
        {"noise": -1.0},
        {"patch_fraction": 1.5},
        {"background": 2.0},
        {"tv_weight": -0.1},
        {"rotation_range": (15.0, -15.0)},
        {"rotation_range": (-15.0,)},
        {"scale_range": (1.2, 0.8)},
        {"scale_range": (0.0, 1.2)},
    ],
)
def test_settings_without_meaning_are_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ultrabreak.Options(**options)
