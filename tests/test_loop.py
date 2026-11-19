"""Train -> checkpoint -> evaluate -> results row, without weights.

The fixtures below are test scaffolding, not a shipped backend: a few `nn.Module` layers standing
in for a vision tower and a language head, and a benchmark over synthetic items. They exist so the
wiring between the loop, the checkpoint, the conditions, and the results file stays covered on a
laptop. Nothing here is importable from the package.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor, nn

from skillpng.benchmarks.base import Benchmark, Item, inject_image, split_by_group
from skillpng.benchmarks.registry import BENCHMARKS
from skillpng.config import Config
from skillpng.evaluate import run_eval
from skillpng.image.preprocess import PreprocessSpec, preprocess_image
from skillpng.models.base import VLM, target_span, token_ce
from skillpng.models.registry import MODELS
from skillpng.results import IMAGE_FIELDS, read_rows
from skillpng.train import RUN_LOCK, Trainer

VOCAB = 32
SIZE = (16, 16)


@MODELS.register("_test_tiny")
class TinyVLM(VLM):
    """Vision tower + projector + language head, wired the way `HFVLM` wires them."""

    def __init__(self, cfg: Any) -> None:
        torch.manual_seed(0)
        self.net = nn.ModuleDict(
            {
                "tower": nn.Conv2d(3, 8, 4, 4),
                "proj": nn.Linear(8, 8),
                "embed": nn.Embedding(VOCAB, 8),
                "head": nn.Linear(8, VOCAB),
            }
        )
        self.net.requires_grad_(False)  # the image is the only leaf
        self._spec = PreprocessSpec(
            image_mean=(0.5,) * 3, image_std=(0.5,) * 3, height=16, width=16
        )
        self.device = "cpu"
        self.size = SIZE

    @property
    def preprocess(self) -> PreprocessSpec:
        return self._spec

    @property
    def image_token_count(self) -> int | None:
        return 16

    def _ids(self, messages: list[dict[str, Any]]) -> Tensor:
        words = [
            block.get("text", "<image>") for message in messages for block in message["content"]
        ]
        tokens = [abs(hash(word)) % VOCAB for word in " ".join(words).split()]
        return torch.tensor([tokens or [0]])

    def _logits(self, ids: Tensor, image: Tensor | None) -> Tensor:
        embeddings = self.net["embed"](ids)
        if image is not None:
            pixel_values = preprocess_image(image, self._spec, SIZE)
            features = self.net["tower"](pixel_values).flatten(2).transpose(1, 2)
            embeddings = embeddings + self.net["proj"](features).mean(dim=1, keepdim=True)
        logits: Tensor = self.net["head"](embeddings)
        return logits

    def loss_on_target(
        self, messages: list[dict[str, Any]], image: Tensor | None, target: str
    ) -> Tensor:
        prompt = self._ids(messages)
        appended = torch.tensor([[abs(hash(word)) % VOCAB for word in target.split()]])
        ids = torch.cat([prompt, appended], dim=1)
        span = target_span(prompt.shape[1], appended.shape[1])
        return token_ce(self._logits(ids, image), ids, span)

    def generate(self, messages: list[dict[str, Any]], image: Tensor | None, **kw: Any) -> str:
        with torch.no_grad():
            nxt = int(self._logits(self._ids(messages), image)[0, -1].argmax())
        return "yes" if nxt % 2 == 0 else "no"

    def describe(self) -> dict[str, Any]:
        return {"model": "_test_tiny", "model_dtype": "float32"}


@BENCHMARKS.register("_test_toy")
class ToyBenchmark(Benchmark):
    def items(self, split: str) -> list[Item]:
        items = [
            Item(id=f"i{i}", target="yes" if i % 2 else "no", ground_truth="yes", group=f"g{i % 6}")
            for i in range(24)
        ]
        return split_by_group(items, self.cfg.splits, seed=0)[split]

    def build_messages(
        self, item: Item, with_image: bool, site: str = "pre_query"
    ) -> list[dict[str, Any]]:
        messages = [{"role": "user", "content": [{"type": "text", "text": f"question {item.id}"}]}]
        return inject_image(messages, site) if with_image else messages

    def score(self, output: str, item: Item) -> float:
        return float(output.strip() == item.ground_truth)


def _config(tmp_path: Path, **train: Any) -> Config:
    return Config.model_validate(
        {
            "model": {
                "name": "_test_tiny",
                "model_id": "tiny",
                "dtype": "float32",
                "device": "cpu",
            },
            "benchmark": {
                "name": "_test_toy",
                "splits": {"train": 0.5, "val": 0.25, "test": 0.25},
            },
            "param": {"kind": "full", "height": 16, "width": 16, "init": "gray"},
            "train": {
                "epochs": 1,
                "batch_size": 2,
                "lr": 0.05,
                "eval_every": 2,
                "run_dir": str(tmp_path / "runs"),
                "run_name": "t",
                **train,
            },
            "eval": {"split": "test"},
            "results": {"path": str(tmp_path / "results.jsonl")},
        }
    )


@pytest.fixture
def trained(tmp_path: Path) -> tuple[Config, Path]:
    cfg = _config(tmp_path)
    Trainer(cfg).run()
    return cfg, tmp_path / "runs" / "t"


def test_training_writes_history_and_moves_the_image(trained: tuple[Config, Path]) -> None:
    _, run_dir = trained
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]

    assert [record["step"] for record in history] == [1, 2, 3, 4]
    assert all(record["grad_norm"] > 0 for record in history)
    assert history[-1]["linf_from_init"] > 0
    assert (run_dir / "latest.png").exists()


def test_checkpoint_selection_uses_the_val_metric_not_the_loss(
    trained: tuple[Config, Path],
) -> None:
    _, run_dir = trained
    state = json.loads((run_dir / "state.json").read_text())
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    evaluated = [record for record in history if "val_score" in record]

    assert evaluated, "no val evaluation ran"
    assert state["best_score"] == max(record["val_score"] for record in evaluated)
    assert state["val_clean_score"] is not None  # the clean number this run is measured against
    sidecar = json.loads((run_dir / "best.json").read_text())
    assert sidecar["best_step"] == state["best_step"]


def _overwriting(cfg: Config) -> Config:
    """Let a run replace the image an earlier one left in its directory.

    Args:
        cfg: A config whose train.overwrite is still the default False.

    Returns:
        A copy with train.overwrite set.
    """
    return cfg.model_copy(update={"train": cfg.train.model_copy(update={"overwrite": True})})


def test_rerunning_with_overwrite_starts_fresh(
    tmp_path: Path, trained: tuple[Config, Path]
) -> None:
    """An overwriting rerun replaces the history instead of appending to it.

    Args:
        tmp_path: Pytest temporary directory.
        trained: Config and run directory of a finished run.
    """
    cfg, run_dir = trained
    first = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    Trainer(_overwriting(cfg)).run()
    second = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]

    assert [record["step"] for record in second] == [record["step"] for record in first]
    assert second[0]["step"] == 1


def test_rerunning_without_overwrite_keeps_the_earlier_image(
    trained: tuple[Config, Path],
) -> None:
    """A rerun under a taken name is refused and leaves the image alone.

    Args:
        trained: Config and run directory of a finished run.
    """
    cfg, run_dir = trained
    saved = {name: (run_dir / name).read_bytes() for name in ("best.png", "history.jsonl")}

    with pytest.raises(FileExistsError, match="train.overwrite"):
        Trainer(cfg)

    assert {name: (run_dir / name).read_bytes() for name in saved} == saved


def test_a_run_directory_in_use_by_another_job_is_refused(tmp_path: Path) -> None:
    """A second job aimed at a running job's directory fails at once.

    Args:
        tmp_path: Pytest temporary directory.
    """
    cfg = _config(tmp_path)
    cfg.run_path.mkdir(parents=True)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            "handle = open(sys.argv[1], 'a')\n"
            "fcntl.lockf(handle, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "time.sleep(60)\n",
            str(cfg.run_path / RUN_LOCK),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"

        with pytest.raises(RuntimeError, match="in use by another training job"):
            Trainer(cfg)
    finally:
        holder.kill()
        holder.wait()

    Trainer(cfg).run()

    assert (cfg.run_path / "best.png").exists()


def _with_wandb(cfg: Config) -> Config:
    """Turn tracking on so Trainer.__init__ opens a wandb run.

    Args:
        cfg: A config whose wandb.enabled is still the default False.

    Returns:
        A copy with wandb.enabled set.
    """
    return cfg.model_copy(update={"wandb": cfg.wandb.model_copy(update={"enabled": True})})


def test_wandb_opens_a_fresh_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> MagicMock:
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr("skillpng.train.wandb.init", fake_init)
    Trainer(_with_wandb(_config(tmp_path, run_name="searchqa")))

    assert len(calls) == 1
    assert "id" not in calls[0]
    assert "resume" not in calls[0]
    assert calls[0]["name"] == "searchqa"


def test_the_step_budget_is_a_whole_number_of_passes_over_the_data(tmp_path: Path) -> None:
    cfg = _config(tmp_path, epochs=2, batch_size=2, run_name="e")
    n_train = len(ToyBenchmark(cfg.benchmark).items("train"))

    state = Trainer(cfg).run()
    assert state.step == 2 * -(-n_train // 2)  # ceil(n_train / batch_size) * epochs


def test_selecting_on_val_loss_skips_generation_and_picks_the_lowest(tmp_path: Path) -> None:
    cfg = _config(tmp_path, run_name="l", select_on="val_loss", eval_every=1)
    state = Trainer(cfg).run()

    history = [
        json.loads(line)
        for line in (tmp_path / "runs" / "l" / "history.jsonl").read_text().splitlines()
    ]
    evaluated = [record for record in history if "val_loss" in record]
    assert evaluated, "no val evaluation ran"
    assert all("val_score" not in record for record in history)  # no generation
    assert state.val_clean_loss is not None
    # Selection is higher-is-better everywhere, so the stored score is the negated best val loss.
    assert state.best_score == -min(record["val_loss"] for record in evaluated)


def test_a_run_shorter_than_eval_every_still_selects_on_the_val_split(tmp_path: Path) -> None:
    """`best.png` is never the last image by default: it is the one the val split scored best."""
    cfg = _config(tmp_path, run_name="short", eval_every=10)
    state = Trainer(cfg).run()

    run_dir = tmp_path / "runs" / "short"
    assert state.best_step == state.step and state.best_score is not None
    assert (run_dir / "best.png").exists()
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [record for record in history if "val_score" in record], "no val evaluation ran"


def test_training_refuses_to_run_without_a_held_out_split(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _config(tmp_path, eval_every=0)

    cfg = _config(tmp_path, run_name="noval")
    no_val = cfg.benchmark.model_copy(update={"splits": {"train": 0.5, "val": 0.0, "test": 0.5}})
    with pytest.raises(ValueError, match="held-out evaluation split"):
        Trainer(cfg.model_copy(update={"benchmark": no_val})).run()


def test_every_condition_produces_one_row_with_one_score(
    trained: tuple[Config, Path],
) -> None:
    cfg, run_dir = trained
    for condition in ("none", "noise", "skillpng"):
        update = {"condition": condition}
        if condition == "skillpng":
            update["image_path"] = str(run_dir / "best.png")
        run_eval(cfg.model_copy(update={"eval": cfg.eval.model_copy(update=update)}))

    rows = list(read_rows(cfg.results.path))
    assert [row["condition"] for row in rows] == ["none", "noise", "skillpng"]
    # One row is one condition: the clean baseline is its own run, not a
    # second pass repeated inside every image eval.
    assert all(row["score"] is not None and row["n_items"] > 0 for row in rows)
    clean, *imaged = rows
    assert not set(IMAGE_FIELDS) & set(clean)
    assert len({row["image_digest"] for row in imaged}) == 2
    assert all(row["resolution"] == [16, 16] for row in imaged)


def test_eval_limit_scores_the_prefix_of_the_split(trained: tuple[Config, Path]) -> None:
    """The debug knob: score a handful of items and say so in the row."""
    cfg, _ = trained

    def scored(**eval_update: Any) -> int:
        updated = cfg.eval.model_copy(update={"condition": "none", **eval_update})
        return run_eval(cfg.model_copy(update={"eval": updated})).n_items

    assert scored() == len(ToyBenchmark(cfg.benchmark).items("test"))
    assert scored(limit=2) == 2


def test_eval_writes_the_per_item_answers_next_to_the_score(
    tmp_path: Path, trained: tuple[Config, Path]
) -> None:
    """A small run has to be readable, not just averaged."""
    cfg, run_dir = trained
    update = {"condition": "skillpng", "image_path": str(run_dir / "best.png"), "limit": 3}
    row = run_eval(cfg.model_copy(update={"eval": cfg.eval.model_copy(update=update)}))

    path = run_dir / "eval" / "test-skillpng.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == row.n_items == 3
    assert row.score == sum(r["score"] for r in rows) / 3
    assert [record["item_id"] for record in rows] == [
        item.id for item in ToyBenchmark(cfg.benchmark).items("test")[:3]
    ]
    assert all(record["output"] for record in rows)
    assert all(record["target"] in ("yes", "no") for record in rows)


def test_each_condition_writes_its_own_per_item_file(
    tmp_path: Path, trained: tuple[Config, Path]
) -> None:
    """The image's effect per item is a join on item_id, not a paired line."""
    cfg, run_dir = trained
    for condition, image_path in (("none", None), ("skillpng", str(run_dir / "best.png"))):
        update = {"condition": condition, "image_path": image_path, "limit": 3}
        run_eval(cfg.model_copy(update={"eval": cfg.eval.model_copy(update=update)}))

    by_condition = {
        condition: {
            json.loads(line)["item_id"]: json.loads(line)
            for line in (run_dir / "eval" / f"test-{condition}.jsonl").read_text().splitlines()
        }
        for condition in ("none", "skillpng")
    }
    assert by_condition["none"].keys() == by_condition["skillpng"].keys()
    assert all(record["output"] for record in by_condition["skillpng"].values())


def test_the_answers_land_while_the_eval_runs_not_only_at_the_end(
    tmp_path: Path, trained: tuple[Config, Path]
) -> None:
    """The point of the file is following a long run, so it must not buffer."""
    cfg, run_dir = trained
    path = run_dir / "eval" / "test-none.jsonl"
    seen: list[int] = []

    class Watched(ToyBenchmark):
        """Records how much of the log is already on disk at each rollout."""

        def score(self, output: str, item: Item) -> float:
            seen.append(len(path.read_text().splitlines()) if path.exists() else 0)
            return super().score(output, item)

    update = {"condition": "none", "limit": 4}
    run_eval(
        cfg.model_copy(update={"eval": cfg.eval.model_copy(update=update)}),
        benchmark=Watched(cfg.benchmark),
    )

    # One line already written per item scored before this one: the file grew
    # during the eval rather than appearing whole once it finished.
    assert seen == [0, 1, 2, 3]
    assert len(path.read_text().splitlines()) == 4


def test_a_partial_log_survives_an_eval_that_dies_midway(
    tmp_path: Path, trained: tuple[Config, Path]
) -> None:
    """A crash at item 3 should still leave the answers for items 1 and 2."""
    cfg, run_dir = trained

    scored: list[str] = []

    class Exploding(ToyBenchmark):
        """Scores two items, then fails the way a real backend might."""

        def score(self, output: str, item: Item) -> float:
            if len(scored) == 2:
                raise RuntimeError("backend fell over")
            scored.append(item.id)
            return super().score(output, item)

    with pytest.raises(RuntimeError, match="fell over"):
        run_eval(
            cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"condition": "none"})}),
            benchmark=Exploding(cfg.benchmark),
        )

    path = run_dir / "eval" / "test-none.jsonl"
    assert len(path.read_text().splitlines()) == 2


def test_a_limit_of_zero_is_refused_rather_than_read_as_no_limit(tmp_path: Path) -> None:
    """`--set eval.limit=0` used to be falsy, so a typo silently ran the whole split."""
    for override in ({"eval": {"limit": 0}}, {"benchmark": {"limit": -1}}):
        with pytest.raises(ValidationError):
            Config.model_validate(override)
