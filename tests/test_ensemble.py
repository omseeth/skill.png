"""One image optimized against several models at once, without weights.

The fakes extend the ones in test_loop: two models that differ in their
weights and in the resolution they resize the canvas to, the way Qwen and
Gemma do, and that record every training call they get.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from skillpng.benchmarks.base import Item
from skillpng.config import Config, ModelCfg, load_config
from skillpng.image.preprocess import PreprocessSpec, preprocess_image
from skillpng.models.registry import MODELS
from skillpng.train import Trainer, model_labels
from test_loop import TinyVLM, _config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
# Every training call of every SizedVLM: model id, messages, image, target.
CALLS: list[tuple[str, list[dict[str, Any]], Tensor, str]] = []


@MODELS.register("_test_sized")
class SizedVLM(TinyVLM):
    """TinyVLM with weights of its own, fed at its model.resolution."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.size = (cfg.resolution[0], cfg.resolution[1])
        self._spec = PreprocessSpec(
            image_mean=(0.5,) * 3, image_std=(0.5,) * 3, height=self.size[0], width=self.size[1]
        )
        # Seeded by the id, so two members never share weights.
        generator = torch.Generator().manual_seed(zlib.crc32(cfg.model_id.encode()))

        with torch.no_grad():
            for parameter in self.net.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)

    def _logits(self, ids: Tensor, image: Tensor | None) -> Tensor:
        embeddings = self.net["embed"](ids)
        if image is not None:
            pixel_values = preprocess_image(image, self._spec, self.size)
            features = self.net["tower"](pixel_values).flatten(2).transpose(1, 2)
            embeddings = embeddings + self.net["proj"](features).mean(dim=1, keepdim=True)
        logits: Tensor = self.net["head"](embeddings)
        return logits

    def loss_on_target(
        self, messages: list[dict[str, Any]], image: Tensor | None, target: str
    ) -> Tensor:
        if image is not None and torch.is_grad_enabled():
            CALLS.append((self.cfg.model_id, messages, image.detach().clone(), target))
        return super().loss_on_target(messages, image, target)


@pytest.fixture(autouse=True)
def _fresh_calls() -> None:
    """Start every test with an empty call log."""
    CALLS.clear()


def _ensemble(
    tmp_path: Path,
    members: list[dict[str, Any]] | None = None,
    ensemble: dict[str, Any] | None = None,
    **train: Any,
) -> Config:
    """Build a two-model config: tiny-a at 16 px plus tiny-b at 32 px.

    Args:
        tmp_path: Directory the runs are written under.
        members: Member blocks, tiny-b at 32 px when None.
        ensemble: Further ensemble keys, e.g. normalize_grads.
        **train: Train keys passed through to test_loop's _config.

    Returns:
        The validated config, the primary on the 16 px canvas.
    """
    data = _config(tmp_path, **train).model_dump()
    data["model"].update(name="_test_sized", model_id="tiny-a", resolution=[16, 16])
    if members is None:
        members = [{"model": {"model_id": "tiny-b", "resolution": [32, 32]}}]
    data["ensemble"] = {"members": members, **(ensemble or {})}
    return Config.model_validate(data)


def _alone(cfg: Config, index: int, run_name: str) -> Config:
    """Turn one model of an ensemble config into a single-model config.

    Args:
        cfg: The ensemble config.
        index: Position of the model, 0 for the primary.
        run_name: Run directory name of the single-model run.

    Returns:
        The config with only that model, everything else unchanged.
    """
    model = (cfg.model, *cfg.member_models)[index]
    data = cfg.model_dump()
    data["model"] = model.model_dump()
    data["ensemble"] = {}
    data["train"]["run_name"] = run_name
    return Config.model_validate(data)


def _first_step(trainer: Trainer) -> tuple[float, Tensor, dict[str, Any]]:
    """Take the first optimizer step and capture the gradient it follows.

    Args:
        trainer: A freshly built trainer.

    Returns:
        The step's loss, the image gradient handed to Adam, and the extras.
    """
    seen: list[Tensor] = []
    step = trainer.optimizer.step

    def capture(*args: Any, **kwargs: Any) -> Any:
        """Copy the gradient, then step as usual.

        Args:
            *args: Positional arguments of Adam.step.
            **kwargs: Keyword arguments of Adam.step.

        Returns:
            Whatever Adam.step returns.
        """
        grad = trainer.param.leaves()[0].grad
        assert grad is not None
        seen.append(grad.detach().clone())
        return step(*args, **kwargs)

    trainer.optimizer.step = capture  # type: ignore[method-assign]
    batch, _ = trainer._next_batch(trainer.benchmark.items("train"), [])
    loss, _, extras = trainer._step(batch)
    return loss, seen[0], extras


def _reference_step(trainer: Trainer, batch: list[Item]) -> tuple[float, Tensor]:
    """The single-model step as train.py took it before ensembles existed.

    Args:
        trainer: A freshly built single-model trainer.
        batch: Items of the step.

    Returns:
        The mean loss and the gradient the optimizer would follow.
    """
    cfg = trainer.cfg.train
    trainer.optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for item in batch:
        messages = trainer.benchmark.build_messages(
            item, with_image=True, site=trainer.cfg.injection.site
        )
        loss = trainer.model.loss_on_target(messages, trainer.param.tensor(), item.target)
        (loss / len(batch)).backward()
        total += float(loss.detach())
    leaves = trainer.param.leaves()
    if cfg.grad_clip:
        torch.nn.utils.clip_grad_norm_(leaves, cfg.grad_clip)
    grad = leaves[0].grad
    assert grad is not None
    return total / len(batch), grad.detach().clone()


def _history(run_dir: Path) -> list[dict[str, Any]]:
    """Read a run's history.

    Args:
        run_dir: The run directory.

    Returns:
        One record per logged line.
    """
    return [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]


def test_a_single_model_steps_as_it_did_before_ensembles(tmp_path: Path) -> None:
    """Same loss and bit-identical gradient, gradient clipping on."""
    train = {"grad_clip": 0.05, "batch_size": 3}
    ours = Trainer(_config(tmp_path, run_name="ours", **train))
    reference = Trainer(_config(tmp_path, run_name="reference", **train))

    loss, grad, extras = _first_step(ours)
    batch, _ = reference._next_batch(reference.benchmark.items("train"), [])
    expected_loss, expected_grad = _reference_step(reference, batch)

    assert loss == expected_loss
    assert torch.equal(grad, expected_grad)
    assert not any("/" in key for key in extras)


def test_a_single_model_run_logs_the_keys_it_always_logged(tmp_path: Path) -> None:
    Trainer(_config(tmp_path, run_name="plain")).run()
    run_dir = tmp_path / "runs" / "plain"

    keys = {key for record in _history(run_dir) for key in record}
    assert keys == {
        "step", "loss", "grad_norm", "lr", "seconds", "batch", "image_token_count",
        "linf_from_init", "l2_from_init", "n_val", "select_on", "val_score",
        "val_clean_score", "metric_name", "new_best",
    }  # fmt: skip
    sidecar = json.loads((run_dir / "best.json").read_text())
    assert "models" not in sidecar and "best_scores" not in sidecar


@pytest.mark.parametrize("normalize", [False, True])
def test_the_step_follows_the_weighted_sum_of_the_models_gradients(
    tmp_path: Path, normalize: bool
) -> None:
    """Checked against each model trained alone on the same batch."""
    members = [{"model": {"model_id": "tiny-b", "resolution": [32, 32]}, "weight": 3.0}]
    cfg = _ensemble(tmp_path, members, {"normalize_grads": normalize}, run_name="ens")
    assert cfg.loss_weights == (0.25, 0.75)

    loss, grad, extras = _first_step(Trainer(cfg))
    loss_a, grad_a, _ = _first_step(Trainer(_alone(cfg, 0, "a")))
    loss_b, grad_b, _ = _first_step(Trainer(_alone(cfg, 1, "b")))

    norm_a, norm_b = float(grad_a.norm()), float(grad_b.norm())
    assert extras["loss/tiny-a"] == loss_a and extras["loss/tiny-b"] == loss_b
    assert extras["grad_norm/tiny-a"] == pytest.approx(norm_a, rel=1e-6)
    assert extras["grad_norm/tiny-b"] == pytest.approx(norm_b, rel=1e-6)
    assert loss == pytest.approx(0.25 * loss_a + 0.75 * loss_b, rel=1e-12)

    if normalize:
        unit_a, unit_b = grad_a / norm_a, grad_b / norm_b
        assert float(unit_a.norm()) == pytest.approx(1.0)
        assert float(unit_b.norm()) == pytest.approx(1.0)
        expected = 0.25 * unit_a + 0.75 * unit_b
    else:
        expected = 0.25 * grad_a + 0.75 * grad_b
    # float32 rounding: the step folds weight and norm into one factor.
    torch.testing.assert_close(grad, expected, rtol=1e-5, atol=1e-7)


def test_unit_norm_gradients_keep_a_large_loss_from_dominating(tmp_path: Path) -> None:
    """Rescaling one model's loss moves the plain step, not the normalized one."""

    def step(normalize: bool, factor: float, run_name: str) -> Tensor:
        """Step once with tiny-b's loss scaled by a factor.

        Args:
            normalize: Whether to rescale each gradient to unit norm.
            factor: Factor on tiny-b's loss.
            run_name: Run directory name.

        Returns:
            The gradient the step followed.
        """
        trainer = Trainer(_ensemble(tmp_path, None, {"normalize_grads": normalize},
                                    run_name=run_name))
        member = trainer.models[1]
        loss_on_target = member.loss_on_target
        member.loss_on_target = (  # type: ignore[method-assign]
            lambda *args: factor * loss_on_target(*args)
        )
        return _first_step(trainer)[1]

    torch.testing.assert_close(step(True, 1.0, "n1"), step(True, 100.0, "n100"))
    assert not torch.allclose(step(False, 1.0, "p1"), step(False, 100.0, "p100"))


def test_every_model_sees_the_same_image_for_each_item(tmp_path: Path) -> None:
    """Each model takes the whole batch in turn, on the one image."""
    cfg = _ensemble(tmp_path, run_name="views", batch_size=3)
    cfg = cfg.model_copy(update={"param": cfg.param.model_copy(update={"init": "random"})})
    _first_step(Trainer(cfg))

    # Models in the outer loop, items in the inner one.
    assert [model_id for model_id, *_ in CALLS] == ["tiny-a"] * 3 + ["tiny-b"] * 3
    seen_a, seen_b = CALLS[:3], CALLS[3:]
    for (_, messages_a, image_a, target_a), (_, messages_b, image_b, target_b) in zip(
        seen_a, seen_b, strict=True
    ):
        assert messages_a == messages_b and target_a == target_b
        assert torch.equal(image_a, image_b)


def test_a_member_inherits_the_primary_block_except_what_it_sets() -> None:
    """Not the ModelCfg defaults: max_new_tokens stays at the config's 16384."""
    cfg = load_config(
        CONFIGS / "livemathematicianbench.yaml",
        [
            "model.model_id=/p/qwen3_5-4B",
            "model.chat_template_kwargs.enable_thinking=false",
            "model.generate.temperature=0.7",
            "ensemble.members=[{model: {model_id: /p/gemma-4-E4B-it, "
            "resolution: [768, 768], generate: {top_p: 0.9}}}]",
        ],
    )
    (member,) = cfg.member_models

    assert member.max_new_tokens == cfg.model.max_new_tokens == 16384
    assert ModelCfg().max_new_tokens != 16384
    assert member.chat_template_kwargs == {"enable_thinking": False}
    # Nested blocks merge key by key, as --set and _base_ do.
    assert member.generate == {"do_sample": False, "temperature": 0.7, "top_p": 0.9}
    overridden = {"model_id", "resolution", "generate"}
    primary = cfg.model.model_dump()
    assert {k: v for k, v in member.model_dump().items() if k not in overridden} == {
        k: v for k, v in primary.items() if k not in overridden
    }
    assert member.model_id == "/p/gemma-4-E4B-it"
    assert member.resolution == (768, 768)


def test_ensemble_settings_are_checked_when_the_config_loads() -> None:
    with pytest.raises(ValidationError, match="unknown model keys"):
        Config.model_validate({"ensemble": {"members": [{"model": {"model_idd": "x"}}]}})
    with pytest.raises(ValidationError):
        Config.model_validate({"ensemble": {"members": [{"model": {"dtype": "int4"}}]}})
    with pytest.raises(ValidationError, match=r"need ensemble\.members"):
        Config.model_validate({"ensemble": {"normalize_grads": True}})
    with pytest.raises(ValidationError):
        Config.model_validate({"ensemble": {"members": [{"weight": 0}]}})


def test_loss_weights_are_equal_by_default_and_sum_to_one() -> None:
    assert Config().loss_weights == (1.0,)
    two = Config.model_validate({"ensemble": {"members": [{}]}})
    assert two.loss_weights == (0.5, 0.5)
    three = Config.model_validate(
        {"ensemble": {"primary_weight": 2, "members": [{"weight": 1}, {"weight": 1}]}}
    )
    assert three.loss_weights == (0.5, 0.25, 0.25)


@pytest.mark.parametrize("path", [None, CONFIGS / "livemathematicianbench.yaml"])
def test_the_hash_of_a_config_without_an_ensemble_is_unchanged(path: Path | None) -> None:
    """The pre-ensemble formula: no ensemble key in the payload at all."""
    cfg = load_config(path, ["model.num_gpus=2"])
    excluded: dict[str, Any] = {
        "wandb": True,
        "model": {"num_gpus"},
        "ensemble": True,
        "train": {"overwrite"},
    }
    payload = json.dumps(cfg.model_dump(mode="json", exclude=excluded), sort_keys=True)
    assert cfg.hash() == hashlib.sha256(payload.encode()).hexdigest()[:16]


def test_member_placement_stays_out_of_the_hash() -> None:
    def config(**member: Any) -> Config:
        """Build a one-member config.

        Args:
            **member: The member's model keys.

        Returns:
            The validated config.
        """
        return Config.model_validate({"ensemble": {"members": [{"model": member}]}})

    base = config(model_id="b")
    assert base.hash() != Config().hash()
    assert config(model_id="b", device="cuda:1", num_gpus=2).hash() == base.hash()
    assert config(model_id="b", resolution=[768, 768]).hash() != base.hash()


@pytest.mark.parametrize("select_on", ["val_metric", "val_loss"])
def test_selection_uses_the_weighted_mean_of_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, select_on: str
) -> None:
    """tiny-a alone would keep step 2; the mean over both picks step 4."""
    # The clean baseline first, then one value per eval, at steps 2 and 4.
    scripted = {"tiny-a": [0.2, 0.9, 0.6], "tiny-b": [0.2, 0.0, 0.6]}
    calls: dict[str, int] = {"tiny-a": 0, "tiny-b": 0}

    def value(model: Any) -> float:
        """Hand out the model's next scripted value.

        Args:
            model: The model being scored.

        Returns:
            Its next value.
        """
        name = model.cfg.model_id
        calls[name] += 1
        return scripted[name][calls[name] - 1]

    monkeypatch.setattr(
        "skillpng.train.evaluate_split", lambda model, *args, **kwargs: (value(model), [])
    )
    # Lower is better for a loss, so the scripted scores are negated.
    monkeypatch.setattr(Trainer, "_val_loss", lambda self, model, items, image: -value(model))
    cfg = _ensemble(tmp_path, run_name="select", select_on=select_on, eval_every=2)
    trainer = Trainer(cfg)
    state = trainer.run()

    sign = 1 if select_on == "val_metric" else -1
    name, names = ("score", "scores") if select_on == "val_metric" else ("loss", "losses")
    assert state.best_step == 4
    assert state.best_score == pytest.approx(0.6)
    assert state.best_scores == pytest.approx({"tiny-a": 0.6, "tiny-b": 0.6})
    clean = state.val_clean_score if select_on == "val_metric" else state.val_clean_loss
    assert clean == pytest.approx(sign * 0.2)
    evaluated = [r for r in _history(cfg.run_path) if f"val_{name}" in r]
    assert [r["step"] for r in evaluated] == [2, 4]
    for record in evaluated:
        mean = 0.5 * record[f"val_{name}/tiny-a"] + 0.5 * record[f"val_{name}/tiny-b"]
        assert record[f"val_{name}"] == pytest.approx(mean)
        assert record[f"val_clean_{name}/tiny-b"] == pytest.approx(sign * 0.2)

    sidecar = json.loads((cfg.run_path / "best.json").read_text())
    assert sidecar["model"] == "tiny-a"
    assert sidecar["best_score"] == state.best_score
    assert sidecar["best_scores"] == state.best_scores
    clean_by_model = {"tiny-a": sign * 0.2, "tiny-b": sign * 0.2}
    assert sidecar[f"val_clean_{names}"] == pytest.approx(clean_by_model)
    assert sidecar["models"] == [
        {"label": "tiny-a", "model_id": "tiny-a", "resolution": [16, 16], "weight": 0.5},
        {"label": "tiny-b", "model_id": "tiny-b", "resolution": [32, 32], "weight": 0.5},
    ]
    assert sidecar["normalize_grads"] is False


def test_each_model_is_logged_to_history_and_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = MagicMock()
    # Where the run would keep its files, which are pruned after finish.
    run.dir = str(tmp_path / "wandb" / "run" / "files")
    monkeypatch.setattr("skillpng.train.wandb.init", lambda **kwargs: run)
    cfg = _ensemble(tmp_path, run_name="logged")
    cfg = cfg.model_copy(update={"wandb": cfg.wandb.model_copy(update={"enabled": True})})
    Trainer(cfg).run()

    steps = [r for r in _history(cfg.run_path) if "loss" in r]
    per_model = {"loss/tiny-a", "loss/tiny-b", "grad_norm/tiny-a", "grad_norm/tiny-b",
                 "seconds/tiny-a", "seconds/tiny-b", "image_token_count/tiny-b"}
    assert all(per_model <= set(record) for record in steps)
    for record in steps:
        mean = 0.5 * record["loss/tiny-a"] + 0.5 * record["loss/tiny-b"]
        assert record["loss"] == pytest.approx(mean)

    tracked = {key for call in run.log.call_args_list for key in call.args[0]}
    assert per_model | {"loss", "grad_norm", "val_score/tiny-a", "val_score/tiny-b"} <= tracked


def test_labels_are_basenames_made_unique() -> None:
    ids = ["/p/qwen3_5-4B", "/p/gemma-4-E4B-it/", "google/gemma-4-E4B-it", ""]
    assert model_labels(ids) == ["qwen3_5-4B", "gemma-4-E4B-it", "gemma-4-E4B-it-2", "model"]
