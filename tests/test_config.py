"""`_base_` inheritance, dotted overrides, and validation at parse time."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from skillpng.config import Config, apply_override, load_config


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_base_inheritance_merges_deeply(tmp_path: Path) -> None:
    _write(tmp_path, "base.yaml", "seed: 1\ntrain:\n  lr: 0.01\n  epochs: 3\n")
    child = _write(tmp_path, "child.yaml", "_base_: base.yaml\ntrain:\n  lr: 0.5\n")

    cfg = load_config(child)
    assert cfg.train.lr == 0.5
    assert cfg.train.epochs == 3  # inherited, not clobbered by the partial override
    assert cfg.seed == 1


def test_cli_overrides_beat_both(tmp_path: Path) -> None:
    _write(tmp_path, "base.yaml", "train:\n  lr: 0.01\n")
    child = _write(tmp_path, "child.yaml", "_base_: base.yaml\ntrain:\n  lr: 0.5\n")

    cfg = load_config(child, ["train.lr=0.05", "model.dtype=float32", "train.grad_clip=null"])
    assert cfg.train.lr == 0.05
    assert cfg.model.dtype == "float32"
    assert cfg.train.grad_clip is None


def test_override_values_are_typed_not_strings() -> None:
    data = apply_override({}, "benchmark.splits={train: 0.5, val: 0.5}")
    assert data["benchmark"]["splits"] == {"train": 0.5, "val": 0.5}
    assert apply_override({}, "train.epochs=10")["train"]["epochs"] == 10
    assert apply_override({}, "wandb.enabled=false")["wandb"]["enabled"] is False


def test_circular_base_is_caught(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", "_base_: b.yaml\n")
    b = _write(tmp_path, "b.yaml", "_base_: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(b)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"train": {"learning_rate": 0.1}})


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
# base.yaml is the inherited defaults, not a runnable config: it names no
# benchmark on purpose, so only the experiment configs are checked here.
EXPERIMENT_CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if p.name != "base.yaml")


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_parse(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.benchmark.name, f"{path.name} names no benchmark"
    assert cfg.train.eval_every > 0


def test_a_config_without_a_base_still_gets_every_default() -> None:
    """The defaults live in the schema, not in a configs/base.yaml. An experiment config that
    sets one key must still come out fully populated."""
    cfg = Config.model_validate({"benchmark": {"name": "searchqa"}})
    assert cfg.train.lr == Config().train.lr
    assert cfg.model.generate == {"do_sample": False}  # deterministic decoding by default
    # The default canvas matches the default pinned resolution, so nothing resamples silently.
    assert cfg.model.resolution == (cfg.param.height, cfg.param.width)


def test_config_hash_tracks_content() -> None:
    first = Config()
    second = Config.model_validate({"train": {"lr": 0.02}})
    assert first.hash() == Config().hash()
    assert first.hash() != second.hash()


def test_gpu_count_is_placement_not_content() -> None:
    """Splitting the layers over more GPUs optimizes the same thing."""
    split = Config.model_validate({"model": {"num_gpus": 2}})
    assert split.model.num_gpus == 2
    assert split.hash() == Config().hash()


def test_overwrite_is_permission_not_content() -> None:
    """Letting a run replace an earlier image optimizes the same thing."""
    overwriting = Config.model_validate({"train": {"overwrite": True}})
    assert overwriting.hash() == Config().hash()


def test_zero_gpus_is_refused() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"model": {"num_gpus": 0}})


def test_run_path_hangs_off_run_dir() -> None:
    cfg = Config.model_validate(
        {"train": {"run_dir": "runs/searchqa/qwen3.5-9B", "run_name": "lr0.03"}}
    )
    assert cfg.run_path == Path("runs/searchqa/qwen3.5-9B/lr0.03")


def test_results_default_next_to_the_runs_they_came_from() -> None:
    """Unset, results.path follows run_dir, so one benchmark and model keep their own file."""
    cfg = Config.model_validate({"train": {"run_dir": "runs/searchqa/qwen3.5-9B"}})
    assert cfg.results_path == Path("runs/searchqa/qwen3.5-9B/results.jsonl")


def test_an_explicit_results_path_still_wins() -> None:
    cfg = Config.model_validate(
        {"train": {"run_dir": "runs/searchqa/qwen3.5-9B"}, "results": {"path": "runs/all.jsonl"}}
    )
    assert cfg.results_path == Path("runs/all.jsonl")


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_scope_their_run_dir(path: Path) -> None:
    """Every experiment writes under its own benchmark directory, not into runs/ directly."""
    cfg = load_config(path)
    assert Path(cfg.train.run_dir) != Path("runs")
