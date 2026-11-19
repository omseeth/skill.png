"""`skillpng` — probe, train, eval.

skillpng probe -c configs/example.yaml
skillpng train -c configs/example.yaml --set train.lr=0.05
skillpng eval  -c configs/example.yaml --set eval.condition=noise
"""

from __future__ import annotations

import json
import logging

import typer

from .config import Config, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

CONFIG_OPT = typer.Option(None, "--config", "-c", help="YAML config path")
SET_OPT = typer.Option(
    None, "--set", "-s", help="Override a config key: --set train.lr=0.05 (repeatable)"
)


def _setup(config: str | None, overrides: list[str] | None, verbose: bool = True) -> Config:
    """Configure logging for the process and load the config the command runs on.

    Args:
        config: Path to a YAML config file, or None for the built-in defaults.
        overrides: Repeated key=value strings applied on top of the file.
        verbose: Log at INFO instead of WARNING.

    Returns:
        The merged configuration.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return load_config(config, overrides)


def _echo(payload: object) -> None:
    """Write one command result to stdout as sorted, indented JSON.

    Args:
        payload: Any JSON-serializable result; anything else is stringified.
    """
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.command()
def probe(
    config: str | None = CONFIG_OPT,
    set_: list[str] | None = SET_OPT,
    grad: bool = typer.Option(
        False, "--grad", help="Backward a scalar and report whether the image got gradient"
    ),
) -> None:
    """Print the preprocessing spec derived from the loaded processor. Needs real weights.

    Builds the model, reports the resolution it feeds, how many image tokens
    that costs, and optionally whether gradient reaches the image.

    Args:
        config: Path to a YAML config file, or None for the built-in defaults.
        set_: Repeated key=value overrides applied on top of the file.
        grad: Also run one backward pass and report the gradient check.
    """
    cfg = _setup(config, set_)
    from .models.registry import build_model

    model = build_model(cfg.model)
    spec = model.preprocess
    size = getattr(model, "size", None)
    payload: dict[str, object] = {
        "model_id": cfg.model.model_id,
        "fed_resolution": list(size) if size else None,
        "image_token_count": spec.token_count(*size) if size else None,
        "spec": spec.describe(),
    }
    if grad:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}],
            }
        ]
        payload["grad_check"] = model.check_grad_route(messages)  # type: ignore[attr-defined]
    _echo(payload)


@app.command()
def train(config: str | None = CONFIG_OPT, set_: list[str] | None = SET_OPT) -> None:
    """Optimize one universal image.

    Runs the training loop and prints where it stopped, its best checkpoint,
    and the run directory holding the image.

    Args:
        config: Path to a YAML config file, or None for the built-in defaults.
        set_: Repeated key=value overrides applied on top of the file.
    """
    cfg = _setup(config, set_)
    from .train import train as run_train

    state = run_train(cfg)
    _echo(
        {
            "steps": state.step,
            "best_step": state.best_step,
            "best_score": state.best_score,
            "val_clean_score": state.val_clean_score,
            "run_dir": str(cfg.run_path),
        }
    )


@app.command("eval")
def eval_(config: str | None = CONFIG_OPT, set_: list[str] | None = SET_OPT) -> None:
    """Score one condition on one split and append one result row.

    Prints the appended row. The no-image baseline is its own run
    (eval.condition=none), scored once per benchmark, model and split; the
    delta against it is computed by analysis/analyze_runs.py.

    Args:
        config: Path to a YAML config file, or None for the built-in defaults.
        set_: Repeated key=value overrides applied on top of the file.
    """
    cfg = _setup(config, set_)
    from .evaluate import run_eval

    _echo(run_eval(cfg).model_dump(mode="json"))


@app.command()
def registries(config: str | None = CONFIG_OPT, set_: list[str] | None = SET_OPT) -> None:
    """List registered models and benchmarks (import your modules first).

    Imports the default model wrapper if it is installed, then prints every
    name the two registries know about.

    Args:
        config: Path to a YAML config file, or None for the built-in defaults.
        set_: Repeated key=value overrides applied on top of the file.
    """
    _setup(config, set_, verbose=False)
    from .benchmarks.registry import BENCHMARKS
    from .models.registry import MODELS

    try:
        from .models import hf, vllm  # noqa: F401  (register the wrappers)
    except ImportError as error:  # transformers or httpx missing
        typer.echo(f"# could not import the model wrappers: {error}", err=True)

    _echo({"models": MODELS.names(), "benchmarks": BENCHMARKS.names()})


if __name__ == "__main__":
    app()
