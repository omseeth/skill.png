"""A checkpoint that never declared sdpa support still runs the kernel.

transformers reads sdpa support off a class attribute that defaults to
False, so Nemotron 3 Nano Omni's own modeling file, which never sets it, was
refused the kernel at load however it was asked for and fell back to eager.
Eager keeps a whole attention matrix per layer for the backward, which is
what a 5k token sequence ran out of memory on. The flag is now repaired
after the load for the classes whose attention reads its implementation off
the config, and the kernel is counted at the first call rather than assumed.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from skillpng.config import ModelCfg
from skillpng.models import compat, hf


def _tiny(monkeypatch: pytest.MonkeyPatch, supports_sdpa: bool) -> LlamaModel:
    """A two-layer Llama on eager, standing in for a checkpoint of any size.

    Llama is what the remote file is shaped like: an attention module that
    looks its implementation up per forward. The support flag is taken away
    to put it in the position the Nemotron classes are in.

    Args:
        monkeypatch: The pytest patcher, which puts the flag back.
        supports_sdpa: What the class declares before the promotion runs.

    Returns:
        The model, loaded on eager.
    """
    monkeypatch.setattr(LlamaModel, "_supports_sdpa", supports_sdpa, raising=False)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        attn_implementation="eager",
    )

    return LlamaModel(config)


def _forward(model: LlamaModel) -> None:
    """Run one forward over a short sequence.

    Args:
        model: The model to call.
    """
    model(input_ids=torch.zeros(1, 8, dtype=torch.long))


def test_a_named_sdpa_falls_back_to_eager_at_load() -> None:
    assert hf._attn_attempts("sdpa") == ("sdpa", "eager")


def test_any_other_named_implementation_is_tried_alone() -> None:
    assert hf._attn_attempts("eager") == ("eager",)


def test_an_unnamed_implementation_walks_the_fallbacks() -> None:
    assert hf._attn_attempts(None) == hf._ATTN_FALLBACKS


def test_the_promotion_moves_a_model_the_load_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(monkeypatch, supports_sdpa=False)

    landed = compat.promote_to_sdpa(model)

    assert model.config._attn_implementation == "sdpa"
    assert landed == {"": "sdpa"}


def test_the_promoted_model_reaches_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(monkeypatch, supports_sdpa=False)
    compat.promote_to_sdpa(model)

    with compat.require_sdpa("the promoted forward"):
        _forward(model)


def test_the_same_model_on_eager_reaches_no_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(monkeypatch, supports_sdpa=False)

    with pytest.raises(RuntimeError, match="reached no sdpa kernel"), compat.require_sdpa("eager"):
        _forward(model)


def test_a_family_that_cannot_be_switched_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(monkeypatch, supports_sdpa=False)
    # What transformers answers for a file whose attention never asks its
    # config which kernel to run.
    monkeypatch.setattr(LlamaModel, "_can_set_attn_implementation", classmethod(lambda cls: False))

    with pytest.raises(RuntimeError, match="nothing on this checkpoint took it"):
        compat.promote_to_sdpa(model)


def test_the_kernel_is_put_back_after_a_call_that_raised() -> None:
    original = torch.nn.functional.scaled_dot_product_attention

    with pytest.raises(ZeroDivisionError), compat.require_sdpa("a call that raises"):
        raise ZeroDivisionError

    assert torch.nn.functional.scaled_dot_product_attention is original


def test_the_kernel_is_put_back_after_a_call_that_reached_none() -> None:
    original = torch.nn.functional.scaled_dot_product_attention

    with pytest.raises(RuntimeError), compat.require_sdpa("a call that reaches nothing"):
        pass

    assert torch.nn.functional.scaled_dot_product_attention is original


def test_a_model_already_on_sdpa_is_not_promoted(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(monkeypatch, supports_sdpa=True)
    model.set_attn_implementation("sdpa")

    assert not hf._promote_sdpa(_cfg("sdpa"), model)


def test_a_run_that_did_not_ask_for_sdpa_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(monkeypatch, supports_sdpa=False)

    assert not hf._promote_sdpa(_cfg(None), model)
    assert model.config._attn_implementation == "eager"


def _cfg(attn: str | None) -> ModelCfg:
    """A model configuration naming one attention implementation.

    Args:
        attn: What the caller asked for, or None for the checkpoint's own.

    Returns:
        The configuration.
    """
    return ModelCfg(model_id="fake", attn_implementation=attn)
