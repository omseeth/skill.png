"""A checkpoint that asks for a kernel this environment has no wheel for still loads.

The nemotron omni checkpoints name flash_attention_2 in their own config, and
flash-attn is not a dependency here, so the load raised before a single step ran.
The auto class loop now walks the kernels torch always has before giving up, and
only when the failure was about the kernel rather than about the checkpoint.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from skillpng.config import ModelCfg
from skillpng.models import hf


class FakeAutoClass:
    """An auto class that records its calls and fails on a script.

    Args:
        errors: Exception per attn_implementation value, None where it loads.

    Returns:
        None.
    """

    def __init__(self, errors: dict[str | None, Exception | None]) -> None:
        self.errors = errors
        self.calls: list[str | None] = []

    def from_pretrained(self, model_id: str, **kwargs: object) -> str:
        """Load, or raise whatever the script holds for this kernel.

        Args:
            model_id: The checkpoint path, ignored.
            kwargs: Load kwargs; only attn_implementation is read.

        Returns:
            A stand-in for the loaded model.
        """
        value = kwargs.get("attn_implementation")
        attn = value if isinstance(value, str) else None
        self.calls.append(attn)
        error = self.errors.get(attn)

        if error is not None:
            raise error

        return f"model:{attn}"


def _patch(monkeypatch: pytest.MonkeyPatch, **classes: object) -> None:
    """Stand in for transformers with a module holding only the named auto classes.

    Args:
        monkeypatch: The pytest patcher.
        classes: Auto class name to stand-in object.

    Returns:
        None.
    """
    module = types.ModuleType("transformers")

    for name, value in classes.items():
        setattr(module, name, value)

    monkeypatch.setitem(sys.modules, "transformers", module)


_FLASH = ImportError(
    "FlashAttention2 has been toggled on, but it cannot be used due to the "
    "following error: the package for FlashAttention2 doesn't seem to be installed."
)
_WRONG_CLASS = ValueError(
    "Unrecognized configuration class NemotronH_Nano_Omni_Reasoning_V3_Config "
    "for this kind of AutoModel: AutoModelForImageTextToText."
)


def test_falls_back_when_the_checkpoints_kernel_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    causal = FakeAutoClass({None: _FLASH, "sdpa": None})
    _patch(monkeypatch, AutoModelForCausalLM=causal)

    model = hf._load_model(ModelCfg(model_id="fake"), torch.float32)

    assert model == "model:sdpa"
    assert causal.calls == [None, "sdpa"]


def test_falls_back_to_eager_when_sdpa_is_refused_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = ValueError("does not support an attention implementation through attn_implementation")
    causal = FakeAutoClass({None: _FLASH, "sdpa": refused, "eager": None})
    _patch(monkeypatch, AutoModelForCausalLM=causal)

    assert hf._load_model(ModelCfg(model_id="fake"), torch.float32) == "model:eager"
    assert causal.calls == [None, "sdpa", "eager"]


def test_a_named_sdpa_still_loads_when_the_class_refuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = ValueError(
        "LlamaModel does not support an attention implementation through "
        "torch.nn.functional.scaled_dot_product_attention yet."
    )
    causal = FakeAutoClass({"sdpa": refused, "eager": None})
    _patch(monkeypatch, AutoModelForCausalLM=causal)

    cfg = ModelCfg(model_id="fake", attn_implementation="sdpa")

    # Loaded on eager here; promote_to_sdpa sets the kernel afterwards.
    assert hf._load_model(cfg, torch.float32) == "model:eager"
    assert causal.calls == ["sdpa", "eager"]


def test_another_named_implementation_is_not_second_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    causal = FakeAutoClass({"eager": _WRONG_CLASS})
    _patch(monkeypatch, AutoModelForCausalLM=causal)

    cfg = ModelCfg(model_id="fake", attn_implementation="eager")

    with pytest.raises(RuntimeError, match="could not load"):
        hf._load_model(cfg, torch.float32)

    assert causal.calls == ["eager"]


def test_an_auto_class_that_refuses_the_checkpoint_is_tried_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_text = FakeAutoClass({None: _WRONG_CLASS})
    causal = FakeAutoClass({None: _FLASH, "sdpa": None})
    _patch(
        monkeypatch,
        AutoModelForImageTextToText=image_text,
        AutoModelForCausalLM=causal,
    )

    assert hf._load_model(ModelCfg(model_id="fake"), torch.float32) == "model:sdpa"
    assert image_text.calls == [None]


def test_a_named_implementation_is_not_second_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    causal = FakeAutoClass({"flash_attention_2": _FLASH, "sdpa": None})
    _patch(monkeypatch, AutoModelForCausalLM=causal)

    with pytest.raises(RuntimeError, match="could not load"):
        hf._load_model(
            ModelCfg(model_id="fake", attn_implementation="flash_attention_2"), torch.float32
        )

    assert causal.calls == ["flash_attention_2"]
