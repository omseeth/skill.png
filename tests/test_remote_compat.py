"""A checkpoint's own modeling code, written for an older transformers, still runs.

Nemotron 3 Nano Omni's modeling file calls create_causal_mask with the
argument names transformers used before 5.4, and brings a cache class of its
own that predates the query offset 5.14 asks every cache for. Both crashed
the first forward of a run. The calls are now translated at the boundary, so
the remote file runs unchanged.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from skillpng.models import compat


def _fake_masking(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], Any]:
    """Stand in for transformers.masking_utils with a factory that records calls.

    A second module holds the same function object under the same name, the
    way a checkpoint's modeling file does once it has imported it.

    Args:
        monkeypatch: The pytest patcher.

    Returns:
        The list the recorded calls land in, and the module standing in for
        the remote modeling file.
    """
    calls: list[dict[str, Any]] = []

    def create_causal_mask(
        config: Any,
        inputs_embeds: Any,
        attention_mask: Any = None,
        past_key_values: Any = None,
    ) -> str:
        """Record the call as this transformers would receive it.

        Args:
            config: The model config.
            inputs_embeds: The embeddings, under the name 5.4 gave them.
            attention_mask: The 2-D padding mask.
            past_key_values: The cache.

        Returns:
            A stand-in for the mask.
        """
        calls.append(
            {
                "config": config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
        )

        return "mask"

    masking = types.ModuleType("transformers.masking_utils")
    vars(masking)["create_causal_mask"] = create_causal_mask
    transformers = types.ModuleType("transformers")
    vars(transformers)["masking_utils"] = masking
    remote = types.ModuleType("transformers_modules.fake_checkpoint.modeling_fake")
    vars(remote)["create_causal_mask"] = create_causal_mask
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking)
    monkeypatch.setitem(sys.modules, remote.__name__, remote)

    return calls, remote


def test_the_older_argument_names_reach_the_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, remote = _fake_masking(monkeypatch)

    assert compat.patch_mask_factories() == ["create_causal_mask"]

    result = remote.create_causal_mask(
        config="cfg", input_embeds="embeds", cache_position="pos", past_key_values="cache"
    )

    assert result == "mask"
    assert calls == [
        {
            "config": "cfg",
            "inputs_embeds": "embeds",
            "attention_mask": None,
            "past_key_values": "cache",
        }
    ]


def test_a_call_in_the_current_spelling_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _remote = _fake_masking(monkeypatch)
    compat.patch_mask_factories()

    sys.modules["transformers.masking_utils"].create_causal_mask(
        "cfg", "embeds", attention_mask="mask2d"
    )

    assert calls[0]["inputs_embeds"] == "embeds"
    assert calls[0]["attention_mask"] == "mask2d"


def test_the_factory_is_wrapped_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_masking(monkeypatch)
    compat.patch_mask_factories()

    assert compat.patch_mask_factories() == []


def _fake_remote_module(monkeypatch: pytest.MonkeyPatch) -> tuple[str, Any]:
    """Put a fresh cache class from a fake remote checkpoint in sys.modules.

    The class is built per call, so patching it leaks into no other test.

    Args:
        monkeypatch: The pytest patcher.

    Returns:
        The module name it was registered under, and the class itself.
    """
    name = "transformers_modules.fake_checkpoint.modeling_fake"

    class FakeCache:
        """A checkpoint's own cache, with only the methods it shipped with."""

        def get_seq_length(self, layer_idx: int = 0) -> int:
            """Tokens cached for a layer.

            Args:
                layer_idx: The layer asked about.

            Returns:
                A fixed length standing in for the cached tokens.
            """
            return 7

        def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
            """Length and offset of the cache, as the mask factories read them.

            Args:
                query_length: Tokens in this forward.
                layer_idx: The layer asked about.

            Returns:
                The key-value length and its offset.
            """
            return self.get_seq_length(layer_idx) + query_length, 0

    FakeCache.__module__ = name
    module = types.ModuleType(name)
    vars(module)["FakeCache"] = FakeCache
    monkeypatch.setitem(sys.modules, name, module)

    return name, FakeCache


def test_a_remote_cache_gets_the_query_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    name, cache = _fake_remote_module(monkeypatch)

    assert compat.patch_remote_caches() == [f"{name}.FakeCache"]
    assert cache().get_query_offset(0) == 7


def test_a_cache_that_already_has_one_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _name, cache = _fake_remote_module(monkeypatch)
    cache.get_query_offset = lambda self, layer_idx=0: 99

    assert compat.patch_remote_caches() == []
    assert cache().get_query_offset() == 99
