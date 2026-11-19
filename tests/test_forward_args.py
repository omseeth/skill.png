"""A processor key its own model's forward cannot take no longer ends the run.

Nemotron 3 Nano Omni's processor returns num_patches, which its forward has no
argument for: the first image step died on a TypeError. Its forward is also
InternVL-shaped and dereferences image_flags before anything else, which no
processor returns, so dropping the one key only moved the crash a line down.
Both are settled from the forward's own signature.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from skillpng.models.hf import (
    HFVLM,
    _forward_accepts,
    _forward_takes,
    _requires_pixel_values,
)


class NemotronShaped:
    """A stand-in with Nemotron 3 Nano Omni's forward signature and no **kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        pixel_values: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        image_flags: Any = None,
    ) -> Any:
        """Record the call and hand back logits.

        Args:
            pixel_values: The tiles.
            input_ids: Prompt and target ids.
            attention_mask: Mask over input_ids.
            image_flags: The tile mask.

        Returns:
            An object carrying logits, as a model output does.
        """
        self.calls.append(
            {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_flags": image_flags,
            }
        )

        return types.SimpleNamespace(logits=torch.zeros(1, int(input_ids.shape[1]), 7))

    def __call__(self, **kwargs: Any) -> Any:
        """Route the call the way nn.Module does.

        Args:
            kwargs: Whatever the caller passed.

        Returns:
            The forward's output.
        """
        return self.forward(**kwargs)


class Permissive:
    """A stand-in whose forward takes anything, the way most families' do."""

    def forward(self, input_ids: Any = None, **kwargs: Any) -> Any:
        """Take every keyword there is.

        Args:
            input_ids: Prompt and target ids.
            kwargs: Everything else.

        Returns:
            An object carrying logits.
        """
        return types.SimpleNamespace(logits=torch.zeros(1, 1, 7))

    def __call__(self, **kwargs: Any) -> Any:
        """Route the call the way nn.Module does.

        Args:
            kwargs: Whatever the caller passed.

        Returns:
            The forward's output.
        """
        return self.forward(**kwargs)


def _vlm(model: Any) -> HFVLM:
    """An HFVLM around a stand-in model, with no weights and no processor.

    Args:
        model: The stand-in the wrapper calls.

    Returns:
        A wrapper carrying only what the argument routing touches.
    """
    vlm = HFVLM.__new__(HFVLM)
    vlm.model = model
    vlm.device = torch.device("cpu")
    vlm._forward_accepts = _forward_accepts(model)
    vlm._keeps_logits = _forward_takes(model, "logits_to_keep")
    vlm._needs_pixels = _requires_pixel_values(model)
    vlm._dropped_extras = set()

    return vlm


def test_a_fixed_signature_is_read_off_the_forward() -> None:
    accepts = _forward_accepts(NemotronShaped())

    assert accepts == frozenset({"pixel_values", "input_ids", "attention_mask", "image_flags"})


def test_a_forward_taking_kwargs_constrains_nothing() -> None:
    assert _forward_accepts(Permissive()) is None


def test_a_key_the_forward_cannot_name_is_dropped() -> None:
    vlm = _vlm(NemotronShaped())

    kept = vlm._accepted_extras({"num_patches": torch.tensor([1]), "image_flags": None})

    assert set(kept) == {"image_flags"}


def test_the_drop_is_reported_once() -> None:
    vlm = _vlm(NemotronShaped())
    extra = {"num_patches": torch.tensor([1])}

    vlm._accepted_extras(extra)
    vlm._accepted_extras(extra)

    assert vlm._dropped_extras == {"num_patches"}


def test_nothing_is_dropped_from_a_forward_taking_kwargs() -> None:
    vlm = _vlm(Permissive())
    extra = {"num_patches": torch.tensor([1])}

    assert vlm._accepted_extras(extra) == extra


def test_image_flags_marks_every_tile_we_pass_as_real() -> None:
    vlm = _vlm(NemotronShaped())

    flags = vlm._image_flags(torch.zeros(3, 3, 8, 8), {})["image_flags"]

    assert flags.shape == (3, 1)
    assert bool((flags == 1).all())


def test_image_flags_is_left_alone_where_the_forward_has_none() -> None:
    assert _vlm(Permissive())._image_flags(torch.zeros(1, 3, 8, 8), {}) == {}


def test_a_processor_that_returns_image_flags_keeps_its_own() -> None:
    vlm = _vlm(NemotronShaped())
    own = torch.ones(1, 1, dtype=torch.long)

    assert vlm._image_flags(torch.zeros(1, 3, 8, 8), {"image_flags": own}) == {}


def test_the_image_forward_runs_on_this_family() -> None:
    model = NemotronShaped()
    vlm = _vlm(model)
    input_ids = torch.zeros(1, 5, dtype=torch.long)

    logits = vlm._forward(input_ids, torch.ones_like(input_ids), torch.zeros(1, 3, 8, 8), {})

    assert logits.shape == (1, 5, 7)
    assert model.calls[0]["image_flags"].shape == (1, 1)


def test_an_unaccepted_key_reaching_the_forward_would_still_raise() -> None:
    # The drop happens at the processor boundary; nothing downstream of it
    # filters, and this pins that the stand-in is strict enough to catch it.
    vlm = _vlm(NemotronShaped())
    input_ids = torch.zeros(1, 5, dtype=torch.long)

    with pytest.raises(TypeError, match="num_patches"):
        vlm._forward(
            input_ids,
            torch.ones_like(input_ids),
            torch.zeros(1, 3, 8, 8),
            {"num_patches": torch.tensor([1])},
        )


def test_a_forward_with_no_text_only_route_says_so() -> None:
    # Its pixel_values has no default, so the clean condition cannot be
    # measured as a loss on this family. The TypeError that used to come
    # back named nothing the caller could act on.
    vlm = _vlm(NemotronShaped())
    input_ids = torch.zeros(1, 5, dtype=torch.long)

    with pytest.raises(NotImplementedError, match="select_on"):
        vlm._forward(input_ids, torch.ones_like(input_ids), None, {})


def test_a_text_only_forward_still_runs_without_pixels() -> None:
    vlm = _vlm(Permissive())
    input_ids = torch.zeros(1, 5, dtype=torch.long)

    assert vlm._forward(input_ids, torch.ones_like(input_ids), None, {}).shape == (1, 1, 7)


class KeepsLogits:
    """A stand-in that declares logits_to_keep and slices the way transformers does."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def forward(self, input_ids: Any = None, logits_to_keep: int = 0, **kwargs: Any) -> Any:
        """Record the call and hand back as many trailing logits as were asked for.

        Args:
            input_ids: Prompt and target ids.
            logits_to_keep: Trailing positions to score, 0 for all of them.
            kwargs: Everything else.

        Returns:
            An object carrying logits.
        """
        self.calls.append({"logits_to_keep": logits_to_keep})
        kept = logits_to_keep or int(input_ids.shape[1])

        return types.SimpleNamespace(logits=torch.zeros(1, kept, 7))

    def __call__(self, **kwargs: Any) -> Any:
        """Route the call the way nn.Module does.

        Args:
            kwargs: Whatever the caller passed.

        Returns:
            The forward's output.
        """
        return self.forward(**kwargs)


def test_only_the_trailing_logits_are_asked_for_where_the_forward_takes_them() -> None:
    model = KeepsLogits()
    input_ids = torch.zeros(1, 9, dtype=torch.long)

    logits = _vlm(model)._forward(input_ids, torch.ones_like(input_ids), None, {}, keep=4)

    assert model.calls[0]["logits_to_keep"] == 4
    assert logits.shape == (1, 4, 7)


def test_a_forward_without_the_argument_still_gets_the_whole_sequence() -> None:
    # Its signature has no logits_to_keep and no **kwargs, so the keyword
    # would be a TypeError; full-length logits are right, only larger.
    model = NemotronShaped()
    input_ids = torch.zeros(1, 5, dtype=torch.long)

    logits = _vlm(model)._forward(
        input_ids, torch.ones_like(input_ids), torch.zeros(1, 3, 8, 8), {}, keep=3
    )

    assert logits.shape == (1, 5, 7)


def test_a_kwargs_only_forward_is_not_guessed_at() -> None:
    # **kwargs says nothing about what the model does with an unknown keyword,
    # and one it ignores would be a silent no-op rather than a saving.
    assert _forward_takes(KeepsLogits(), "logits_to_keep")
    assert not _forward_takes(Permissive(), "logits_to_keep")
