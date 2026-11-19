"""`model.max_sequence_tokens` bounds the whole training sequence, not just the target.

The prompt half of that sequence is the image placeholders plus the injected turns, and it is the
half we must never cut: trimming it moves the image token count the objective is defined on. So the
budget is spent on the prompt first and the target absorbs what is left — and when the prompt alone
fills it, that is an error rather than a quietly reshaped objective.
"""

from __future__ import annotations

import pytest
import torch

from skillpng.config import ModelCfg
from skillpng.models.hf import HFVLM


class FakeTokenizer:
    """One token per whitespace-separated word, so token counts are readable in the assertions."""

    eos_token = "</s>"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [hash(word) % 100 for word in text.split()]}


def _vlm(**cfg: object) -> HFVLM:
    """An HFVLM with the tokenizer wired up and no weights loaded — `_encode_target` touches
    `cfg`, `tokenizer` and `device`, and nothing else."""
    vlm = HFVLM.__new__(HFVLM)
    vlm.cfg = ModelCfg(model_id="fake", **cfg)  # type: ignore[arg-type]
    vlm.processor = FakeTokenizer()
    vlm.device = torch.device("cpu")
    return vlm


def _target(words: int) -> str:
    return " ".join(f"w{i}" for i in range(words))


def test_target_is_trimmed_to_what_the_prompt_leaves() -> None:
    vlm = _vlm(max_sequence_tokens=100)
    ids = vlm._encode_target(_target(50), prompt_len=80)
    assert ids.shape[1] == 20  # 80 + 20 == the budget


def test_a_sequence_inside_the_budget_is_untouched() -> None:
    vlm = _vlm(max_sequence_tokens=100)
    assert vlm._encode_target(_target(15), prompt_len=80).shape[1] == 15


def test_max_target_tokens_applies_before_the_sequence_budget() -> None:
    # The target cap is the tighter of the two here, so it is the one that binds.
    vlm = _vlm(max_target_tokens=10, max_sequence_tokens=1000)
    assert vlm._encode_target(_target(50), prompt_len=5).shape[1] == 10

    # ...and the other way round: the sequence budget cuts further into an already-capped target.
    vlm = _vlm(max_target_tokens=100, max_sequence_tokens=60)
    assert vlm._encode_target(_target(50), prompt_len=45).shape[1] == 15


def test_a_prompt_that_fills_the_budget_is_an_error() -> None:
    vlm = _vlm(max_sequence_tokens=100)
    with pytest.raises(ValueError, match="max_sequence_tokens"):
        vlm._encode_target(_target(5), prompt_len=100)
    with pytest.raises(ValueError, match="no room for the target"):
        vlm._encode_target(_target(5), prompt_len=250)


def test_no_budget_means_no_trimming() -> None:
    vlm = _vlm()
    assert vlm._encode_target(_target(400), prompt_len=9000).shape[1] == 400
