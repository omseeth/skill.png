"""The vision tower's index buffer survives a load that does not carry it.

C-RADIO computes summary_idxs in its own __init__ from the teachers its
config names, and no checkpoint stores it. Transformers builds the model on
the meta device and materializes every key the checkpoint is missing, so the
buffer arrives holding whatever the allocation held. Nemotron 3 Nano Omni
then indexes its class tokens with it, and the first image forward died on a
CUDA device-side assert reported by the next kernel to run.
"""

from __future__ import annotations

import torch
from torch import nn

from skillpng.models.compat import repair_radio_summary_idxs

_TEACHERS = [
    {"name": "clip", "use_summary": True},
    {"name": "siglip", "use_summary": True},
    {"name": "dino_v2", "use_summary": True},
    {"name": "sam", "use_summary": False},
]


class Config:
    """A RADIO config, which carries its training arguments as a plain dict.

    Args:
        args: The argument dict, teachers among them.

    Returns:
        None.
    """

    def __init__(self, args: dict[str, object]) -> None:
        self.args = args


class Tower(nn.Module):
    """The module holding the buffer, standing in for RADIOModelBase.

    Args:
        idxs: What the buffer holds after the load.

    Returns:
        None.
    """

    def __init__(self, idxs: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("summary_idxs", idxs)


class Radio(nn.Module):
    """The wrapper carrying the config, standing in for RADIOModel.

    Args:
        idxs: What the tower's buffer holds after the load.
        args: The config arguments, defaulting to a real teacher list.

    Returns:
        None.
    """

    def __init__(self, idxs: torch.Tensor, args: dict[str, object] | None = None) -> None:
        super().__init__()
        self.radio_model = Tower(idxs)
        self.config = Config(args if args is not None else {"teachers": _TEACHERS})


class Model(nn.Module):
    """A model with a vision tower under it, the way the omni checkpoint is.

    Args:
        vision_model: The tower wrapper.

    Returns:
        None.
    """

    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model


def test_an_uninitialized_buffer_is_rebuilt_from_the_teachers() -> None:
    # The three teachers with use_summary, which is what the tower's own
    # __init__ would have computed.
    model = Model(Radio(torch.tensor([94055, -1, 0, 7], dtype=torch.int64)))

    rebuilt = repair_radio_summary_idxs(model)

    assert model.vision_model.radio_model.summary_idxs.tolist() == [0, 1, 2]
    assert rebuilt == ["vision_model.radio_model.summary_idxs"]


def test_a_buffer_that_already_indexes_its_teachers_is_left_alone() -> None:
    model = Model(Radio(torch.tensor([0, 1, 2], dtype=torch.int64)))

    assert repair_radio_summary_idxs(model) == []


def test_the_rebuilt_buffer_keeps_its_dtype_and_stays_a_buffer() -> None:
    model = Model(Radio(torch.tensor([9, 9, 9, 9], dtype=torch.int64)))

    repair_radio_summary_idxs(model)
    tower = model.vision_model.radio_model

    assert tower.summary_idxs.dtype is torch.int64
    assert "summary_idxs" in dict(tower.named_buffers())


def test_a_wrong_length_buffer_is_rebuilt_too() -> None:
    # In range, but one index per teacher rather than one per summary: this
    # is not what the tower computes, and the count feeds its embed_dim.
    model = Model(Radio(torch.tensor([0, 1, 2, 3], dtype=torch.int64)))

    assert repair_radio_summary_idxs(model) == ["vision_model.radio_model.summary_idxs"]
    assert model.vision_model.radio_model.summary_idxs.tolist() == [0, 1, 2]


def test_a_model_with_no_radio_config_is_untouched() -> None:
    plain = Model(nn.Linear(2, 2))

    assert repair_radio_summary_idxs(plain) == []


def test_a_config_whose_args_is_not_a_dict_is_passed_over() -> None:
    model = Model(Radio(torch.tensor([9], dtype=torch.int64), args={}))

    assert repair_radio_summary_idxs(model) == []


def test_a_buffer_still_on_meta_is_rebuilt_on_a_real_device() -> None:
    # Nothing to read at all, which is the same load leaving even less
    # behind. The value must not come back on meta.
    model = Model(Radio(torch.empty(4, dtype=torch.int64, device="meta")))

    assert repair_radio_summary_idxs(model) == ["vision_model.radio_model.summary_idxs"]
    assert model.vision_model.radio_model.summary_idxs.tolist() == [0, 1, 2]
