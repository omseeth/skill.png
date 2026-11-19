"""Splits are grouped, and ungrouped data is refused.

A universal image memorizes whatever the training items share. If a source document appears in
both train and test, the test number measures memorization of that document.
"""

from __future__ import annotations

import pytest

from skillpng.benchmarks.base import Item, split_by_group

FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def _items(n: int = 120, groups: int = 20) -> list[Item]:
    return [
        Item(id=f"i{i}", target="x", ground_truth=None, group=f"doc{i % groups}") for i in range(n)
    ]


def test_no_group_appears_in_two_splits() -> None:
    splits = split_by_group(_items(), FRACTIONS, seed=0)
    seen: dict[str, str] = {}
    for name, items in splits.items():
        for item in items:
            assert item.group is not None
            assert seen.setdefault(item.group, name) == name


def test_every_item_lands_somewhere_exactly_once() -> None:
    items = _items()
    splits = split_by_group(items, FRACTIONS, seed=0)
    placed = [item.id for split in splits.values() for item in split]
    assert sorted(placed) == sorted(item.id for item in items)


def test_splitter_refuses_ungrouped_data() -> None:
    items = _items(10, 5)
    items[3].group = None
    with pytest.raises(ValueError, match="group=None"):
        split_by_group(items, FRACTIONS)


def test_ungrouped_data_can_be_forced() -> None:
    items = _items(10, 5)
    items[3].group = None
    splits = split_by_group(items, FRACTIONS, allow_ungrouped=True)
    assert sum(len(split) for split in splits.values()) == 10


def test_assignment_is_deterministic_and_seed_dependent() -> None:
    items = _items()
    first = split_by_group(items, FRACTIONS, seed=0)
    again = split_by_group(items, FRACTIONS, seed=0)
    other = split_by_group(items, FRACTIONS, seed=1)

    assert {k: [i.id for i in v] for k, v in first.items()} == {
        k: [i.id for i in v] for k, v in again.items()
    }
    assert [i.id for i in first["train"]] != [i.id for i in other["train"]]


def test_proportions_are_roughly_respected_with_many_groups() -> None:
    items = _items(2000, groups=400)
    splits = split_by_group(items, FRACTIONS, seed=0)
    for name, fraction in FRACTIONS.items():
        assert abs(len(splits[name]) / len(items) - fraction) < 0.07


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        split_by_group(_items(10, 3), {"train": 0.5, "test": 0.2})
