"""One JSONL row per evaluation, appended and read back as plain JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skillpng.results import IMAGE_FIELDS, ResultRow, append_row, read_rows


def _row(**overrides: Any) -> ResultRow:
    fields: dict[str, Any] = {
        "image_digest": "abc123",
        "param_kind": "full",
        "resolution": [448, 448],
        "image_token_count": 256,
        "benchmark": "bench_a",
        "split": "test",
        "model": "vlm-x",
        "model_dtype": "bfloat16",
        "injection_site": "pre_query",
        "condition": "skillpng",
        "score": 0.55,
        "metric_name": "accuracy",
        "n_items": 100,
        "seed": 0,
        "git_sha": "deadbee",
        "config_hash": "0123456789abcdef",
    }
    fields.update(overrides)
    return ResultRow(**fields)


def test_rows_append_without_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    append_row(path, _row())
    append_row(path, _row(benchmark="bench_b", score=0.31))

    rows = list(read_rows(path))
    assert len(rows) == 2
    assert [row["benchmark"] for row in rows] == ["bench_a", "bench_b"]
    assert rows[0]["timestamp"]


def test_a_row_carries_one_score_for_one_condition(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    append_row(path, _row(condition="none", score=0.40, **_without_image()))
    rows = list(read_rows(path))
    assert rows[0]["condition"] == "none"
    assert rows[0]["score"] == 0.40
    assert "clean_score" not in rows[0] and "image_score" not in rows[0]


def test_only_a_row_with_an_image_carries_the_image_fields(tmp_path: Path) -> None:
    """A text-only row is written without any image field.

    Args:
        tmp_path: Directory for the results file.
    """
    path = tmp_path / "results.jsonl"
    append_row(path, _row())
    append_row(path, _row(condition="llmskill", skill_digest="f00d", **_without_image()))

    imaged, text_only = read_rows(path)
    assert set(IMAGE_FIELDS) <= set(imaged)
    assert not set(IMAGE_FIELDS) & set(text_only)
    assert text_only["skill_digest"] == "f00d"


def test_an_image_row_keeps_an_unknown_token_count(tmp_path: Path) -> None:
    """Only a missing image drops the fields, not a missing token count.

    Args:
        tmp_path: Directory for the results file.
    """
    path = tmp_path / "results.jsonl"
    append_row(path, _row(image_token_count=None))

    [row] = read_rows(path)
    assert row["image_digest"] == "abc123"
    assert row["image_token_count"] is None


def _without_image() -> dict[str, Any]:
    """Overrides that unset every image field.

    Returns:
        Each image field mapped to None.
    """
    return dict.fromkeys(IMAGE_FIELDS)
