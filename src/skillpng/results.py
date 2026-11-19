"""Append-only JSONL results, read back by the scripts in analysis/.

No database and no ORM: a run that crashes still leaves every row it had
already written.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

# Fields that describe the injected image, left out of rows without one.
IMAGE_FIELDS = (
    "image_digest", "param_kind", "resolution", "image_token_count", "injection_site"
)


class ResultRow(BaseModel):
    """One evaluated (image, benchmark, model, split).

    One row is one condition. The no-image baseline is its own row (condition none), run once per benchmark, model and split and reported against every image scored on that same triple, so a reasoning model is never asked to answer the split twice per eval.

    A row that injects no image (none, llmskill, skillopt, lora) leaves image_digest
    unset and is written without any of the IMAGE_FIELDS.
    """
    image_digest: str | None = None
    skill_digest: str = "none"
    param_kind: str | None = None
    resolution: list[int] | None = None
    image_token_count: int | None = None
    benchmark: str
    n_distractors: int | None = None
    split: str
    model: str
    model_dtype: str
    injection_site: str | None = None
    condition: str
    score: float
    metric_name: str
    n_items: int
    seed: int
    git_sha: str
    config_hash: str
    run_name: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_serializer(mode="wrap")
    def _drop_image_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Serialise the row, without the image fields when no image was injected.

        Args:
            handler: Pydantic's default serialiser for the row.

        Returns:
            The serialised fields.
        """
        data: dict[str, object] = handler(self)
        if self.image_digest is None:
            for name in IMAGE_FIELDS:
                data.pop(name, None)

        return data


class PredictionRow(BaseModel):
    """One item's generated answer, for reading a run instead of averaging it.

    One file per condition, so what the image changed about a given answer is
    a join on item_id against the baseline run's file.
    """
    item_id: str
    question: str = ""
    target: str
    output: str
    score: float
    # Whatever the benchmark's rollout put in Result.meta: component metrics
    # and the keys a report groups them by. score is the one number the run
    # averages; everything else a benchmark measures rides here.
    meta: dict = Field(default_factory=dict)


def append_row(path: str | Path, row: ResultRow) -> Path:
    """Append one row to the JSONL file, creating parent directories.

    Args:
        path: Path to the JSONL results file.
        row: The evaluated row to serialise and append.

    Returns:
        The path that was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
    return path


class PredictionLog:
    """The per-item JSONL of one eval, written while the eval is still running.

    Truncated on open and flushed per line, so it always holds the latest eval
    of a split and can be followed with tail -f instead of only read at the
    end. The run's history lives in the results file, not here.
    """

    def __init__(self, path: str | Path) -> None:
        """Open the file for writing, creating parent directories.

        Args:
            path: Path to the per-item JSONL file.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w")

    def write(self, row: PredictionRow) -> None:
        """Append one row and flush it, so a reader sees it immediately.

        Args:
            row: The item's answer under the condition that was scored.
        """
        self._handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self._handle.close()

    def __enter__(self) -> PredictionLog:
        """Enter the context manager.

        Returns:
            This log.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the file, whether or not the eval finished.

        Args:
            exc_info: Exception details, ignored: a partial log is still worth
                keeping when a run dies midway.
        """
        self.close()


def read_rows(path: str | Path) -> Iterator[dict[str, object]]:
    """Yield each non-empty line of the JSONL file as a decoded dict.

    Args:
        path: Path to the JSONL results file.

    Returns:
        An iterator over the decoded rows.
    """
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def git_sha() -> str:
    """Return the short HEAD sha, suffixed with -dirty for uncommitted work.
    
    Returns:
        The short sha, or unknown when git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )

    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    suffix = "-dirty" if dirty.stdout.strip() else ""

    return out.stdout.strip() + suffix
