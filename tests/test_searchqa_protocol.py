"""SearchQA prompt protocol: truncation, extraction, and reference parity.

The protocol has to stay byte-identical to the implementation the published
baseline numbers were measured against, so beyond behavioral tests there are
parity tests against the vendored reference clone under baselines/skillopt/
code. Those are skipped when the clone is absent; everything else runs
without it, without network, and without weights.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from skillpng.benchmarks.registry import build_benchmark
from skillpng.benchmarks.searchqa import searchqa
from skillpng.benchmarks.searchqa.searchqa import (
    MAX_CONTEXT_CHARS,
    SearchQA,
    extract_answer,
    truncate_context,
)
from skillpng.config import BenchmarkCfg
from test_searchqa import DEFAULT_SYSTEM_PROMPT, _mirror

REFERENCE_CODE = Path(__file__).parent.parent / "baselines" / "skillopt" / "code"
needs_clone = pytest.mark.skipif(
    not REFERENCE_CODE.is_dir(), reason="reference clone not present"
)


def _bench(monkeypatch: pytest.MonkeyPatch, dataset: dict | None = None) -> SearchQA:
    """A benchmark backed by the mirror fixture, scored as exact match.

    Args:
        monkeypatch: Fixture that patches the Hub loader.
        dataset: Mirror rows to serve, or None for the default mirror.

    Returns:
        The built adapter.
    """
    shipped = _mirror() if dataset is None else dataset
    monkeypatch.setattr(searchqa, "_load", lambda repo_id: shipped)
    options: dict[str, Any] = {"metric": "em"}
    bench = build_benchmark(
        BenchmarkCfg(
            name="searchqa", system_prompt=DEFAULT_SYSTEM_PROMPT, options=options
        )
    )
    assert isinstance(bench, SearchQA)

    return bench


def test_the_turns_are_the_system_prompt_then_context_and_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]
    messages = bench.build_messages(item, with_image=False)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert messages[1]["content"][0]["text"] == (
        f"## Context\n{item.meta['context']}\n\n## Question\n{item.meta['question']}"
    )


def test_image_injection_still_lands_in_the_user_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _bench(monkeypatch)
    messages = bench.build_messages(
        bench.items("train")[0], with_image=True, site="post_query"
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["content"][-1] == {"type": "image"}


def test_short_context_passes_through_untouched() -> None:
    context = "[DOC] [TLE] a [PAR] b"
    assert truncate_context(context) == context


def test_truncation_cuts_at_doc_boundaries() -> None:
    doc = "[DOC] [TLE] title [PAR] " + "x" * 800
    context = doc * 12
    kept = truncate_context(context)

    assert len(kept) <= MAX_CONTEXT_CHARS
    # The quirk: the leading split piece is empty, so the first kept
    # document loses its [DOC] marker.
    assert kept.startswith(" [TLE] title")
    assert kept.count("[DOC]") == 6


def test_truncation_without_a_boundary_falls_back_to_a_hard_cut() -> None:
    context = "y" * (MAX_CONTEXT_CHARS + 100)
    kept = truncate_context(context)

    assert kept == "y" * MAX_CONTEXT_CHARS + "\n...[truncated]"


def test_extraction_prefers_the_last_answer_tag() -> None:
    output = "<answer>draft</answer> reasoning goes on\n<answer> Paris </answer>\n"
    assert extract_answer(output) == "Paris"
    assert extract_answer("<ANSWER>Paris</ANSWER>") == "Paris"


def test_extraction_falls_back_to_the_last_non_empty_line() -> None:
    assert extract_answer("thinking...\n\nParis\n\n") == "Paris"
    assert extract_answer("   ") == ""


def test_scoring_reads_the_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]
    item.ground_truth = ["the Sistine Chapel"]
    output = "Step by step.\n<answer>Sistine Chapel</answer>\nDone."

    assert bench.score(output, item) == 1.0
    assert bench.score("Sistine Chapel is my answer\nno tag here", item) == 0.0


@needs_clone
def test_system_prompt_matches_the_vendored_template() -> None:
    template = (
        REFERENCE_CODE / "skillopt" / "envs" / "searchqa" / "prompts" / "rollout_system.md"
    ).read_text()
    # The reference wraps an optional skill section the default prompt drops;
    # with no skill the two must be byte-identical, or the baseline numbers
    # stop being comparable.
    assert template.format(skill_section="") == DEFAULT_SYSTEM_PROMPT


@needs_clone
def test_truncation_matches_the_vendored_implementation() -> None:
    """Compile the reference _truncate_context from source and compare outputs."""
    source = (REFERENCE_CODE / "skillopt" / "envs" / "searchqa" / "rollout.py").read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_truncate_context"
    )
    namespace: dict[str, Any] = {"_MAX_CONTEXT_CHARS": MAX_CONTEXT_CHARS}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<vendored>", "exec"), namespace)
    theirs = namespace["_truncate_context"]

    doc = "[DOC] [TLE] t [PAR] " + "z" * 700
    cases = [
        "short",
        doc * 3,
        doc * 20,
        "q" * (MAX_CONTEXT_CHARS + 5),
        ("[DOC]" + "w" * 7000) * 2,
    ]

    for case in cases:
        assert truncate_context(case) == theirs(case)


@needs_clone
def test_scoring_matches_the_vendored_evaluator() -> None:
    """Import the reference evaluator standalone and compare on shared cases."""
    import importlib.util

    path = REFERENCE_CODE / "skillopt" / "envs" / "searchqa" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("reference_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    outputs = [
        "<answer>The Sistine Chapel.</answer>",
        "thinking\n<answer>draft</answer>\n<answer>Lou Gehrig</answer>",
        "no tags\nlast line answer",
        "   ",
        "<answer>a b c</answer>",
    ]
    golds = [["the Sistine Chapel"], ["(Lou) Gehrig", "Lou Gehrig"], ["answer"], ["x"], ["b c d"]]

    for output, gold in zip(outputs, golds, strict=True):
        theirs = module.evaluate(output, gold)
        prediction = extract_answer(output)
        assert prediction == theirs["predicted_answer"]
        ours_em = max(searchqa.exact_match(prediction, value) for value in gold)
        ours_f1 = max(searchqa.token_f1(prediction, value) for value in gold)
        assert ours_em == theirs["em"]
        assert ours_f1 == pytest.approx(theirs["f1"])
