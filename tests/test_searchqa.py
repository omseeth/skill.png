"""SearchQA adapter: the shipped id split, the MRQA row shape, the prompt, and scoring.

No network and no weights — `_load` is patched with the row shape `lucadiliello/searchqa` hands
back, keyed by the ids the manifest actually names, so nothing here downloads the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from skillpng.benchmarks.registry import build_benchmark
from skillpng.benchmarks.searchqa import searchqa
from skillpng.benchmarks.searchqa.searchqa import (
    SPLITS,
    SearchQA,
    SearchQAOptions,
    _repo_id,
    _split_ids,
    exact_match,
    extract_answer,
    normalize_answer,
    token_f1,
)
from skillpng.config import BenchmarkCfg, load_config

Rows = list[dict[str, Any]]

# The shipped prompt lives in the base config, not in the schema: reading it
# from there is also what checks the two stay wired together.
BASE_CONFIG = Path(__file__).parent.parent / "configs" / "base.yaml"
DEFAULT_SYSTEM_PROMPT = load_config(BASE_CONFIG).benchmark.system_prompt

# The shipped manifest, read once: these are the only ids the adapter will ever ask for.
TRAIN_IDS = _split_ids("train")
VAL_IDS = _split_ids("val")
TEST_IDS = _split_ids("test")
SIZES = {"train": 400, "val": 200, "test": 1400}


def _context(item_id: str, docs: int = 11) -> str:
    """MRQA's flattened snippet block: one `[DOC]` per result, `[TLE]`/`[PAR]` inside it.

    The adapter feeds this block to the model, so the fixture has to look like the real one.
    """
    return " ".join(
        f"[DOC] [TLE] title {rank} [PAR] snippet {rank} about clue {item_id}"
        for rank in range(docs)
    )


def _row(item_id: str, answers: list[str] | None = None) -> dict[str, Any]:
    """The Hub shape: context/question/answers/key/labels, and nothing else."""
    return {
        "context": _context(item_id),
        "question": f"This is the clue keyed {item_id}",
        "answers": answers if answers is not None else [f"answer {item_id}"],
        "key": item_id,
        "labels": [{"start": [10], "end": [20]}],
    }


def _mirror(drop: set[str] | None = None) -> dict[str, Rows]:
    """A mirror carrying every manifest id, spread over the two splits the real one ships.

    Unrelated rows sit alongside them: the adapter is supposed to walk past everything the manifest
    does not name.
    """
    dropped = drop or set()
    named = [item_id for item_id in TRAIN_IDS + VAL_IDS + TEST_IDS if item_id not in dropped]
    noise = [_row(f"unlisted-{i}") for i in range(5)]
    half = len(named) // 2
    return {
        "train": [_row(item_id) for item_id in named[:half]] + noise,
        "validation": [_row(item_id) for item_id in named[half:]],
    }


def _bench(
    monkeypatch: pytest.MonkeyPatch,
    dataset: dict[str, Rows] | None = None,
    limit: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    **options: Any,
) -> SearchQA:
    """A benchmark whose `_load` returns `dataset` (default: a mirror with every manifest id)."""
    shipped = _mirror() if dataset is None else dataset
    monkeypatch.setattr(searchqa, "_load", lambda repo_id: shipped)
    bench = build_benchmark(
        BenchmarkCfg(
            name="searchqa", limit=limit, system_prompt=system_prompt, options=options
        )
    )
    assert isinstance(bench, SearchQA)
    return bench


def test_manifest_ships_the_expected_splits() -> None:
    assert (len(TRAIN_IDS), len(VAL_IDS), len(TEST_IDS)) == (400, 200, 1400)
    assert len(set(TRAIN_IDS) | set(VAL_IDS) | set(TEST_IDS)) == 2000  # no id in two splits


def test_splits_are_the_manifest_ids_in_manifest_order(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch)
    assert [item.id for item in bench.items("train")] == TRAIN_IDS
    assert [item.id for item in bench.items("val")] == VAL_IDS
    assert [item.id for item in bench.items("test")] == TEST_IDS


def test_unknown_split_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch)
    with pytest.raises(KeyError, match="unknown split"):
        bench.items("validation")


def test_rows_outside_the_manifest_are_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch)
    loaded = {item.id for name in SPLITS for item in bench.items(name)}
    assert loaded == set(TRAIN_IDS) | set(VAL_IDS) | set(TEST_IDS)
    assert not any(item_id.startswith("unlisted-") for item_id in loaded)


def test_missing_manifest_ids_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial split is not this split, so a mirror without the ids is an error, not a warning."""
    bench = _bench(monkeypatch, _mirror(drop={TRAIN_IDS[0], TEST_IDS[-1]}))
    with pytest.raises(ValueError, match="2 of 2000 manifest id"):
        bench.items("train")


def test_row_without_an_answer_fails_rather_than_shrinking_the_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _mirror(drop={TRAIN_IDS[0]})
    dataset["train"].append(_row(TRAIN_IDS[0], answers=[]))
    bench = _bench(monkeypatch, dataset)
    with pytest.raises(ValueError, match="no usable question or answer"):
        bench.items("train")


def test_rows_without_a_context_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest names the row, and nothing reads `context` anyway."""
    dataset = _mirror(drop={TRAIN_IDS[0]})
    dataset["train"].append({**_row(TRAIN_IDS[0]), "context": ""})
    bench = _bench(monkeypatch, dataset)
    item = bench.items("train")[0]
    assert item.id == TRAIN_IDS[0] and "context" not in item.meta


def test_limit_takes_the_prefix_of_every_split(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch, limit=20)
    assert [item.id for item in bench.items("train")] == TRAIN_IDS[:20]
    assert [item.id for item in bench.items("val")] == VAL_IDS[:20]
    assert len(bench.items("test")) == 20


def test_loads_hub_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch)
    items = [item for name in SIZES for item in bench.items(name)]
    assert len(items) == sum(SIZES.values())
    assert all(item.target == f"<answer>{item.ground_truth[0]}</answer>" for item in items)
    assert all(item.id == item.id.strip() and len(item.id) == 32 for item in items)


def test_prompt_is_the_system_turn_then_context_and_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]

    clean = bench.build_messages(item, with_image=False)
    assert [message["role"] for message in clean] == ["system", "user"]
    assert clean[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert DEFAULT_SYSTEM_PROMPT.startswith("You are an expert question answering agent.")

    text = clean[1]["content"][0]["text"]
    assert text.startswith("## Context\n")
    assert item.meta["question"] in text
    assert "[DOC]" in text  # the snippet block is fed, not withheld

    with_image = bench.build_messages(item, with_image=True, site="post_query")
    # The image rides in the user turn; the system turn stays text-only.
    assert with_image[0]["content"] == clean[0]["content"]
    assert with_image[1]["content"][-1] == {"type": "image"}
    assert clean[1]["content"][0]["text"] == text  # inject_image did not mutate the input


def test_system_prompt_is_configurable_and_can_be_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _bench(monkeypatch, system_prompt="Answer in <answer> tags.")
    messages = bench.build_messages(bench.items("train")[0], with_image=False)
    assert messages[0]["content"] == "Answer in <answer> tags."

    bare = _bench(monkeypatch, system_prompt="")
    messages = bare.build_messages(bare.items("train")[0], with_image=False)
    assert [message["role"] for message in messages] == ["user"]


def test_an_item_without_a_context_cannot_be_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _mirror(drop={TRAIN_IDS[0]})
    dataset["train"].append({**_row(TRAIN_IDS[0]), "context": ""})
    bench = _bench(monkeypatch, dataset)

    with pytest.raises(ValueError, match="no context"):
        bench.build_messages(bench.items("train")[0], with_image=False)


def test_repo_id_defaults_to_the_mirror_and_accepts_the_hf_prefix() -> None:
    assert _repo_id(None) == "lucadiliello/searchqa"
    assert _repo_id("  ") == "lucadiliello/searchqa"
    assert _repo_id("hf:lucadiliello/searchqa") == "lucadiliello/searchqa"
    assert _repo_id("someone/searchqa_mirror") == "someone/searchqa_mirror"


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown searchqa option"):
        SearchQAOptions.parse({"n_snippet": 5})


def test_options_dropped_with_the_old_code_are_rejected() -> None:
    # group_by and hf_config named fields this mirror does not have; splits_from and min_snippets
    # named a re-split and a filter the id manifest replaced; n_snippets and max_snippet_chars sized
    # a snippet block this adapter no longer builds; style and skill_path chose between prompt
    # protocols there is now only one of. Failing loudly beats a config that looks like it is
    # still choosing the split, the protocol, or what the model is shown.
    for dead in (
        "group_by",
        "hf_config",
        "splits_from",
        "min_snippets",
        "n_snippets",
        "max_snippet_chars",
        "style",
        "skill_path",
    ):
        with pytest.raises(ValueError, match="unknown searchqa option"):
            SearchQAOptions.parse({dead: "category"})


def test_metric_name_follows_the_configured_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _bench(monkeypatch).metric_name == "unigram_f1"
    assert _bench(monkeypatch, metric="em").metric_name == "exact_match"


def test_scoring_is_squad_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch, metric="em")
    item = bench.items("train")[0]
    item.ground_truth = ["the Sistine Chapel"]

    assert bench.score("<answer>Sistine Chapel</answer>", item) == 1.0
    assert bench.score("<answer>the sistine chapel.</answer>", item) == 1.0
    assert bench.score("Michelangelo painted it.\n<answer>The Sistine Chapel</answer>", item) == 1.0
    assert bench.score("<answer>the Louvre</answer>", item) == 0.0
    # Explanation around the tag is ignored; explanation *inside* it is not.
    assert bench.score("<answer>The Sistine Chapel ceiling</answer>", item) == 0.0


def test_any_accepted_answer_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _mirror(drop={TRAIN_IDS[0]})
    dataset["train"].append(_row(TRAIN_IDS[0], answers=["(Lou) Gehrig", "Lou Gehrig"]))
    bench = _bench(monkeypatch, dataset, metric="em")
    item = bench.items("train")[0]
    # The target carries the tags the prompt asks for; the gold forms do not.
    assert item.target == "<answer>(Lou) Gehrig</answer>"
    assert item.ground_truth == ["(Lou) Gehrig", "Lou Gehrig"]
    assert bench.score("<answer>Lou Gehrig</answer>", item) == 1.0
    assert bench.score("<answer>Babe Ruth</answer>", item) == 0.0


def test_duplicate_and_empty_answers_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _mirror(drop={TRAIN_IDS[0]})
    dataset["train"].append(_row(TRAIN_IDS[0], answers=["Ethan Allen", "Ethan Allen", ""]))
    bench = _bench(monkeypatch, dataset)
    assert bench.items("train")[0].ground_truth == ["Ethan Allen"]


def test_f1_gives_partial_credit_where_exact_match_does_not() -> None:
    assert exact_match("Sistine Chapel ceiling", "the Sistine Chapel") == 0.0
    assert 0.0 < token_f1("Sistine Chapel ceiling", "the Sistine Chapel") < 1.0
    assert token_f1("the Sistine Chapel", "Sistine chapel!") == 1.0
    assert token_f1("Louvre", "Sistine Chapel") == 0.0


def test_extraction_and_normalization_helpers() -> None:
    assert extract_answer("reasoning\n<answer>  Paris </answer>") == "Paris"
    assert extract_answer("no tag here\nParis") == "Paris"
    assert extract_answer("Paris") == "Paris"
    assert normalize_answer("The Grapes  of Wrath!") == "grapes of wrath"


def test_a_reasoning_block_does_not_hide_the_tagged_answer() -> None:
    """A thinking template opens the block in the prompt, so the generated text
    carries the closing tag without its opener. The tag is read wherever it
    lands, and a later tag overrides a draft one made while reasoning."""
    assert extract_answer("A city.\n</think>\n<answer>Paris</answer>") == "Paris"
    drafted = "<think>\n<answer>Rome</answer>\n</think>\n<answer>Paris</answer>"
    assert extract_answer(drafted) == "Paris"


def test_a_tag_named_while_reasoning_does_not_swallow_the_answer() -> None:
    """Nemotron names the tag in its trace before answering. A block that ran
    from that mention to the real closing tag scored the trace as the answer."""
    assert extract_answer("Put it in <answer> tags.\n</think>\n<answer>Paris</answer>") == "Paris"
    assert extract_answer("thought\nUse <answer> tags.\n<answer>Paris</answer>") == "Paris"
    assert extract_answer("<answer>Paris</answer>\nThat was the <answer> block.") == "Paris"


def test_a_draft_inside_the_reasoning_trace_is_never_scored() -> None:
    assert extract_answer("<answer>Rome</answer>\n</think>\nParis") == "Paris"


def test_truncated_reasoning_does_not_pass_as_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this guards: a budget too small to close the block leaves
    only reasoning, which must score 0 rather than pick up stray credit."""
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]
    item.ground_truth = ["Paris"]
    assert bench.score("Thinking Process:\n\n1.  **Analyze the", item) == 0.0
