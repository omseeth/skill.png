"""LiveMathematicianBench adapter: the shipped id split, the option permutation, the prompt, and
scoring.

No network and no weights: _load is patched with records in the shape the pinned
qa_<month>_final.json files hand back, built for the ids the manifest actually names, so nothing
here downloads the dataset.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from skillpng.benchmarks.base import Item, inject_image
from skillpng.benchmarks.livemathematicianbench import livemathematicianbench
from skillpng.benchmarks.livemathematicianbench.livemathematicianbench import (
    HF_REPO,
    LABELS,
    SPLITS,
    LiveMathematicianBench,
    _manifest,
    _split_entries,
    extract_label,
    present_options,
)
from skillpng.benchmarks.registry import build_benchmark
from skillpng.config import BenchmarkCfg, load_config

CONFIG = Path(__file__).parent.parent / "configs" / "livemathematicianbench.yaml"
SYSTEM_PROMPT = load_config(CONFIG).benchmark.system_prompt

# The shipped manifest, read once: these are the only ids the adapter will ever ask for.
MANIFEST = _manifest()
ENTRIES = {name: _split_entries(name) for name in SPLITS}
IDS = {name: [entry["id"] for entry in entries] for name, entries in ENTRIES.items()}
ALL_IDS = [item_id for name in SPLITS for item_id in IDS[name]]


def _record(entry: dict) -> dict:
    """The shipped record shape: the correct choice labelled A, four distractors B-E."""
    item_id = entry["id"]
    return {
        "no": entry["no"],
        "month": entry["month"],
        "paper_link": entry["paper_link"],
        "theorem": f"THEOREM-{item_id}",
        "sketch": f"SKETCH-{item_id}",
        "theorem_type": ["existence"],
        "mcq": {
            "question": f"  Which statement holds for {item_id}?\n\\[\n  x^2 \\ge 0\n\\]\n",
            "correct_choice": {"label": "A", "text": f"correct {item_id}"},
            "choices": [
                {"label": label, "text": f" distractor {label} {item_id}\n"} for label in "BCDE"
            ],
            "meta": {
                "weaker_true_label": "C",
                "false_labels": ["B", "D", "E"],
                "wildcard_false_label": "B",
            },
            "sketch_usage_meta": {},
        },
    }


def _files(drop: set[str] | None = None) -> dict[str, list[dict]]:
    """Source file to its records, carrying every manifest id not in drop."""
    files: dict[str, list[dict]] = {name: [] for name in MANIFEST["source_files"]}
    for entries in ENTRIES.values():
        for entry in entries:
            if entry["id"] not in (drop or set()):
                files[entry["source_file"]].append(_record(entry))
    return files


def _find(files: dict[str, list[dict]], item_id: str) -> dict:
    """The record for item_id, to be changed in place."""
    return next(
        record
        for records in files.values()
        for record in records
        if f"{record['month']}:{record['no']}" == item_id
    )


def _bench(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, list[dict]] | None = None,
    limit: int | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    calls: list[tuple[str, str, str]] | None = None,
) -> LiveMathematicianBench:
    """A benchmark whose _load serves files (default: every manifest id)."""
    shipped = _files() if files is None else files

    def load(repo_id: str, revision: str, source_file: str) -> list[dict]:
        if calls is not None:
            calls.append((repo_id, revision, source_file))
        return shipped[source_file]

    monkeypatch.setattr(livemathematicianbench, "_load", load)
    bench = build_benchmark(
        BenchmarkCfg(name="livemathematicianbench", limit=limit, system_prompt=system_prompt)
    )
    assert isinstance(bench, LiveMathematicianBench)
    return bench


def _items(bench: LiveMathematicianBench) -> list[Item]:
    """Every item of every split, in split order."""
    return [item for name in SPLITS for item in bench.items(name)]


# --- the split ----------------------------------------------------------------------------------


def test_manifest_ships_the_expected_splits() -> None:
    assert {name: len(ids) for name, ids in IDS.items()} == {"train": 73, "val": 33, "test": 239}
    assert MANIFEST["counts"] == {"train": 73, "val": 33, "test": 239}
    assert len(set(ALL_IDS)) == 345  # no id in two splits
    papers = {name: {entry["paper_link"] for entry in ENTRIES[name]} for name in SPLITS}
    assert not papers["train"] & papers["val"]
    assert not papers["train"] & papers["test"]
    assert not papers["val"] & papers["test"]


def test_splits_are_the_manifest_ids_in_manifest_order(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch)
    for name in SPLITS:
        assert [item.id for item in bench.items(name)] == IDS[name]


def test_only_the_pinned_files_are_read_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    _bench(monkeypatch, calls=calls).items("train")
    revision = MANIFEST["source_revision"]
    assert calls == [(HF_REPO, revision, name) for name in MANIFEST["source_files"]]
    assert MANIFEST["source_repo"] == HF_REPO


def test_unknown_split_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(KeyError, match="unknown split"):
        _bench(monkeypatch).items("validation")


def test_limit_takes_the_prefix_of_every_split(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch, limit=10)
    for name in SPLITS:
        assert [item.id for item in bench.items(name)] == IDS[name][:10]


def test_items_are_grouped_by_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch)
    for name in SPLITS:
        links = [entry["paper_link"] for entry in ENTRIES[name]]
        assert [item.group for item in bench.items(name)] == links


# --- failing loudly -----------------------------------------------------------------------------


def test_missing_manifest_ids_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial split is not this split, so missing ids are an error, not a warning."""
    bench = _bench(monkeypatch, _files(drop={IDS["train"][0], IDS["test"][-1]}))
    with pytest.raises(ValueError, match="2 of 345 manifest id"):
        bench.items("val")


@pytest.mark.parametrize("field", ["month", "no", "paper_link"])
def test_a_manifest_field_disagreeing_with_the_record_fails(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    target = IDS["val"][3]
    real = livemathematicianbench._split_entries

    def entries(split: str) -> list[dict]:
        out = real(split)
        for entry in out:
            if entry["id"] == target:
                entry[field] = "999999" if field == "month" else 999 if field == "no" else "x"
        return out

    monkeypatch.setattr(livemathematicianbench, "_split_entries", entries)
    with pytest.raises(ValueError, match=f"has {field}="):
        _bench(monkeypatch).items("val")


def test_a_record_found_in_another_file_than_the_manifest_says_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = ENTRIES["train"][0]
    files = _files(drop={target["id"]})
    other = next(name for name in files if name != target["source_file"])
    files[other].append(_record(target))
    with pytest.raises(ValueError, match="has source_file="):
        _bench(monkeypatch, files).items("train")


def test_a_source_file_the_manifest_does_not_list_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = livemathematicianbench._split_entries

    def entries(split: str) -> list[dict]:
        out = real(split)
        out[0]["source_file"] = "data/202607/qa_202607_final.json"
        return out

    monkeypatch.setattr(livemathematicianbench, "_split_entries", entries)
    with pytest.raises(ValueError, match="source_files do not list"):
        _bench(monkeypatch).items("train")


def test_an_id_shipped_twice_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _files()
    entry = ENTRIES["test"][0]
    files[entry["source_file"]].append(_record(entry))
    with pytest.raises(ValueError, match="occurs in both"):
        _bench(monkeypatch, files).items("test")


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda mcq: mcq["choices"][0].update(label="F"),
        lambda mcq: mcq["choices"][1].update(label="B"),
        lambda mcq: mcq["choices"].pop(),
        lambda mcq: mcq["choices"].append({"label": "F", "text": "sixth"}),
        lambda mcq: mcq["meta"].update(wildcard_false_label="F"),
    ],
    ids=["unknown label", "duplicate label", "four options", "six options", "meta label"],
)
def test_option_labels_must_be_exactly_a_to_e(
    monkeypatch: pytest.MonkeyPatch, corrupt: object
) -> None:
    files = _files()
    corrupt(_find(files, IDS["train"][5])["mcq"])  # type: ignore[operator]
    with pytest.raises(ValueError, match="option labels"):
        _bench(monkeypatch, files).items("train")


def test_options_are_refused() -> None:
    with pytest.raises(ValueError, match="takes no options"):
        build_benchmark(BenchmarkCfg(name="livemathematicianbench", options={"metric": "em"}))


# --- the permutation ----------------------------------------------------------------------------


def _order(item_id: str) -> str:
    """The source labels of one item's options, in shown order."""
    return "".join(option["label"] for option in present_options(item_id, _source_options()))


def _source_options() -> list[dict]:
    """Five bare options labelled as the source labels them."""
    return [{"label": label} for label in LABELS]


def test_the_permutation_is_the_same_in_a_fresh_process() -> None:
    """hash() is salted per process; the permutation must not be."""
    script = (
        "import json, sys\n"
        "from skillpng.benchmarks.livemathematicianbench.livemathematicianbench import "
        "present_options\n"
        "options = [{'label': label} for label in 'ABCDE']\n"
        "print(json.dumps(["
        "''.join(o['label'] for o in present_options(i, options)) for i in json.load(sys.stdin)"
        "]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(ALL_IDS),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "12345"},
    )
    assert json.loads(done.stdout) == [_order(item_id) for item_id in ALL_IDS]


def test_the_permutation_is_pinned() -> None:
    """The option order is protocol: this fails if the seed or the hash changes."""
    assert [_order(item_id) for item_id in IDS["train"][:5]] == [
        "BDEAC",
        "BCDEA",
        "DCEAB",
        "EBDAC",
        "ABDEC",
    ]


def test_gold_labels_are_spread_over_a_to_e(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every record ships its gold as A; after the permutation no label dominates."""
    bench = _bench(monkeypatch)
    overall = Counter(item.ground_truth for item in _items(bench))
    assert set(overall) == set(LABELS)
    assert max(overall.values()) < 0.3 * 345  # expected 20%, chance-level shortcut only
    for name in SPLITS:
        assert set(Counter(item.ground_truth for item in bench.items(name))) == set(LABELS)


def test_gold_and_meta_labels_follow_their_options(monkeypatch: pytest.MonkeyPatch) -> None:
    for item in _items(_bench(monkeypatch)):
        shown = {option["label"]: option["text"] for option in item.meta["options"]}
        assert list(shown) == list(LABELS)
        assert item.target == f"<answer>{item.ground_truth}</answer>"
        assert shown[item.ground_truth] == f"correct {item.id}"
        assert shown[item.meta["weaker_true_label"]] == f"distractor C {item.id}"
        assert shown[item.meta["wildcard_false_label"]] == f"distractor B {item.id}"
        assert sorted(shown[label] for label in item.meta["false_labels"]) == [
            f"distractor {label} {item.id}" for label in "BDE"
        ]


# --- the prompt ---------------------------------------------------------------------------------


def test_prompt_is_the_system_turn_then_question_and_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]

    clean = bench.build_messages(item, with_image=False)
    assert [message["role"] for message in clean] == ["system", "user"]
    assert clean[0]["content"] == SYSTEM_PROMPT
    assert "<answer>" in SYSTEM_PROMPT

    text = clean[1]["content"][0]["text"]
    options = item.meta["options"]
    choices = "\n".join(f"{option['label']}. {option['text']}" for option in options)
    question = f"Which statement holds for {item.id}?\n\\[\n  x^2 \\ge 0\n\\]"
    # Verbatim but for the ends: the LaTeX newlines and indentation survive.
    assert text == f"## Question\n{question}\n\n## Choices\n{choices}"
    assert choices.startswith("A. ") and "\nE. " in choices
    rendered = json.dumps(clean)
    assert "THEOREM-" not in rendered and "SKETCH-" not in rendered


def test_image_rides_in_the_user_turn_and_leaves_the_input_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]
    clean = bench.build_messages(item, with_image=False)
    before = json.dumps(clean)

    pre = bench.build_messages(item, with_image=True)
    post = bench.build_messages(item, with_image=True, site="post_query")
    assert pre[0]["content"] == clean[0]["content"]  # the system turn stays text-only
    assert pre[1]["content"][0] == {"type": "image"}
    assert post[1]["content"][-1] == {"type": "image"}
    assert pre[1]["content"][1:] == clean[1]["content"]

    messages = [*bench.system_messages(), bench._query(item)]
    snapshot = json.dumps(messages)
    inject_image(messages, "pre_query")
    assert json.dumps(messages) == snapshot
    assert json.dumps(clean) == before


def test_system_prompt_is_configurable_and_can_be_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _bench(monkeypatch, system_prompt="Answer in <answer> tags.")
    messages = bench.build_messages(bench.items("train")[0], with_image=False)
    assert messages[0]["content"] == "Answer in <answer> tags."

    bare = _bench(monkeypatch, system_prompt="")
    messages = bare.build_messages(bare.items("train")[0], with_image=False)
    assert [message["role"] for message in messages] == ["user"]


def test_shipped_config() -> None:
    cfg = load_config(CONFIG)
    assert cfg.benchmark.name == "livemathematicianbench"
    assert cfg.benchmark.options == {}
    assert cfg.train.eval_items == 33
    assert cfg.injection.site == "pre_query"
    assert "<answer>B</answer>" in cfg.benchmark.system_prompt


# --- scoring ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        "<answer>B</answer>",
        "Reasoning.\n<answer> B </answer>",
        "<answer>(B)</answer>",
        "<answer>**B**</answer>",
        "<answer>B.</answer>",
        "<answer>\n**(B)**\n</answer>",
        "<ANSWER>B</ANSWER>",
    ],
)
def test_a_lone_label_scores(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]
    item.ground_truth = "B"
    assert bench.score(output, item) == 1.0


@pytest.mark.parametrize(
    "output",
    [
        "<answer>A</answer>",
        "<answer>B, C</answer>",
        "<answer>B or D</answer>",
        "<answer>BD</answer>",
        "<answer>The answer is B</answer>",
        "<answer>B. The strongest statement.</answer>",
        "<answer></answer>",
        "<answer>F</answer>",
        "B",
        "The answer is B.",
        "<answer>B",
    ],
)
def test_anything_else_scores_zero(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]
    item.ground_truth = "B"
    assert bench.score(output, item) == 0.0


def test_the_target_scores_full_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch)
    assert all(bench.score(item.target, item) == 1.0 for item in _items(bench))


def test_truncated_reasoning_does_not_pass_as_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A budget too small to close the trace leaves only reasoning, which must score 0, even when
    the cut lands on a line that reads like the gold label."""
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]
    item.ground_truth = "A"
    assert bench.score("Thinking Process:\n\n1.  **Analyze the", item) == 0.0
    assert bench.score("Option C fails for n = 2.\nA", item) == 0.0


def test_the_last_tag_overrides_a_draft_one(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _bench(monkeypatch)
    item = bench.items("train")[0]
    drafted = "<think>\nMaybe <answer>A</answer>.\n</think>\n<answer>D</answer>"
    item.ground_truth = "D"
    assert bench.score(drafted, item) == 1.0
    item.ground_truth = "A"
    assert bench.score(drafted, item) == 0.0
    # An empty final tag is not rescued by the draft before it.
    assert extract_label("<answer>A</answer> then <answer></answer>") is None


def test_a_tag_named_while_reasoning_does_not_swallow_the_answer() -> None:
    """Nemotron names the tag in its trace before answering. A block that ran
    from that mention to the real closing tag held no lone label."""
    assert extract_label("Put it in <answer> tags.\n</think>\n<answer>D</answer>") == "D"
    assert extract_label("thought\nUse <answer> tags.\n<answer>D</answer>") == "D"
    assert extract_label("<answer>D</answer>\nThat was the <answer> block.") == "D"


def test_a_draft_inside_the_reasoning_trace_is_never_scored() -> None:
    assert extract_label("<answer>A</answer>\n</think>\nI cannot decide.") is None
