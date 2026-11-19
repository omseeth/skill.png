"""T1 adapter: shipped artifacts, prompt, per-turn metrics, the worker, items, report.

No network, no weights, no GPU and no 1.35 GB ontology. The vendored tools are
only ever imported in a subprocess: the executor's worker, run here on
tests/fixtures/t1, the first 60 rows of every ontology CSV. The item tests
build a synthetic dataset in a temporary directory with the Hub's layout and
point the adapter at it; the shipped-artifact tests read the committed
manifest, items and prompt straight off disk.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from skillpng.benchmarks.base import Item, _unit_hash
from skillpng.benchmarks.registry import build_benchmark
from skillpng.benchmarks.t1 import t1, t1_build
from skillpng.benchmarks.t1.t1 import (
    METRICS,
    T1,
    T1Options,
    build_target,
    cache_exact_match,
    extract_code,
    fill_prompt,
    generated_calls,
    load_template,
    non_executable,
    turn_metrics,
)
from skillpng.benchmarks.t1.t1_build import (
    CACHE_SLOT,
    COLUMNS,
    CONVERSATION_SLOT,
    DOMAINS,
    HF_SPLITS,
    ITEM_FIELDS,
    NEAREST,
    NEAREST_UPSTREAM,
    OUTPUT_FORMAT,
    PROMPT_PATH,
    SAVE_RULE,
    SAVE_RULE_UPSTREAM,
    SHIPPED,
    SHIPPED_DIGESTS,
    SHIPPED_KINDS,
    SPLIT_ROOT,
    SPLITS,
    SYSTEM_PROMPT,
    dialogue_file,
    digest,
    ids_digest,
    item_id,
    parse_item_id,
)
from skillpng.benchmarks.t1.t1_exec import Executor, Row
from skillpng.benchmarks.t1.t1_report import build_report, read_eval, render, upstream_metrics
from skillpng.config import BenchmarkCfg, load_config

ROOT = Path(__file__).parent.parent
T1_CONFIG = ROOT / "configs" / "t1.yaml"
FIXTURE = Path(__file__).parent / "fixtures" / "t1"

HOTELS = (
    'hotels = search_hotels(city="New York", checkin_date=["May 10, 2025"], '
    'checkout_date=["May 12, 2025"])\nsave_to_cache(key="hotels", value=hotels)'
)
RESTAURANTS = (
    'restaurants = search_restaurants(city="New York", cuisine=["Ethiopian"])\n'
    'save_to_cache(key="restaurants", value=restaurants)'
)
FILTER = (
    "filtered_hotels = filter_hotels(prior_results=hotels, stars=[4, 4])\n"
    'save_to_cache(key="filtered_hotels", value=filtered_hotels)'
)
SEEK = 'seek_information("We need the check-in and check-out dates.")'
# One synthetic dialogue: an empty row, an assistant row with a plan, a turn
# reading a variable an earlier turn bound, and a sign-off without a plan.
DIALOGUE: list[Row] = [
    ["assistant: Hi, how can I help?", None],
    ["user: I need a hotel in New York {for real}", SEEK],
    [None, None],
    ["assistant: When are you staying?", None],
    ["user: From May 10, 2025 to May 12, 2025", HOTELS],
    ["assistant: Found some. Want food too?", RESTAURANTS],
    ["user: Only 4 star hotels please", FILTER],
    ["user: Thanks!", None],
]
# Turn to its gold kind and plan.
TURNS = {0: ("seek_only", SEEK), 1: ("tools", HOTELS), 2: ("tools", FILTER)}


def _dict(value: object) -> dict[str, object]:
    """A report or record value, narrowed to the mapping it is."""
    assert isinstance(value, dict)
    return value


# -------------------------------------------------------- the synthetic corpus


def _write_dialogues(path: Path, dialogues: dict[str, list[Row]]) -> None:
    """Write one dialogue CSV in the Hub's column layout."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(COLUMNS)
        index = 0

        for template_id, rows in dialogues.items():
            for template, plan in rows:
                writer.writerow([index, template_id, "", "", template or "", plan or ""])
                index += 1


@dataclass
class Corpus:
    """A synthetic dataset and manifest the adapter is pointed at."""

    data: Path
    root: Path
    splits: dict[str, list[dict[str, object]]]
    manifest: dict[str, object]

    def write(self) -> None:
        """Write the manifest and the items files as they stand."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "split_manifest.json").write_text(json.dumps(self.manifest))

        for name, entries in self.splits.items():
            (self.root / name).mkdir(parents=True, exist_ok=True)
            (self.root / name / "items.json").write_text(json.dumps(entries))


def _entry(
    domain: str, hf_split: str, n: int, template_id: str, turn: int, cache: str
) -> dict[str, object]:
    """A manifest entry for one turn of the synthetic dialogue."""
    kind, plan = TURNS.get(turn, ("tools", FILTER))

    return {
        "id": item_id(domain, hf_split, n, template_id, turn),
        "domain": domain,
        "hf_split": hf_split,
        "file_index": n,
        "template_id": template_id,
        "turn": turn,
        "gold_kind": kind,
        "plan_digest": digest(plan),
        "cache_digest": digest(cache),
    }


@pytest.fixture(scope="module")
def turn2_cache() -> str:
    """Turn 2's pre-turn cache: what turns 0-1 and the assistant row saved."""
    probe = Executor(str(FIXTURE))

    try:
        return probe.build(DIALOGUE[:7]).cache
    finally:
        probe.close()


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, turn2_cache: str) -> Corpus:
    """A synthetic dataset and manifest, with the adapter pointed at them.

    Template ID 3 occurs in two files, as every T1 ID does. Template 7 in
    file 2 has a gold plan that raises before its turn 2.
    """
    data = tmp_path / "data"
    data.mkdir()
    (data / "ontology").symlink_to(FIXTURE / "ontology", target_is_directory=True)
    broken = [row[:] for row in DIALOGUE]
    broken[4] = [broken[4][0], "hotels = search_hotels(city=undefined_name)"]
    _write_dialogues(data / dialogue_file("hotel", "train", 1), {"3": DIALOGUE})
    _write_dialogues(data / dialogue_file("hotel", "train", 2), {"3": DIALOGUE, "7": broken})
    _write_dialogues(data / dialogue_file("hotel", "validation", 1), {"9": DIALOGUE})
    _write_dialogues(data / dialogue_file("hotel", "test", 1), {"5": DIALOGUE})
    splits = {
        "train": [
            _entry("hotel", "train", 1, "3", 0, "{}"),
            _entry("hotel", "train", 2, "3", 1, "{}"),
            _entry("hotel", "train", 1, "3", 2, turn2_cache),
        ],
        "val": [_entry("hotel", "validation", 1, "9", 1, "{}")],
        "test": [
            _entry("hotel", "test", 1, "5", 2, turn2_cache),
            _entry("hotel", "test", 1, "5", 0, "{}"),
        ],
    }
    manifest: dict[str, object] = {
        "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    }
    built = Corpus(data=data, root=tmp_path / "split", splits=splits, manifest=manifest)
    built.write()
    monkeypatch.setattr(t1, "SPLIT_ROOT", built.root)
    monkeypatch.setattr(t1, "snapshot", lambda files, repo_id="": data)
    monkeypatch.setattr(t1, "ontology_root", lambda repo_id="": data)
    monkeypatch.setattr(t1, "ITEM_CACHE_ROOT", tmp_path / "item-cache")

    return built


def _bench(options: dict[str, object] | None = None, limit: int | None = None) -> T1:
    """The adapter with the shipped system prompt."""
    return T1(
        BenchmarkCfg(name="t1", system_prompt=SYSTEM_PROMPT, options=options or {}, limit=limit)
    )


def _item(conversation: str = "[{'user': ' hi'}]", cache: str = "{}") -> Item:
    """A hand-made item, enough for prompt tests that execute nothing."""
    return Item(
        id="hotel:test:1:5:0",
        target=build_target(SEEK),
        ground_truth=SEEK,
        group="hotel:5",
        meta={"question": "hi", "conversation": conversation, "cache": cache, "replay": []},
    )


def _user_text(messages: list[dict[str, object]]) -> str:
    """Concatenate the text blocks of the last user turn."""
    content = messages[-1]["content"]
    assert isinstance(content, list)

    return "".join(block["text"] for block in content if block["type"] == "text")


def _by_turn(items: list[Item]) -> dict[object, Item]:
    """Items keyed by their turn."""
    return {item.meta["turn"]: item for item in items}


class Scripted:
    """A fake VLM giving the same answer to every prompt."""

    def __init__(self, answer: str) -> None:
        """Remember the answer."""
        self.answer = answer

    def generate(self, messages: list[dict[str, object]], image: object = None) -> str:
        """Answer."""
        return self.answer


# --------------------------------------------------------------------- options


def test_options_defaults() -> None:
    parsed = T1Options.parse({})

    assert parsed.metric == "cache_em"
    assert parsed.exec_timeout_s == 30.0


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown t1 option"):
        T1Options.parse({"few_shot": True})


@pytest.mark.parametrize(
    "options",
    [
        {"metric": "bleu"},
        {"exec_timeout_s": 0},
        {"exec_timeout_s": -1},
        {"exec_timeout_s": True},
        {"exec_timeout_s": "30"},
    ],
)
def test_bad_options_are_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        T1Options.parse(options)


@pytest.mark.parametrize("metric", METRICS)
def test_every_metric_has_a_reporting_name(metric: str) -> None:
    assert _bench({"metric": metric}).metric_name


def test_registry_builds_the_adapter() -> None:
    bench = build_benchmark(BenchmarkCfg(name="t1", system_prompt=SYSTEM_PROMPT))

    assert isinstance(bench, T1)


# ------------------------------------------------------------------------- ids


def test_item_ids_round_trip() -> None:
    value = item_id("flighthotel", "test", 17, "23", 2)

    assert value == "flighthotel:test:17:23:2"
    assert parse_item_id(value) == {
        "domain": "flighthotel",
        "hf_split": "test",
        "file_index": 17,
        "template_id": "23",
        "turn": 2,
    }


def test_ids_stay_unique_when_template_ids_repeat_across_files() -> None:
    ids = {item_id("hotel", "train", n, "3", 0) for n in range(1, 26)}

    assert len(ids) == 25


# --------------------------------------------------------------- shipped split


def _shipped() -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    """The committed manifest and items files."""
    manifest = json.loads((SPLIT_ROOT / "split_manifest.json").read_text())
    items = {s: json.loads((SPLIT_ROOT / s / "items.json").read_text()) for s in SPLITS}

    return manifest, items


def test_shipped_counts_and_digests_match_the_items() -> None:
    manifest, items = _shipped()

    for split, entries in items.items():
        column = SPLITS.index(split)
        ids = [str(entry["id"]) for entry in entries]

        assert _dict(manifest["counts"])[split] == len(entries) == (400, 200, 1400)[column]
        assert _dict(manifest["digests"])[split] == ids_digest(ids) == SHIPPED_DIGESTS[split]
        assert len(set(ids)) == len(ids)

        for domain in DOMAINS:
            count = sum(entry["domain"] == domain for entry in entries)
            per_domain = _dict(_dict(manifest["per_domain"])[split])
            assert per_domain[domain] == count == SHIPPED[domain][column]

        kinds = Counter(str(entry["gold_kind"]) for entry in entries)
        assert kinds == {kind: counts[column] for kind, counts in SHIPPED_KINDS.items()}
        assert _dict(manifest["gold_kinds"])[split] == dict(sorted(kinds.items()))


def test_shipped_items_carry_exactly_the_declared_fields() -> None:
    manifest, items = _shipped()

    assert manifest["item_fields"] == ITEM_FIELDS

    for split, entries in items.items():
        for entry in entries:
            assert list(entry) == ITEM_FIELDS
            assert parse_item_id(str(entry["id"])) == {
                name: entry[name]
                for name in ("domain", "hf_split", "file_index", "template_id", "turn")
            }
            assert entry["hf_split"] == HF_SPLITS[split]
            assert len(str(entry["plan_digest"])) == len(str(entry["cache_digest"])) == 16


def test_no_dialogue_appears_twice() -> None:
    _, items = _shipped()
    dialogues = [
        (e["domain"], e["hf_split"], e["file_index"], e["template_id"])
        for entries in items.values()
        for e in entries
    ]

    assert len(set(dialogues)) == len(dialogues) == 2000


def test_shipped_items_are_in_canonical_order() -> None:
    _, items = _shipped()

    for entries in items.values():
        keys = [_unit_hash(f"42:order:{entry['id']}") for entry in entries]
        assert keys == sorted(keys)


def test_the_dropped_template_contributes_nothing() -> None:
    manifest, items = _shipped()

    assert manifest["dropped_templates"] == ["flighthotelrestaurant:test:26"]
    assert not any(
        e["domain"] == "flighthotelrestaurant" and e["template_id"] == "26" for e in items["test"]
    )


def test_the_manifest_records_the_protocol() -> None:
    manifest, _ = _shipped()

    assert manifest["source_revision"] == t1_build.HF_REVISION
    assert manifest["vendored_commit"] == t1_build.UPSTREAM_COMMIT
    assert manifest["seed"] == 42
    assert "session" in str(manifest["replay_semantics"])
    assert len(manifest["eligibility"]) == 3  # type: ignore[arg-type]
    assert manifest["notes"]


# ---------------------------------------------------------------------- prompt


def test_the_shipped_template_matches_its_digest() -> None:
    manifest, _ = _shipped()

    assert hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest() == manifest["prompt_sha256"]
    assert load_template() == PROMPT_PATH.read_text()


def test_the_template_is_upstreams_user_text_with_the_four_edits() -> None:
    template = load_template()

    assert template.startswith("Your task is to generate a Python code")
    assert template.endswith("generate a Python code for the last user turn.")
    assert template.count(CONVERSATION_SLOT) == template.count(CACHE_SLOT) == 1
    # Edit 1: no few-shot block.
    assert "# EXAMPLES" not in template
    # Edit 2: code only, with the conventions the few-shot block showed.
    assert "# OUTPUT FORMAT\n" + OUTPUT_FORMAT + "\n\n# CONVERSATION\n" in template
    assert "reasoning" not in template.split("# OUTPUT FORMAT")[1]
    # Edit 3: save every result, by keyword.
    assert SAVE_RULE_UPSTREAM not in template
    assert template.count(SAVE_RULE) == 1
    # Edit 4: find_nearest, which T1 never defines, is not advertised.
    assert NEAREST_UPSTREAM not in template
    assert template.count(NEAREST) == 2
    assert "# TOOL CONFIG\n" in template
    assert template.count("Tool Number:") == 15


def test_the_templates_trailing_whitespace_survives() -> None:
    template = load_template()
    line = "check CONVERSATION to know what parameters have been provided by the user. \n"

    assert line in template
    assert "#############  \n" in template


def test_a_template_that_lost_its_whitespace_is_refused(tmp_path: Path) -> None:
    stripped = tmp_path / "t1_prompt.txt"
    stripped.write_text("\n".join(line.rstrip() for line in PROMPT_PATH.read_text().split("\n")))

    with pytest.raises(ValueError, match="trailing whitespace"):
        load_template(stripped)


def test_filling_is_one_pass_and_leaves_braces_alone() -> None:
    template = f"A {CONVERSATION_SLOT} B {CACHE_SLOT} C {{literal}}"
    text = fill_prompt(template, f"say {{x}} and {CACHE_SLOT}", "{'k': 'v'}")

    assert text == f"A say {{x}} and {CACHE_SLOT} B {{'k': 'v'}} C {{literal}}"


def test_the_system_prompt_is_upstreams_text() -> None:
    cfg = load_config(T1_CONFIG)

    assert cfg.benchmark.system_prompt == SYSTEM_PROMPT
    assert not SYSTEM_PROMPT.endswith((" ", "\n"))


# -------------------------------------------------------------------- messages


def test_turns_are_system_then_one_user() -> None:
    messages = _bench().build_messages(_item(), with_image=False)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_no_system_prompt_means_no_system_turn() -> None:
    bench = T1(BenchmarkCfg(name="t1", system_prompt=""))

    assert [m["role"] for m in bench.build_messages(_item(), with_image=False)] == ["user"]


def test_the_user_turn_is_the_template_filled_with_the_item() -> None:
    item = _item("[{'user': ' I need {a hotel}'}]", "{'hotels': 'x'}")
    text = _user_text(_bench().build_messages(item, with_image=False))

    assert text == fill_prompt(load_template(), str(item.meta["conversation"]), "{'hotels': 'x'}")
    assert "# CONVERSATION\n[{'user': ' I need {a hotel}'}]\n\n# CACHE\n{'hotels': 'x'}\n" in text


def test_the_image_rides_in_the_user_turn_at_the_requested_site() -> None:
    bench = _bench()

    for site, position in (("pre_query", 0), ("post_query", -1)):
        messages = bench.build_messages(_item(), with_image=True, site=site)
        system, user = messages[0]["content"], messages[-1]["content"]
        assert isinstance(system, str) and isinstance(user, list)

        assert user[position]["type"] == "image"
        assert sum(block["type"] == "image" for block in user) == 1


def test_the_text_is_byte_identical_with_and_without_the_image() -> None:
    bench = _bench()
    item = _item("[{'user': ' a'}, {'assistant': ' b'}]", "{'k': 'v'}")
    plain = _user_text(bench.build_messages(item, with_image=False))

    for site in ("pre_query", "post_query"):
        assert _user_text(bench.build_messages(item, with_image=True, site=site)) == plain


# ------------------------------------------------------- target and extraction


def test_the_target_is_the_stripped_plan_in_one_code_block() -> None:
    assert build_target("  x = 1\n\n") == "<CODE>\nx = 1\n</CODE>"
    assert build_target("   ") == "<CODE>\n\n</CODE>"


def test_the_last_code_block_wins_stripped() -> None:
    output = "plan:\n<CODE>a = 1</CODE> then <CODE>\n b = 2 \n</CODE>"

    assert extract_code(output) == ("b = 2", 2, False)


def test_a_draft_in_a_trace_without_an_end_never_runs() -> None:
    """vLLM strips Gemma 4's channel tokens, so its trace reaches the
    extractor as a bare thought prefix with no </think> after it."""
    output = "thought\nFirst <CODE>search_hotels(city='A')</CODE>, no.\n<CODE>y()</CODE>"

    assert extract_code(output) == ("y()", 2, False)


def test_a_tag_named_while_reasoning_does_not_swallow_the_code() -> None:
    """A block that ran from the mention to the real closing tag ran the
    trace as code."""
    assert extract_code("Wrap it in <CODE> tags.\n</think>\n<CODE>y()</CODE>")[0] == "y()"
    assert extract_code("thought\nUse <CODE> tags.\n<CODE>y()</CODE>")[0] == "y()"
    assert extract_code("<CODE>y()</CODE>\nThat was the <CODE> block.")[0] == "y()"


def test_no_code_block_is_no_code() -> None:
    assert extract_code("I would search hotels.") == ("", 0, False)


def test_a_reasoning_trace_is_dropped_up_to_its_last_end() -> None:
    output = "<think>draft <CODE>x()</CODE></think>more</think><CODE>y()</CODE>"

    assert extract_code(output) == ("y()", 1, True)


def test_the_target_round_trips_through_the_extractor() -> None:
    assert extract_code(build_target(HOTELS))[0] == HOTELS


# ------------------------------------------------------------ per-turn metrics


def test_the_oracle_scores_every_component_at_its_maximum() -> None:
    after: dict[str, object] = {"hotels": "summary"}
    out = turn_metrics(HOTELS, HOTELS, "success", after, dict(after))

    assert (out["tool_tp"], out["tool_fp"], out["tool_fn"]) == (2, 0, 0)
    assert (out["param_tp"], out["param_fp"], out["param_fn"]) == (3, 0, 0)
    assert out["tool_f1"] == 1.0
    assert out["non_executable"] == 0 and out["code_success"] == 1.0
    assert out["cache_em"] == 1.0
    assert not out["no_calls"]


def test_a_lone_print_is_no_calls_and_leaves_the_tool_metrics_out() -> None:
    out = turn_metrics(HOTELS, 'print("No planning needed")', "success", {"h": "s"}, {})

    assert generated_calls('print("No planning needed")') == (None, "")
    assert out["tool_tp"] is None and out["param_tp"] is None
    assert out["no_calls"] and out["non_executable"] == 1
    assert out["tool_f1"] == 0.0 and out["cache_em"] == 0.0


def test_two_prints_are_an_empty_call_list_that_is_scored() -> None:
    """Upstream only drops a *lone* print; two leave [], which is not None."""
    out = turn_metrics(HOTELS, 'print("a")\nprint("b")', "success", {"h": "s"}, {})

    assert generated_calls('print("a")\nprint("b")') == ([], "")
    assert (out["tool_tp"], out["tool_fp"], out["tool_fn"]) == (0, 0, 2)
    assert out["no_calls"] and out["non_executable"] == 1


def test_other_prints_are_dropped() -> None:
    calls, _ = generated_calls(HOTELS + '\nprint("done")')

    assert [next(iter(call)) for call in calls or []] == ["search_hotels", "save_to_cache"]


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        ("x = (", "SyntaxError"),
        # Upstream would crash on this one: it only catches SyntaxError.
        ("f(x=" + "1+" * 3000 + "1)", "RecursionError"),
        ("", ""),
    ],
    ids=["syntax", "recursion", "empty"],
)
def test_code_that_does_not_parse_is_no_calls(code: str, kind: str) -> None:
    assert generated_calls(code) == (None, kind)
    assert turn_metrics(HOTELS, code, "success", {}, {})["parse_error"] == kind


def test_non_executable_follows_upstream() -> None:
    gold: list[dict[str, object]] = [{"search_hotels": {}}]

    assert non_executable(gold, None, "success") == 1
    assert non_executable(gold, [], "success") == 1
    assert non_executable(gold, gold, "name 'x' is not defined") == 1
    assert non_executable(gold, gold, "success") == 0
    # A gold plan without calls (the three whitespace-only train items) is
    # never non-executable.
    assert non_executable([], None, "boom") == 0


def test_cache_match_ignores_keys_and_counts_values() -> None:
    assert cache_exact_match({}, {}) == 1
    assert cache_exact_match(None, {}) == 1
    assert cache_exact_match({"a": "x"}, {"b": "x"}) == 1
    assert cache_exact_match({"a": "x", "b": "x"}, {"c": "x"}) == 0
    assert cache_exact_match({"a": "x"}, {}) == 0


def test_a_generation_that_did_not_finish_has_no_cache_match() -> None:
    out = turn_metrics(SEEK, "while True: pass", "timeout", {}, None, executed=False)

    assert out["cache_em"] == 0.0
    assert out["non_executable"] == 1


def test_a_whitespace_gold_plan_has_no_calls() -> None:
    out = turn_metrics("   ", 'print("No planning needed")', "success", {}, {})

    assert out["n_gold_calls"] == 0
    assert out["tool_tp"] is None
    assert out["non_executable"] == 0 and out["cache_em"] == 1.0


def test_tool_f1_is_computed_from_the_counts() -> None:
    code = 'hotels = search_hotels(city="New York")\nsearch_hotels(city="x")'
    out = turn_metrics(HOTELS, code, "success", {}, {})

    # Gold: search_hotels, save_to_cache. Generated: search_hotels twice.
    assert (out["tool_tp"], out["tool_fp"], out["tool_fn"]) == (1, 1, 1)
    assert out["tool_f1"] == pytest.approx(0.5)


# ------------------------------------------------------------------ the worker


@pytest.fixture(scope="module")
def executor() -> Iterator[Executor]:
    """One worker on the fixture ontology, shared by the worker tests."""
    worker = Executor(str(FIXTURE))
    yield worker
    worker.close()


PREFIX = [plan for template, plan in DIALOGUE[:6] if template is not None and plan is not None]


def test_the_tools_import_on_the_fixture(executor: Executor) -> None:
    run = executor.execute(PREFIX, FILTER, FILTER, 30)

    assert run.gold_error == run.error == "success"
    assert set(run.gold_after) == {"hotels", "restaurants", "filtered_hotels"}
    assert run.after == run.gold_after


def test_a_timeout_kills_the_worker_and_the_next_item_is_unaffected(executor: Executor) -> None:
    before = executor.execute(PREFIX, FILTER, FILTER, 30)
    starts = executor.starts
    run = executor.execute(PREFIX, FILTER, "while True:\n    pass", 1.5)

    assert run.error == "timeout" and run.timeout and not run.executed
    assert run.after is None
    assert run.gold_after == before.gold_after
    assert executor.execute(PREFIX, FILTER, FILTER, 30) == before
    assert executor.starts == starts + 1


def test_input_raises_instead_of_waiting(executor: Executor) -> None:
    run = executor.execute([], SEEK, "answer = input()", 10)

    assert (run.error, run.error_type) == ("input(): lost sys.stdin", "RuntimeError")


def test_an_exception_is_recorded_as_its_text(executor: Executor) -> None:
    run = executor.execute([], SEEK, 'raise ValueError("no such city")', 10)

    assert (run.error, run.error_type) == ("no such city", "ValueError")
    assert run.executed and run.after == {}


def test_sys_exit_is_the_codes_outcome_not_the_workers(executor: Executor) -> None:
    pid = executor.pid
    run = executor.execute([], SEEK, "import sys\nsys.exit(3)", 10)

    assert (run.error, run.error_type) == ("3", "SystemExit")
    assert executor.pid == pid


def test_a_dead_worker_is_recorded_and_replaced(executor: Executor) -> None:
    run = executor.execute([], SEEK, "import os\nos._exit(7)", 10)

    assert run.error == "worker died (exit code 7)" and not run.executed and not run.timeout
    assert executor.execute([], SEEK, "", 10).error == "success"


def test_empty_code_is_not_executed_and_succeeds(executor: Executor) -> None:
    run = executor.execute(PREFIX, FILTER, "", 10)

    assert run.error == "success" and run.executed
    assert set(run.after or {}) == {"hotels", "restaurants"}


def test_a_stdout_flood_is_discarded(executor: Executor, capfd: pytest.CaptureFixture[str]) -> None:
    run = executor.execute([], SEEK, 'for _ in range(20000):\n    print("x" * 1000)', 30)
    captured = capfd.readouterr()

    assert run.error == "success"
    assert "xxxx" not in captured.out + captured.err


def test_generated_code_cannot_change_the_gold_state(executor: Executor) -> None:
    clean = executor.execute(PREFIX, FILTER, "", 10)
    vandal = (
        'get_results_from_cache(key="hotels").search_results.clear()\n'
        'hotels.query_summary = "gone"\n'
        "import builtins\nbuiltins.len = None\n__builtins__['sorted'] = None"
    )
    run = executor.execute(PREFIX, FILTER, vandal, 10)

    assert run.gold_after == clean.gold_after
    # The variable and the cache entry were one object, and still are.
    assert _dict(run.after)["hotels"] == "gone"
    # Rebinding a real builtin is undone before anything else runs.
    assert run.restored == ["builtins.len"]
    assert executor.execute(PREFIX, FILTER, FILTER, 10).after == clean.gold_after


def test_no_name_leaks_from_one_item_to_the_next(executor: Executor) -> None:
    executor.execute([], SEEK, "leaked = 1", 10)

    assert executor.execute([], SEEK, "leaked", 10).error_type == "NameError"


def test_locals_are_not_visible_inside_a_lambda_as_upstream(executor: Executor) -> None:
    run = executor.execute([], SEEK, "x = 1\nf = lambda: x\nf()", 10)

    assert (run.error, run.error_type) == ("name 'x' is not defined", "NameError")


def test_find_nearest_is_advertised_but_not_callable_as_upstream(executor: Executor) -> None:
    assert executor.execute([], SEEK, "find_nearest(hotels=None)", 10).error_type == "NameError"


def test_a_replay_that_raises_is_an_error_unless_not_strict(executor: Executor) -> None:
    with pytest.raises(RuntimeError, match="gold replay raised NameError"):
        executor.execute(["undefined_name"], SEEK, "", 10)

    assert executor.execute(["undefined_name"], SEEK, "", 10, strict=False).error == "success"


# ----------------------------------------------------------------------- items


def test_items_are_built_from_the_manifest(corpus: Corpus) -> None:
    bench = _bench()

    assert [i.id for i in bench.items("train")] == [e["id"] for e in corpus.splits["train"]]
    assert len(bench.items("val")) == 1 and len(bench.items("test")) == 2
    bench.close()


def test_the_history_holds_utterances_never_code(corpus: Corpus) -> None:
    item = _by_turn(_bench().items("train"))[2]
    history = [
        {"assistant": " Hi, how can I help?"},
        {"user": " I need a hotel in New York {for real}"},
        {"assistant": " When are you staying?"},
        {"user": " From May 10, 2025 to May 12, 2025"},
        {"assistant": " Found some. Want food too?"},
        {"user": " Only 4 star hotels please"},
    ]

    assert item.meta["conversation"] == f"{history}"
    assert "search_hotels" not in str(item.meta["conversation"])
    assert item.meta["question"] == "Only 4 star hotels please"


def test_the_cache_is_taken_before_the_turns_gold_runs(corpus: Corpus) -> None:
    item = _by_turn(_bench().items("train"))[1]

    # Turn 1's own gold saves hotels; the prompt must not show it.
    assert item.meta["cache"] == "{}"
    assert item.meta["replay"] == [SEEK]


def test_assistant_row_gold_executes(corpus: Corpus) -> None:
    item = _by_turn(_bench().items("train"))[2]

    assert "'restaurants':" in str(item.meta["cache"])
    assert item.meta["replay"] == [SEEK, HOTELS, RESTAURANTS]


def test_empty_rows_are_skipped(corpus: Corpus) -> None:
    items = _by_turn(_bench().items("train"))

    assert "None" not in str(items[2].meta["conversation"])
    assert items[1].meta["question"] == "From May 10, 2025 to May 12, 2025"


def test_item_fields(corpus: Corpus) -> None:
    item = _by_turn(_bench().items("train"))[1]

    assert item.target == build_target(HOTELS)
    assert item.ground_truth == HOTELS
    assert item.group == "hotel:3"
    assert item.meta["gold_kind"] == "tools"
    json.dumps(item.meta)


def test_a_second_construction_reads_the_cache_without_a_worker(corpus: Corpus) -> None:
    first = _bench()
    built = first.items("train")
    first.close()
    second = _bench()
    items = second.items("train")

    assert second._executor is None
    assert [i.meta for i in items] == [i.meta for i in built]


def test_limit_caps_every_split_in_manifest_order(corpus: Corpus) -> None:
    bench = _bench(limit=1)

    assert [i.id for i in bench.items("train")] == [corpus.splits["train"][0]["id"]]
    assert len(bench.items("test")) == 1


def test_a_digest_that_moved_is_refused(corpus: Corpus) -> None:
    corpus.splits["test"][0]["cache_digest"] = "0" * 16
    corpus.write()

    with pytest.raises(ValueError, match="the manifest says"):
        _bench().items("test")


def test_a_turn_whose_dialogue_raises_is_refused(corpus: Corpus) -> None:
    corpus.splits["val"] = [_entry("hotel", "train", 2, "7", 2, "{}")]
    corpus.write()

    with pytest.raises(ValueError, match="rule 2"):
        _bench().items("val")


def test_a_manifest_id_the_data_lacks_is_refused(corpus: Corpus) -> None:
    corpus.splits["val"] = [_entry("hotel", "validation", 1, "9", 9, "{}")]
    corpus.write()

    with pytest.raises(ValueError, match="partial split"):
        _bench().items("val")


def test_the_oracle_rollout_scores_the_maximum(corpus: Corpus) -> None:
    bench = _bench()

    for item in bench.items("train"):
        result = asyncio.run(bench.rollout(item, Scripted(item.target)))

        assert result.score == 1.0
        assert result.meta["non_executable"] == 0
        assert result.meta["tool_fp"] == result.meta["tool_fn"] == 0

    bench.close()


def test_the_do_nothing_rollout_matches_only_when_gold_leaves_the_cache(corpus: Corpus) -> None:
    bench = _bench()
    dummy = Scripted('<CODE>print("No planning needed")</CODE>')

    for item in bench.items("train"):
        result = asyncio.run(bench.rollout(item, dummy))

        assert result.score == float(item.meta["gold_kind"] == "seek_only")
        assert result.meta["code_success"] == 0.0 and result.meta["no_calls"]

    bench.close()


# ----------------------------------------------------------------- import rule


def test_importing_skillpng_never_imports_the_tools() -> None:
    code = (
        "import sys, skillpng.benchmarks, skillpng.benchmarks.t1.t1_build\n"
        "import skillpng.benchmarks.t1.t1_report\n"
        "print([m for m in sys.modules if '.upstream.tools' in m or '.upstream.planner' in m])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------- config


def test_shipped_config_sets_budgets_the_items_need() -> None:
    cfg = load_config(T1_CONFIG)

    assert cfg.benchmark.name == "t1"
    assert T1Options.parse(cfg.benchmark.options).metric == "cache_em"
    # Measured with the Qwen3.5-9B tokenizer: target max 277, prompt + target
    # max 13,715, plus up to 256 image tokens. The 512 / 10240 defaults of
    # base.yaml would refuse every item.
    assert (cfg.model.max_target_tokens or 0) >= 277
    assert (cfg.model.max_sequence_tokens or 0) >= 13715 + 256
    assert cfg.model.max_new_tokens == 4000
    assert cfg.train.eval_items == 200
    assert cfg.train.run_dir == "runs/t1"


def test_the_analysis_table_knows_t1() -> None:
    sys.path.insert(0, str(ROOT / "analysis"))

    try:
        analyze_runs = importlib.import_module("analyze_runs")
    finally:
        sys.path.pop(0)

    assert analyze_runs.BENCHMARKS["t1"] == "T1"


# ---------------------------------------------------------------------- report


def _eval_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a per-item JSONL in the shape evaluate.py produces."""
    base: dict[str, object] = {
        "domain": "hotel",
        "hf_split": "test",
        "file_index": 1,
        "gold_kind": "tools",
        "tool_tp": 1,
        "tool_fp": 0,
        "tool_fn": 0,
        "param_tp": 2,
        "param_fp": 0,
        "param_fn": 0,
        "non_executable": 0,
        "code_success": 1.0,
        "cache_em": 1.0,
        "tool_f1": 1.0,
        "no_calls": False,
        "parse_error": "",
        "timeout": False,
        "executed": True,
        "n_blocks": 1,
    }
    lines = []

    for index, row in enumerate(rows):
        meta = {**base, **row}
        lines.append(
            json.dumps(
                {
                    "item_id": f"i{index}",
                    "target": "t",
                    "output": "o",
                    "score": meta["cache_em"],
                    "meta": meta,
                }
            )
        )

    path.write_text("\n".join(lines) + "\n")

    return path


def _summary(path: Path) -> dict[str, object]:
    """The one condition a single-file report holds."""
    conditions = _dict(build_report(path)["conditions"])

    return _dict(next(iter(conditions.values())))


def test_code_success_is_grouped_by_file_then_averaged(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [{"file_index": 1, "non_executable": 1, "code_success": 0.0}]
    rows += [{"file_index": 2} for _ in range(3)]
    summary = _summary(_eval_rows(tmp_path / "a.jsonl", rows))

    assert _dict(_dict(summary["per_domain"])["hotel"])["code_success_rate"] == 0.5
    assert summary["code_success_per_item"] == 0.75


def test_micro_metrics_sum_counts_and_skip_turns_without_them(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {"tool_tp": 1, "tool_fp": 1, "tool_fn": 0},
        {"tool_tp": 1, "tool_fp": 0, "tool_fn": 2},
        {"tool_tp": None, "tool_fp": None, "tool_fn": None, "non_executable": 1, "cache_em": 0.0},
    ]
    metrics = upstream_metrics(list(read_eval(_eval_rows(tmp_path / "a.jsonl", rows)).values()))

    assert metrics["tool_calling_micro_precision"] == pytest.approx(2 / 3)
    assert metrics["tool_calling_micro_recall"] == pytest.approx(2 / 4)
    assert metrics["tool_calling_micro_accuracy"] == pytest.approx(2 / 5)
    assert metrics["cache_summary_exact_match_avg"] == pytest.approx(2 / 3)
    assert metrics["exec_only_cache_summary_exact_match_avg"] == 1.0


def test_the_macro_average_is_over_domains(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [{"domain": "hotel", "cache_em": 1.0}]
    rows += [{"domain": "flight", "cache_em": 0.0} for _ in range(3)]
    summary = _summary(_eval_rows(tmp_path / "a.jsonl", rows))

    assert _dict(summary["macro"])["cache_summary_exact_match_avg"] == 0.5
    assert summary["headline"] == 0.25


def test_the_report_prints_the_side_columns(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {"no_calls": True, "gold_kind": "seek_only"},
        {"timeout": True, "executed": False},
    ]
    text = render(build_report(_eval_rows(tmp_path / "a.jsonl", rows)))

    for needle in ("macro avg", "no tool calls", "timed out", "by gold_kind", "seek_only"):
        assert needle in text


def test_the_report_pairs_on_item_id(tmp_path: Path) -> None:
    base = _eval_rows(tmp_path / "a.jsonl", [{"cache_em": 0.0}, {"cache_em": 1.0}])
    treat = _eval_rows(tmp_path / "b.jsonl", [{"cache_em": 1.0}, {"cache_em": 1.0}])
    delta = _dict(build_report(base, treat, draws=200)["paired"])

    assert delta["delta"] == 0.5 and delta["wins"] == 1 and delta["losses"] == 0
    assert float(delta["ci_low"]) <= 0.5 <= float(delta["ci_high"])  # type: ignore[arg-type]


def test_the_report_refuses_rows_of_another_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"item_id": "a", "score": 1.0, "meta": {"full_em": 1.0}}) + "\n")

    with pytest.raises(ValueError, match="no T1 components"):
        read_eval(path)
