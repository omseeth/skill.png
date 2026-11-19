"""TRAJECT-Bench adapter: the shipped artifacts, the row shape, items, and scoring.

No network and no weights. Two kinds of test live here. The ones on a synthetic
corpus patch `SPLIT_ROOT`, `TOOL_INDEX` and `_load`, so the adapter's own logic
is exercised without downloading 1800 rows. The ones on the shipped artifacts
read the committed manifest and index straight off disk, which is what checks
that what we ship is internally consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillpng.benchmarks.registry import build_benchmark
from skillpng.benchmarks.trajectbench import traject
from skillpng.benchmarks.trajectbench.traject import (
    METRICS,
    TrajectBench,
    TrajectOptions,
    _repo_id,
    build_target,
    canonical_calls,
    extract_calls,
    load_index,
    pool_digest,
    recoverable_flags,
    score_calls,
    serialize_calls,
)
from skillpng.benchmarks.trajectbench.traject_build import (
    CONFIGS,
    EXPECTED_REACHABLE,
    HF_REPO,
    SPLIT_ROOT,
    SPLITS,
    TOOL_INDEX,
    UNREACHABLE,
    index_digest,
    normalize_name,
)
from skillpng.config import BenchmarkCfg, load_config
from skillpng.results import read_rows

ROOT = Path(__file__).parent.parent
TRAJECT_CONFIG = ROOT / "configs" / "traject.yaml"
# The shipped prompt lives in the experiment config, not in the schema:
# reading it from there is also what checks the two stay wired together.
DEFAULT_SYSTEM_PROMPT = load_config(TRAJECT_CONFIG).benchmark.system_prompt
# Two synthetic domains, each with enough endpoints that a pool of gold plus
# N_DISTRACTORS can always be drawn.
DOMAINS = {"Alpha": "parallel_ecommerce_simple", "Beta": "parallel_gaming_simple"}
PARENTS = ("Acme", "Bravo", "Cosmo")
ENDPOINTS = ("search", "detail", "list", "count", "history")


def _tool(domain: str, parent: str, api: str) -> dict:
    """One synthetic index entry, shaped exactly like a shipped one."""
    name = f"{parent}: {api}"
    return {
        "key": normalize_name(name),
        "tool name": name,
        "parent tool name": parent,
        "parent tool description": f"{parent} is a {domain} provider.",
        "API name": api,
        "tool description": f"The {api} endpoint of {parent}.",
        "domain name": domain,
        "required": [{"name": "q", "type": "STRING", "description": "the search term"}],
        "optional": [{"name": "page", "type": "NUMBER", "description": "page number"}],
        "connected": [],
        "output_info": {"output_info_summary": f"what {api} returns"},
    }


def _index_document() -> dict:
    """A synthetic tool index with a correct self-digest."""
    tools = sorted(
        (
            _tool(domain, parent, f"{api}{n}")
            for domain in DOMAINS
            for parent in PARENTS
            for n, api in enumerate(ENDPOINTS)
        ),
        key=lambda tool: tool["key"],
    )
    return {"manifest_type": "tool_index", "digest": index_digest(tools), "tools": tools}


def _call(parent: str, api: str, query_value: str, page: str | None = "1") -> dict:
    """One gold call in the mirror's row shape, before canonicalization."""
    return {
        "tool name": f"{parent}: {api}",
        "tool description": "ignored by the adapter",
        "required parameters": [{"name": "q", "value": query_value}],
        "optional parameters": [] if page is None else [{"name": "page", "value": page}],
        "parent tool name": parent,
        "API name": api,
        "domain name": "Alpha",
        "executed_output": "dropped",
    }


def _corpus() -> tuple[dict, dict]:
    """Twelve synthetic rows over two domains, plus their manifest entries."""
    rows: dict[str, list[dict]] = {config: [] for config in DOMAINS.values()}
    entries: list[dict] = []

    for position, (domain, config) in enumerate(DOMAINS.items()):
        for offset in range(6):
            index = offset
            parents = PARENTS[: 2 + offset % 2]
            calls = [
                _call(parent, f"{ENDPOINTS[offset % len(ENDPOINTS)]}{offset % 5}", f"term{offset}")
                for parent in parents
            ]

            for call in calls:
                call["domain name"] = domain
            rows[config].append(
                {
                    "query": f"Please look up term{offset} for me on page 1.",
                    "tool_list": json.dumps(calls),
                    "task_name": f"task {offset % 3}",
                    "tool_count": len(calls),
                    "trajectory_type": "parallel",
                    "final_answer": "dropped",
                    "task_description": "dropped",
                }
            )
            entries.append(
                {
                    "id": f"{config}:{index}",
                    "domain": domain,
                    "group": f"{domain}::{'|'.join(parents)}",
                    "tool_count": len(calls),
                    "has_unseen_values": bool(position and offset > 3),
                }
            )

    return rows, {"train": entries[:4], "val": entries[4:6], "test": entries[6:]}


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Point the adapter at a synthetic index, manifest and mirror."""
    rows, splits = _corpus()
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(_index_document()))
    root = tmp_path / "split"

    for name in SPLITS:
        (root / name).mkdir(parents=True)
        (root / name / "items.json").write_text(json.dumps(splits[name]))
    (root / "split_manifest.json").write_text(
        json.dumps({"counts": {n: len(s) for n, s in splits.items()},
                    "pool_defaults": {"n_distractors": 4, "hard_negatives": 0}})
    )
    monkeypatch.setattr(traject, "TOOL_INDEX", index_path)
    monkeypatch.setattr(traject, "SPLIT_ROOT", root)
    monkeypatch.setattr(traject, "_load", lambda repo_id, config: rows.get(config, []))

    return {"rows": rows, "splits": splits, "root": root, "index": index_path}


def gold_of(item: object) -> list[dict]:
    """The item's gold calls, typed: Item.ground_truth is declared as object."""
    calls = getattr(item, "ground_truth", None)
    assert isinstance(calls, list)

    return calls


def _bench(options: dict | None = None) -> TrajectBench:
    """The adapter over the synthetic corpus, with a small pool by default."""
    merged = {"n_distractors": 4, **(options or {})}
    return TrajectBench(
        BenchmarkCfg(
            name="traject_bench", system_prompt=DEFAULT_SYSTEM_PROMPT, options=merged
        )
    )


# ---------------------------------------------------------------- options


def test_options_defaults_and_metric_names() -> None:
    parsed = TrajectOptions.parse({})
    assert parsed.metric == "full_em"
    assert parsed.n_distractors == 10
    assert parsed.hard_negatives == 0
    assert parsed.description_chars is None
    assert not parsed.render_output_info


@pytest.mark.parametrize("metric", METRICS)
def test_every_metric_has_a_reporting_name(metric: str, corpus: dict) -> None:
    assert _bench({"metric": metric}).metric_name


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown traject option"):
        TrajectOptions.parse({"metrics": "full_em"})


@pytest.mark.parametrize(
    "options",
    [
        {"metric": "bleu"},
        {"n_distractors": -1},
        {"hard_negatives": 11},
        {"description_chars": 0},
    ],
)
def test_out_of_range_options_are_rejected(options: dict) -> None:
    with pytest.raises(ValueError):
        TrajectOptions.parse(options)


def test_hard_negatives_may_not_exceed_the_pool() -> None:
    assert TrajectOptions.parse({"n_distractors": 4, "hard_negatives": 4})

    with pytest.raises(ValueError, match="hard_negatives"):
        TrajectOptions.parse({"n_distractors": 4, "hard_negatives": 5})


def test_repo_id_takes_the_hf_prefix_or_falls_back() -> None:
    assert _repo_id("hf:someone/mirror") == "someone/mirror"
    assert _repo_id("someone/mirror") == "someone/mirror"
    assert _repo_id(None) == HF_REPO
    assert _repo_id("  ") == HF_REPO


# ---------------------------------------------------------------- items


def test_splits_come_from_the_manifest(corpus: dict) -> None:
    bench = _bench()
    sizes = {name: len(bench.items(name)) for name in SPLITS}

    assert sizes == {name: len(split) for name, split in corpus["splits"].items()}


def test_unknown_split_is_an_error(corpus: dict) -> None:
    with pytest.raises(KeyError, match="unknown split"):
        _bench().items("dev")


def test_item_carries_the_manifest_fields_and_the_gold_calls(corpus: dict) -> None:
    item = _bench().items("train")[0]
    entry = corpus["splits"]["train"][0]

    assert item.id == entry["id"]
    assert item.group == entry["group"]
    assert item.meta["domain"] == entry["domain"]
    assert item.meta["tool_count"] == entry["tool_count"]
    assert item.meta["has_unseen_values"] == entry["has_unseen_values"]
    assert item.meta["question"].startswith("Please look up")
    assert len(gold_of(item)) == entry["tool_count"]


def test_dropped_fields_never_reach_the_item(corpus: dict) -> None:
    for item in _bench().items("test"):
        assert "dropped" not in json.dumps(gold_of(item))
        assert "dropped" not in item.target
        for call in gold_of(item):
            assert set(call) == {"tool name", "required parameters", "optional parameters"}


def test_group_is_always_set(corpus: dict) -> None:
    for name in SPLITS:
        assert all(item.group for item in _bench().items(name))


def test_limit_caps_each_split(corpus: dict) -> None:
    bench = TrajectBench(
        BenchmarkCfg(name="traject_bench", limit=2, options={"n_distractors": 4})
    )
    assert all(len(bench.items(name)) <= 2 for name in SPLITS)


def test_a_manifest_id_the_mirror_lacks_is_refused(corpus: dict, tmp_path: Path) -> None:
    entries = json.loads((corpus["root"] / "test" / "items.json").read_text())
    entries.append({**entries[0], "id": "parallel_ecommerce_simple:999"})
    (corpus["root"] / "test" / "items.json").write_text(json.dumps(entries))

    with pytest.raises(ValueError, match="not in"):
        _bench().items("test")


def test_a_tool_count_that_moved_is_refused(corpus: dict) -> None:
    entries = json.loads((corpus["root"] / "train" / "items.json").read_text())
    entries[0]["tool_count"] += 1
    (corpus["root"] / "train" / "items.json").write_text(json.dumps(entries))

    with pytest.raises(ValueError, match="manifest says"):
        _bench().items("train")


def test_a_pool_digest_that_moved_is_refused(corpus: dict) -> None:
    entries = json.loads((corpus["root"] / "train" / "items.json").read_text())

    for entry in entries:
        entry["pool_digest"] = "0" * 16
    (corpus["root"] / "train" / "items.json").write_text(json.dumps(entries))

    with pytest.raises(ValueError, match="pool digest"):
        _bench().items("train")


def test_a_different_pool_size_skips_the_digest_check(corpus: dict) -> None:
    entries = json.loads((corpus["root"] / "train" / "items.json").read_text())

    for entry in entries:
        entry["pool_digest"] = "0" * 16
    (corpus["root"] / "train" / "items.json").write_text(json.dumps(entries))
    # The manifest's pool_defaults say n_distractors=4; asking for 5 is a
    # deliberately different pool, so the shipped digests no longer apply.
    assert _bench({"n_distractors": 5}).items("train")


def test_recoverable_counts_that_moved_are_refused(corpus: dict) -> None:
    entries = json.loads((corpus["root"] / "train" / "items.json").read_text())

    for entry in entries:
        entry.update(
            {"n_required": 99, "n_required_recoverable": 0, "n_optional": 0,
             "n_optional_recoverable": 0}
        )
    (corpus["root"] / "train" / "items.json").write_text(json.dumps(entries))

    with pytest.raises(ValueError, match="parameter counts"):
        _bench().items("train")


def test_a_gold_tool_missing_from_the_index_is_refused(corpus: dict) -> None:
    rows = corpus["rows"]["parallel_ecommerce_simple"]
    calls = json.loads(rows[0]["tool_list"])
    calls[0]["tool name"] = "Nowhere: nothing"
    rows[0]["tool_list"] = json.dumps(calls)

    with pytest.raises(ValueError, match="tool index does not carry"):
        _bench().items("train")


def test_registry_builds_the_adapter(corpus: dict) -> None:
    bench = build_benchmark(BenchmarkCfg(name="traject_bench", options={"n_distractors": 4}))

    assert isinstance(bench, TrajectBench)


def test_the_adapter_reports_its_pool_size(corpus: dict) -> None:
    """The count evaluate.py records on every result row.

    Args:
        corpus: Synthetic corpus patched in for the adapter.
    """
    assert _bench({"n_distractors": 5}).n_distractors == 5


# ---------------------------------------------------------------- target


def test_target_is_the_shape_the_scorer_reads_back(corpus: dict) -> None:
    item = _bench().items("train")[0]

    assert item.target.startswith("<answer>[")
    assert item.target.endswith("]</answer>")
    assert extract_calls(item.target) == gold_of(item)


def test_target_is_sorted_and_deterministic() -> None:
    calls = canonical_calls(
        [
            {"tool name": "B: x", "required parameters": [{"name": "z", "value": "1"},
                                                          {"name": "a", "value": "2"}]},
            {"tool name": "A: y", "optional parameters": [{"name": "p", "value": "3"}]},
        ]
    )
    assert [call["tool name"] for call in calls] == ["A: y", "B: x"]
    assert [p["name"] for p in calls[1]["required parameters"]] == ["a", "z"]
    assert serialize_calls(calls) == serialize_calls(canonical_calls(list(reversed(calls))))


def test_target_round_trips_through_the_extractor() -> None:
    calls = canonical_calls([{"tool name": "A: y", "required parameters": []}])

    assert extract_calls(build_target(calls)) == calls


# ---------------------------------------------------------------- scoring


def _gold() -> list[dict]:
    """Two gold calls used by the scoring tests."""
    return canonical_calls(
        [
            {
                "tool name": "Acme: search0",
                "required parameters": [{"name": "q", "value": "term0"}],
                "optional parameters": [{"name": "page", "value": "1"}],
            },
            {
                "tool name": "Bravo: detail1",
                "required parameters": [{"name": "q", "value": "term1"}],
                "optional parameters": [],
            },
        ]
    )


def test_a_perfect_prediction_scores_one() -> None:
    gold = _gold()
    scores = score_calls(gold, gold)

    assert scores["parse_success"] == scores["tool_em"] == scores["full_em"] == 1.0
    assert scores["required_em"] == scores["optional_em"] == scores["usage_em"] == 1.0


def test_a_failed_parse_scores_zero_everywhere() -> None:
    scores = score_calls(None, _gold())

    assert scores["parse_success"] == 0.0
    assert all(scores[key] == 0.0 for key in ("tool_em", "tool_inclusion", "full_em"))
    assert scores["n_gold"] == 2


def test_a_missing_tool_costs_the_set_match_but_not_the_arguments() -> None:
    gold = _gold()
    scores = score_calls(gold[:1], gold)

    assert scores["tool_em"] == 0.0
    assert scores["tool_inclusion"] == 0.5
    # The one tool it did select was called correctly.
    assert scores["required_em"] == 1.0
    assert scores["full_em"] == 0.0


def test_a_wrong_required_value_costs_the_headline() -> None:
    gold = _gold()
    wrong = json.loads(json.dumps(gold))
    wrong[0]["required parameters"][0]["value"] = "somethingelse"
    scores = score_calls(wrong, gold)

    assert scores["tool_em"] == 1.0
    assert scores["required_em"] == 0.5
    assert scores["full_em"] == 0.0


def test_a_wrong_optional_value_is_reported_but_not_folded_in() -> None:
    gold = _gold()
    wrong = json.loads(json.dumps(gold))
    wrong[0]["optional parameters"][0]["value"] = "9"
    scores = score_calls(wrong, gold)

    assert scores["required_em"] == 1.0
    assert scores["full_em"] == 1.0
    assert scores["optional_em"] == 0.5
    assert scores["usage_em"] == 0.5


def test_values_are_compared_after_the_reference_normalization() -> None:
    gold = _gold()
    same = json.loads(json.dumps(gold))
    same[0]["optional parameters"][0]["value"] = 1
    same[1]["required parameters"][0]["value"] = " term1 "

    assert score_calls(same, gold)["full_em"] == 1.0


def test_an_empty_valued_parameter_is_dropped_before_comparison() -> None:
    gold = _gold()
    padded = json.loads(json.dumps(gold))
    padded[1]["optional parameters"] = [{"name": "unused", "value": ""}]

    assert score_calls(padded, gold)["full_em"] == 1.0


def test_score_selects_the_configured_metric(corpus: dict) -> None:
    bench = _bench({"metric": "tool_inclusion"})
    item = bench.items("train")[0]
    half = build_target(gold_of(item)[:1])

    assert bench.score(item.target, item) == 1.0
    assert bench.score(half, item) == pytest.approx(1 / len(gold_of(item)))


def test_breakdown_carries_the_stratification_keys(corpus: dict) -> None:
    bench = _bench()
    item = bench.items("test")[0]
    scores = bench.breakdown(item.target, item)

    assert scores["domain"] == item.meta["domain"]
    assert scores["tool_count"] == item.meta["tool_count"]
    assert scores["has_unseen_values"] == item.meta["has_unseen_values"]
    assert scores["pool_size"] == len(item.meta["pool"])


def test_recoverable_flags_find_only_what_the_query_contains() -> None:
    calls = canonical_calls(
        [
            {
                "tool name": "Acme: search0",
                "required parameters": [{"name": "q", "value": "Red Chair"}],
                "optional parameters": [{"name": "sort", "value": "HELPFUL"}],
            }
        ]
    )
    flags = recoverable_flags("Find me a red chair, please.", calls)

    assert flags["Acme: search0"] == ["q"]


# ---------------------------------------------------------------- shipped artifacts


def test_shipped_index_matches_its_own_digest() -> None:
    document = load_index(TOOL_INDEX)

    assert document["n_tools"] == len(document["tools"])
    assert document["digest"] == index_digest(document["tools"])


def test_shipped_index_carries_no_defaults_and_no_code() -> None:
    document = load_index(TOOL_INDEX)

    for tool in document["tools"]:
        assert "code" not in tool
        for parameter in tool["required"] + tool["optional"]:
            assert set(parameter) == {"name", "type", "description"}


def test_shipped_index_keys_are_the_provider_and_the_endpoint() -> None:
    for tool in load_index(TOOL_INDEX)["tools"]:
        assert tool["key"] == normalize_name(f"{tool['parent tool name']}: {tool['API name']}")


def test_shipped_manifest_and_items_agree() -> None:
    manifest = json.loads((SPLIT_ROOT / "split_manifest.json").read_text())
    splits = {name: json.loads((SPLIT_ROOT / name / "items.json").read_text()) for name in SPLITS}

    assert {name: len(split) for name, split in splits.items()} == manifest["counts"]
    assert sum(manifest["counts"].values()) == EXPECTED_REACHABLE
    assert manifest["source_rows"] == 1800
    assert EXPECTED_REACHABLE + manifest["n_dropped_unreachable"] == manifest["source_rows"]
    assert set(manifest["source_configs"]) == set(CONFIGS)


def test_shipped_items_are_all_reachable() -> None:
    for name in SPLITS:
        for entry in json.loads((SPLIT_ROOT / name / "items.json").read_text()):
            assert entry["n_required"] == entry["n_required_recoverable"], entry["id"]


def test_dropped_rows_are_listed_and_disjoint_from_the_splits() -> None:
    dropped = json.loads(UNREACHABLE.read_text())
    shipped = {
        entry["id"]
        for name in SPLITS
        for entry in json.loads((SPLIT_ROOT / name / "items.json").read_text())
    }

    assert len(dropped) == 1800 - EXPECTED_REACHABLE
    assert not {row["id"] for row in dropped} & shipped
    assert all(row["unreachable_required"] for row in dropped)


def test_shipped_items_carry_every_declared_field() -> None:
    manifest = json.loads((SPLIT_ROOT / "split_manifest.json").read_text())

    for name in SPLITS:
        for entry in json.loads((SPLIT_ROOT / name / "items.json").read_text()):
            assert set(entry) == set(manifest["item_fields"])


def test_shipped_ids_are_unique_and_well_formed() -> None:
    seen: set[str] = set()

    for name in SPLITS:
        for entry in json.loads((SPLIT_ROOT / name / "items.json").read_text()):
            config, _, index = entry["id"].rpartition(":")
            assert config in CONFIGS
            assert index.isdigit()
            assert entry["id"] not in seen
            seen.add(entry["id"])


def test_no_shipped_group_straddles_a_split() -> None:
    groups: dict[str, str] = {}

    for name in SPLITS:
        for entry in json.loads((SPLIT_ROOT / name / "items.json").read_text()):
            assert groups.setdefault(entry["group"], name) == name


def test_every_shipped_gold_tool_joins_the_index() -> None:
    keys = set(load_index(TOOL_INDEX)["by_key"])

    for name in SPLITS:
        for entry in json.loads((SPLIT_ROOT / name / "items.json").read_text()):
            domain = entry["domain"]
            assert domain in load_index(TOOL_INDEX)["by_domain"]
            assert entry["group"].split("::")[0] == domain
    assert keys


def test_every_domain_has_room_for_the_default_pool() -> None:
    document = load_index(TOOL_INDEX)
    domains = {
        entry["domain"]
        for name in SPLITS
        for entry in json.loads((SPLIT_ROOT / name / "items.json").read_text())
    }

    for domain in domains:
        # Gold tools come out of the same domain, so the worst case is the
        # largest gold set the corpus carries.
        assert len(document["by_domain"][domain]) >= 11 + traject.N_DISTRACTORS


def test_shipped_config_sets_budgets_the_targets_need() -> None:
    cfg = load_config(TRAJECT_CONFIG)

    assert cfg.benchmark.name == "traject_bench"
    assert "<answer>" in cfg.benchmark.system_prompt
    # Measured maximum target is 872 tokens; the 512 default truncates 9% of
    # the corpus silently.
    assert (cfg.model.max_target_tokens or 0) >= 1024
    assert (cfg.model.max_sequence_tokens or 0) >= 12288
    assert TrajectOptions.parse(cfg.benchmark.options).metric in METRICS


def test_recorded_rows_carry_their_pool_size() -> None:
    """analyze_runs splits TRAJECT-Bench runs by this field."""
    rows = [
        row
        for path in sorted((ROOT / "runs").glob("*/*/results.jsonl"))
        for row in read_rows(path)
        if row["benchmark"] == "traject_bench"
    ]

    assert rows, "no recorded rows to check"
    assert all(isinstance(row.get("n_distractors"), int) for row in rows)


def test_pool_digest_depends_on_order() -> None:
    assert pool_digest(["a", "b"]) != pool_digest(["b", "a"])


# ---------------------------------------------------------------- offline report


def _eval_rows(path: Path, scores: list[float], parse: list[float] | None = None) -> Path:
    """Write a per-item JSONL in the shape evaluate.py produces."""
    parse = [1.0] * len(scores) if parse is None else parse
    lines = [
        json.dumps(
            {
                "item_id": f"item:{index}",
                "question": "q",
                "target": "t",
                "output": "o",
                "score": score,
                "meta": {
                    "parse_success": ok,
                    "full_em": score,
                    "tool_em": score,
                    "domain": "Alpha" if index % 2 else "Beta",
                    "tool_count": 2 + index % 3,
                    "has_unseen_values": index % 5 == 0,
                },
            }
        )
        for index, (score, ok) in enumerate(zip(scores, parse, strict=True))
    ]
    path.write_text("\n".join(lines) + "\n")

    return path


def test_report_summarizes_components_and_strata(tmp_path: Path) -> None:
    from skillpng.benchmarks.trajectbench.traject_report import build_report, render

    base = _eval_rows(tmp_path / "none.jsonl", [1.0, 0.0, 1.0, 0.0])
    report = build_report(base)
    summary = report["conditions"][str(base)]

    assert summary["n"] == 4
    assert summary["overall"]["full_em"] == 0.5
    assert set(summary["by_domain"]) == {"Alpha", "Beta"}
    assert sum(s["n"] for s in summary["by_tool_count"].values()) == 4
    assert "parse_success" in render(report)


def test_report_conditions_the_headline_on_parsing(tmp_path: Path) -> None:
    from skillpng.benchmarks.trajectbench.traject_report import build_report

    base = _eval_rows(tmp_path / "none.jsonl", [1.0, 0.0, 0.0, 0.0], parse=[1.0, 1.0, 0.0, 0.0])
    summary = build_report(base)["conditions"][str(base)]

    assert summary["overall"]["parse_success"] == 0.5
    assert summary["overall"]["full_em"] == 0.25
    # Half the rows never produced JSON; over the ones that did, it is 1 of 2.
    assert summary["parsed_only"] == 0.5
    assert summary["n_parsed"] == 2


def test_report_pairs_on_item_id(tmp_path: Path) -> None:
    from skillpng.benchmarks.trajectbench.traject_report import build_report

    base = _eval_rows(tmp_path / "none.jsonl", [0.0, 0.0, 1.0, 1.0])
    treat = _eval_rows(tmp_path / "opt.jsonl", [1.0, 0.0, 1.0, 1.0])
    delta = build_report(base, treat, draws=200)["paired"]

    assert delta["n_paired"] == 4
    assert delta["delta"] == 0.25
    assert delta["wins"] == 1
    assert delta["losses"] == 0
    assert delta["ci_low"] <= delta["delta"] <= delta["ci_high"]


def test_report_refuses_rows_without_meta(tmp_path: Path) -> None:
    from skillpng.benchmarks.trajectbench.traject_report import read_eval

    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"item_id": "a", "target": "t", "output": "o", "score": 1.0}) + "\n")

    with pytest.raises(ValueError, match="carries no meta"):
        read_eval(path)


def test_report_refuses_two_evals_of_different_splits(tmp_path: Path) -> None:
    from skillpng.benchmarks.trajectbench.traject_report import paired_delta, read_eval

    base = read_eval(_eval_rows(tmp_path / "a.jsonl", [1.0]))
    other = {"somewhere:else": {"full_em": 1.0, "parse_success": 1.0}}

    with pytest.raises(ValueError, match="share no item ids"):
        paired_delta(base, other)
