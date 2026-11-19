"""The parts of TRAJECT-Bench that must not move: turns, leak control, determinism.

The prompt protocol is what makes two runs comparable, so it is pinned here
rather than left to the adapter's implementation. The parity tests compile the
reference metrics straight out of the vendored clone and are skipped when the
clone is absent.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

from skillpng.benchmarks.trajectbench import traject
from skillpng.benchmarks.trajectbench.traject import (
    canonical_calls,
    extract_calls,
    normalize_value,
    render_tool,
    score_calls,
    serialize_calls,
)
from skillpng.benchmarks.trajectbench.traject_build import TOOL_INDEX, normalize_name
from test_traject import _bench, corpus, gold_of  # noqa: F401

REFERENCE_CODE = Path(__file__).parent.parent / "baselines" / "traject_bench" / "code"
needs_clone = pytest.mark.skipif(
    not REFERENCE_CODE.is_dir(), reason="reference clone not present"
)
# Fields the reference schemas carry that a prompt must never show: both name
# gold argument values outright.
FORBIDDEN = ("default", "querystring", "connected tools", "connect params")


def _user_text(messages: list[dict]) -> str:
    """Concatenate the text blocks of the last user turn."""
    content = messages[-1]["content"]
    return "".join(block["text"] for block in content if block["type"] == "text")


# ---------------------------------------------------------------- turn structure


def test_turns_are_system_then_one_user(corpus: dict) -> None:  # noqa: F811
    bench = _bench()
    item = bench.items("train")[0]
    messages = bench.build_messages(item, with_image=False)

    assert [message["role"] for message in messages] == ["system", "user"]


def test_no_system_prompt_means_no_system_turn(corpus: dict) -> None:  # noqa: F811
    from skillpng.config import BenchmarkCfg

    bench = traject.TrajectBench(
        BenchmarkCfg(name="traject_bench", system_prompt="", options={"n_distractors": 4})
    )
    messages = bench.build_messages(bench.items("train")[0], with_image=False)

    assert [message["role"] for message in messages] == ["user"]


def test_user_turn_is_tools_then_query(corpus: dict) -> None:  # noqa: F811
    bench = _bench()
    item = bench.items("train")[0]
    text = _user_text(bench.build_messages(item, with_image=False))

    assert text.startswith("## Tools\n")
    assert text.index("## Tools") < text.index("## Query")
    assert text.endswith(item.meta["question"])


def test_the_image_rides_in_the_user_turn_at_the_requested_site(corpus: dict) -> None:  # noqa: F811
    bench = _bench()
    item = bench.items("train")[0]

    for site, position in (("pre_query", 0), ("post_query", -1)):
        messages = bench.build_messages(item, with_image=True, site=site)
        content = messages[-1]["content"]

        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], str)
        assert content[position]["type"] == "image"
        assert sum(block["type"] == "image" for block in content) == 1


def test_the_text_is_byte_identical_with_and_without_the_image(corpus: dict) -> None:  # noqa: F811
    bench = _bench()

    for item in bench.items("test"):
        plain = _user_text(bench.build_messages(item, with_image=False))
        pre = _user_text(bench.build_messages(item, with_image=True, site="pre_query"))
        post = _user_text(bench.build_messages(item, with_image=True, site="post_query"))

        assert plain == pre == post


# ---------------------------------------------------------------- leak control


def test_rendering_cannot_tell_gold_from_distractor(corpus: dict) -> None:  # noqa: F811
    bench = _bench()
    rendered: dict[str, set[str]] = {}

    for name in ("train", "val", "test"):
        for item in bench.items(name):
            gold = {normalize_name(call["tool name"]) for call in gold_of(item)}

            for key in item.meta["pool"]:
                block = render_tool(bench.index["by_key"][key])
                rendered.setdefault(key, set()).add(block)
                # The same endpoint renders the same way whether or not this
                # particular item happens to need it.
                assert (key in gold) or block == render_tool(bench.index["by_key"][key])

    assert all(len(blocks) == 1 for blocks in rendered.values())


def test_no_prompt_shows_a_default_or_a_querystring(corpus: dict) -> None:  # noqa: F811
    bench = _bench()

    for item in bench.items("test"):
        text = _user_text(bench.build_messages(item, with_image=False)).casefold()

        for forbidden in FORBIDDEN:
            assert forbidden not in text


def test_the_shipped_index_cannot_supply_a_default(corpus: dict) -> None:  # noqa: F811
    document = traject.load_index(TOOL_INDEX)
    blob = json.dumps(document["tools"][:200])

    assert '"default"' not in blob
    assert '"code"' not in blob
    assert '"connect params"' not in blob


def test_description_budget_applies_to_every_tool_alike(corpus: dict) -> None:  # noqa: F811
    bench = _bench({"description_chars": 20})
    item = bench.items("train")[0]
    text = _user_text(bench.build_messages(item, with_image=False))
    lengths = [len(line) for line in text.splitlines() if line.startswith("Description: ")]

    assert lengths and max(lengths) <= len("Description: ") + 23


def test_output_info_is_off_unless_asked_for(corpus: dict) -> None:  # noqa: F811
    item = _bench().items("train")[0]

    assert "Output: " not in _user_text(_bench().build_messages(item, with_image=False))
    with_info = _bench({"render_output_info": True})
    assert "Output: " in _user_text(with_info.build_messages(with_info.items("train")[0], False))


# ---------------------------------------------------------------- the pool


def test_the_pool_holds_every_gold_tool_and_the_distractors(corpus: dict) -> None:  # noqa: F811
    bench = _bench()

    for item in bench.items("test"):
        gold = {normalize_name(call["tool name"]) for call in gold_of(item)}
        pool = item.meta["pool"]

        assert gold <= set(pool)
        assert len(pool) == len(gold) + 4
        assert len(set(pool)) == len(pool)


def test_the_pool_stays_inside_the_domain(corpus: dict) -> None:  # noqa: F811
    bench = _bench()

    for item in bench.items("test"):
        for key in item.meta["pool"]:
            assert bench.index["by_key"][key]["domain name"] == item.meta["domain"]


def test_gold_position_is_not_a_giveaway(corpus: dict) -> None:  # noqa: F811
    bench = _bench()
    firsts = set()

    for item in bench.items("test"):
        gold = {normalize_name(call["tool name"]) for call in gold_of(item)}
        firsts.add(item.meta["pool"][0] in gold)

    # Ordering is fixed but uncorrelated with gold membership, so gold does
    # not always land first and does not always land last.
    assert firsts == {True, False}


def test_pools_are_identical_across_two_constructions(corpus: dict) -> None:  # noqa: F811
    one, two = _bench(), _bench()

    for name in ("train", "val", "test"):
        left = {item.id: item.meta["pool"] for item in one.items(name)}
        right = {item.id: item.meta["pool"] for item in two.items(name)}

        assert left == right


def test_prompts_and_targets_are_identical_across_two_constructions(corpus: dict) -> None:  # noqa: F811
    one, two = _bench(), _bench()

    for name in ("train", "val", "test"):
        for left, right in zip(one.items(name), two.items(name), strict=True):
            assert left.target == right.target
            assert _user_text(one.build_messages(left, with_image=False)) == _user_text(
                two.build_messages(right, with_image=False)
            )


def test_hard_negatives_prefer_the_gold_providers(corpus: dict) -> None:  # noqa: F811
    plain = {item.id: item.meta["pool"] for item in _bench().items("test")}
    hard_bench = _bench({"hard_negatives": 2})
    moved = 0

    for item in hard_bench.items("test"):
        gold = {normalize_name(call["tool name"]) for call in gold_of(item)}
        parents = {
            normalize_name(hard_bench.index["by_key"][key]["parent tool name"]) for key in gold
        }
        siblings = [
            key
            for key in item.meta["pool"]
            if key not in gold
            and normalize_name(hard_bench.index["by_key"][key]["parent tool name"]) in parents
        ]
        moved += item.meta["pool"] != plain[item.id]

        assert len(siblings) >= 2

    assert moved


# ---------------------------------------------------------------- extraction


@pytest.mark.parametrize(
    "wrapper",
    [
        "<answer>{payload}</answer>",
        "let me think...\n<answer>{payload}</answer>",
        "<answer>[]</answer> wait, no.\n<answer>{payload}</answer>",
        "```json\n{payload}\n```",
        "```\n{payload}\n```",
        "Here is the answer: {payload}",
    ],
)
def test_the_answer_is_found_wherever_a_model_puts_it(wrapper: str) -> None:
    calls = canonical_calls(
        [{"tool name": "A: b", "required parameters": [{"name": "q", "value": "x"}]}]
    )
    assert extract_calls(wrapper.format(payload=serialize_calls(calls))) == calls


def test_the_last_tagged_block_wins() -> None:
    first = canonical_calls([{"tool name": "A: b"}])
    last = canonical_calls([{"tool name": "C: d"}])
    output = f"<answer>{serialize_calls(first)}</answer>\n<answer>{serialize_calls(last)}</answer>"

    assert extract_calls(output) == last


@pytest.mark.parametrize("end", ["\n</think>\n", "\n"])
def test_a_tag_named_while_reasoning_does_not_swallow_the_answer(end: str) -> None:
    """Nemotron names the tag in its trace before answering. A block that ran
    from that mention to the real closing tag parsed as nothing."""
    calls = canonical_calls([{"tool name": "C: d"}])
    output = f"Wrap it in <answer> tags.{end}<answer>{serialize_calls(calls)}</answer>"

    assert extract_calls(output) == calls
    assert extract_calls(f"{output}\nThat was the <answer> block.") == calls


def test_a_draft_inside_the_reasoning_trace_is_never_scored() -> None:
    draft = canonical_calls([{"tool name": "A: b"}])
    output = f"<answer>{serialize_calls(draft)}</answer>\n</think>\nI cannot decide."

    assert extract_calls(output) is None


def test_the_benchmarks_own_answer_shape_is_accepted() -> None:
    calls = canonical_calls([{"tool name": "A: b", "required parameters": []}])
    output = json.dumps({"tool list": [{"tool name": "A: b", "required_parameters": []}]})

    assert extract_calls(f"<answer>{output}</answer>") == calls


@pytest.mark.parametrize(
    "output", ["", "I cannot help with that.", "<answer></answer>", "<answer>[]</answer>",
               "<answer>{broken", "<answer>[{\"no name\": 1}]</answer>"],
)
def test_an_unusable_output_is_a_parse_failure(output: str) -> None:
    assert extract_calls(output) is None
    assert score_calls(extract_calls(output), [{"tool name": "A: b"}])["parse_success"] == 0.0


def test_underscored_parameter_keys_are_accepted() -> None:
    payload = '[{"tool name": "A: b", "required_parameters": [{"name": "q", "value": "x"}]}]'
    output = f"<answer>{payload}</answer>"
    calls = extract_calls(output)

    assert calls is not None
    assert calls[0]["required parameters"] == [{"name": "q", "value": "x"}]


# ---------------------------------------------------------------- reference parity


def _reference() -> dict[str, Any]:
    """Compile the reference metrics out of the clone, without importing it.

    utils/metrics.py imports the clone's model providers at module scope, which
    reaches for API keys; the functions under test do not need any of that.
    """
    source = (REFERENCE_CODE / "utils" / "metrics.py").read_text()
    tree = ast.parse(source)
    wanted = {
        "normalize_numeric", "normalize_whitespace", "normalize_location", "normalize_date",
        "normalize_boolean", "normalize_value", "is_empty", "normalize_param_list",
        "compare_tool_parameters", "exact_match_tools", "inclusion_tools", "tool_traj_usage",
    }
    body: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace: dict[str, Any] = {"ast": ast, "re": re, "datetime": __import__("datetime").datetime}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<vendored>", "exec"), namespace)

    return namespace


@needs_clone
@pytest.mark.parametrize(
    "value",
    ["1", "1.0", " 1 ", "true", "TRUE", "false", "HELPFUL", "Rome, Italy", "Paris, FR",
     "2023-1-5", "2023-01-05", "BTC/USD", "", "1day", "-119.5936", "en-US"],
)
def test_value_normalization_matches_the_reference(value: str) -> None:
    assert normalize_value(value) == _reference()["normalize_value"](value)


@needs_clone
def test_tool_set_and_inclusion_match_the_reference(corpus: dict) -> None:  # noqa: F811
    reference = _reference()
    bench = _bench()

    for item in bench.items("test"):
        gold = gold_of(item)

        for prediction in (gold, gold[:1], gold[1:], []):
            ours = score_calls(canonical_calls(prediction) or None, gold)
            assert ours["tool_em"] == float(reference["exact_match_tools"](gold, prediction))
            assert ours["tool_inclusion"] == pytest.approx(
                reference["inclusion_tools"](gold, prediction)
            )


@needs_clone
def test_usage_matches_the_reference(corpus: dict) -> None:  # noqa: F811
    reference = _reference()
    bench = _bench()

    for item in bench.items("test"):
        gold = gold_of(item)
        wrong = json.loads(json.dumps(gold))
        wrong[0]["required parameters"] = [{"name": "q", "value": "definitely not it"}]

        for prediction in (gold, wrong):
            usage = reference["tool_traj_usage"](gold, prediction)
            expected = sum(usage) / len(usage) if usage else 0.0

            assert score_calls(prediction, gold)["usage_em"] == pytest.approx(expected)


@needs_clone
def test_the_reference_prompt_builder_still_ships_defaults() -> None:
    """Our protocol strips parameter defaults; the reference one does not.

    The absolute numbers are therefore not comparable with the paper's, and
    this test is what notices if the reference ever changes its mind.
    """
    source = (REFERENCE_CODE / "evaluation" / "tool_evaluation_model.py").read_text()

    assert 'essential_fields = ["parent tool name", "required_parameters"' in source
