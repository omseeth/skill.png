"""T1 against upstream: the vendored code, the metrics, and the whole pipeline.

Three kinds of test, none in the default run's way:

- needs_clone (skipped without baselines/t1/code): the vendored files equal the
  clone's after exactly the documented modifications, the exec namespace is
  process_model_output.py's, the per-turn functions agree with upstream's row
  functions, and our aggregation agrees with upstream's steps 2-3 run on our
  per-turn records.
- integration (real data, the clone, a few minutes each): upstream's three
  evaluation steps end to end on one whole test file per domain, the oracle
  and do-nothing predictors over the shipped test split, and gold replay under
  upstream's own dependency pins.

Upstream's scripts always run with sys.executable, never python3 from PATH.
Step 2 imports sacrebleu at module level; it gets a stub, because this project
does not compute BLEU, and the BLEU columns are left out of every comparison.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from skillpng.benchmarks.t1 import t1_build
from skillpng.benchmarks.t1.t1 import T1, extract_code, gold_calls, turn_metrics
from skillpng.benchmarks.t1.t1_build import (
    DOMAINS,
    HF_REPO,
    SPLITS,
    SYSTEM_PROMPT,
    UPSTREAM_ROOT,
    dialogue_file,
    read_dialogues,
    snapshot,
    user_turns,
)
from skillpng.benchmarks.t1.t1_exec import (
    GLOBALS,
    ONTOLOGY,
    Executor,
    Tools,
    replay_file,
    replay_plans,
)
from skillpng.benchmarks.t1.t1_report import FIELDS, upstream_metrics
from skillpng.config import BenchmarkCfg

ROOT = Path(__file__).parent.parent
CLONE = ROOT / "baselines" / "t1" / "code"
needs_clone = pytest.mark.skipif(not CLONE.is_dir(), reason="upstream T1 clone not present")
VENDORED = "skillpng.benchmarks.t1.upstream"
# Empty __init__.py files added where upstream has namespace packages.
ADDED = {
    "tools/__init__.py",
    "tools/schema/__init__.py",
    "tools/utils/__init__.py",
    "data/__init__.py",
    "data/ontology/__init__.py",
}
DELETED_LINES = {"from google import genai", "from transformers import AutoTokenizer"}
# Apache-2.0 section 4(b): every vendored file we changed says so at its top.
CHANGE_NOTICE = (
    "# Modified by the Skill.png authors from T1 at commit 22a16b7 (Apache-2.0);\n"
    "# the changes are listed in skillpng/benchmarks/t1/upstream/VENDORED.md.\n\n"
)
STUB_SACREBLEU = (
    "class _Score:\n    score = 0.0\n\n\n"
    "def corpus_bleu(hypotheses, references):\n    return _Score()\n"
)


def _dict(value: object) -> dict[str, object]:
    """A record value, narrowed to the mapping it is."""
    assert isinstance(value, dict)
    return value


def _upstream_env(tmp_path: Path, root: Path | None = None) -> dict[str, str]:
    """Environment for upstream's scripts: its package, the stub, the ontology."""
    stub = tmp_path / "stub"
    stub.mkdir(exist_ok=True)
    (stub / "sacrebleu.py").write_text(STUB_SACREBLEU)
    env = {**os.environ, "PYTHONPATH": f"{CLONE / 'src'}{os.pathsep}{stub}"}

    if root is not None:
        env.update({name: str(root / relative) for name, relative in ONTOLOGY.items()})

    return env


def _compile_functions(
    path: Path, names: set[str], namespace: dict[str, object]
) -> dict[str, object]:
    """Compile named top-level functions out of an upstream file, without importing it."""
    tree = ast.parse(path.read_text())
    body: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in body if isinstance(node, ast.FunctionDef)} == names
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)

    return namespace


# ------------------------------------------------------------------- vendoring


def _expected(path: Path, relative: str) -> str:
    """What the vendored copy of one upstream file must read."""
    original = path.read_text()
    text = re.sub(
        r"^(from|import) t1\.", r"\1 skillpng.benchmarks.t1.upstream.", original, flags=re.M
    )

    if relative == "planner/planner_code.py":
        text = "".join(
            line
            for line in text.splitlines(keepends=True)
            if line.rstrip("\n") not in DELETED_LINES
        )

    return text if text == original else CHANGE_NOTICE + text


@needs_clone
def test_vendored_files_are_upstreams_after_the_documented_modifications() -> None:
    vendored = {
        path.relative_to(UPSTREAM_ROOT).as_posix()
        for path in UPSTREAM_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    source = CLONE / "src" / "t1"
    copied = vendored - ADDED - {"LICENSE", "VENDORED.md"}

    for relative in ADDED:
        assert (UPSTREAM_ROOT / relative).read_text() == ""
        assert not (source / relative).exists()

    for relative in sorted(copied):
        assert (UPSTREAM_ROOT / relative).read_text() == _expected(source / relative, relative), (
            relative
        )

    assert (UPSTREAM_ROOT / "LICENSE").read_bytes() == (CLONE / "LICENSE").read_bytes()
    expected = {
        path.relative_to(source).as_posix()
        for folder in ("tools", "tools/schema", "tools/utils", "data/ontology", "planner")
        for path in (source / folder).glob("*.py")
    } - {"tools/test_tools.py"}
    expected |= {"__init__.py", "evaluation/__init__.py", "evaluation/eval_metrics.py"}

    assert copied == expected


@needs_clone
def test_the_rewrite_touches_no_class_body() -> None:
    for path in UPSTREAM_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert "skillpng" not in (ast.get_source_segment(path.read_text(), node) or "")


@needs_clone
def test_the_exec_namespace_is_process_model_outputs() -> None:
    tree = ast.parse((CLONE / "evaluation" / "process_model_output.py").read_text())
    names: dict[str, tuple[str, str | None]] = {}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name] = (alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module).replace("t1.", ".", 1) if node.module else ""
            module = module if module.startswith(".") else str(node.module)

            for alias in node.names:
                names[alias.asname or alias.name] = (module, alias.name)

    assert names == {name: value for name, value in GLOBALS.items()}


# ---------------------------------------------------------- per-turn functions


def _row_functions() -> dict[str, object]:
    """Upstream's row functions from evaluation/, bound to the vendored eval_metrics."""
    from skillpng.benchmarks.t1.upstream.evaluation import eval_metrics

    namespace: dict[str, object] = {"pd": pd, "re": re, "Counter": Counter, **vars(eval_metrics)}
    _compile_functions(
        CLONE / "evaluation" / "generate_evaluation_metrics.py",
        {
            "extract_actual_tool_calls",
            "extract_generated_tool_calls",
            "tool_call_evaluation_metrics",
            "find_non_executable_code",
            "cache_summary_exact_match",
        },
        namespace,
    )
    _compile_functions(
        CLONE / "evaluation" / "process_model_output.py",
        {"extract_code_from_generated_plan"},
        namespace,
    )

    return namespace


def _pair(value: object) -> tuple[object, object]:
    """An upstream row function's two results."""
    assert isinstance(value, tuple) and len(value) == 2
    return value[0], value[1]


def _call(namespace: dict[str, object], name: str, *args: object) -> object:
    """Call one of the compiled upstream functions."""
    function = namespace[name]
    assert callable(function)

    return function(*args)


HOTELS = (
    'hotels = search_hotels(city="Austin", checkin_date=["May 1, 2025"], '
    'checkout_date=["May 3, 2025"], stars=[4, 5])\nsave_to_cache(key="hotels", value=hotels)'
)
Case = tuple[str, str, str, dict[str, object], dict[str, object]]
CASES: list[Case] = [
    # (gold plan, generated code, error, gold summary after, generated
    # summary after)
    (HOTELS, HOTELS, "success", {"hotels": "a"}, {"hotels": "a"}),
    (HOTELS, 'print("No planning needed")', "success", {"hotels": "a"}, {}),
    (HOTELS, 'print("a")\nprint("b")', "success", {"hotels": "a"}, {}),
    (HOTELS, "", "success", {"hotels": "a"}, {}),
    (HOTELS, "x = (", "invalid syntax", {"hotels": "a"}, {}),
    (HOTELS, "x = 1", "success", {"hotels": "a"}, {}),
    (HOTELS, HOTELS.replace("Austin", "Dallas"), "success", {"hotels": "a"}, {"k": "b"}),
    (HOTELS, HOTELS + '\nprint("done")', "name 'y' is not defined", {"hotels": "a"}, {"x": "a"}),
    (
        HOTELS,
        'search_hotels(city="Austin", stars=[5, 4])\nsearch_hotels(city="X")',
        "success",
        {},
        {},
    ),
    ('seek_information("We need dates.")', 'seek_information("Dates?")', "success", {}, {}),
    ('seek_information("We need dates.")', "", "success", {"a": "s"}, {"a": "s"}),
    (
        'h = get_results_from_cache(key="hotels")',
        "h = get_results_from_cache(key='hotels')",
        "success",
        {"hotels": "s"},
        {"hotels": "s"},
    ),
    ("   ", 'print("No planning needed")', "success", {}, {}),
    ("   ", "search_hotels(city='A')", "boom", {}, {"h": "x"}),
]


@needs_clone
@pytest.mark.parametrize("case", CASES, ids=range(len(CASES)))
def test_per_turn_components_match_upstreams_row_functions(case: Case) -> None:
    upstream = _row_functions()
    gold, code, error, gold_after, after = case
    row: dict[str, object] = {
        "role": "user",
        "Filled_Plan": gold,
        "generated_code": code,
        "code_error": error,
        "cache_query_history": gold_after,
        "entire_planner_cache_history": after,
    }
    row["actual_tool_calls"] = _pair(_call(upstream, "extract_actual_tool_calls", row))[0]
    row["generated_tool_calls"] = _pair(_call(upstream, "extract_generated_tool_calls", row))[0]
    calling, param = _pair(_call(upstream, "tool_call_evaluation_metrics", row))
    ours = turn_metrics(gold, code, error, gold_after, after)

    assert gold_calls(gold) == row["actual_tool_calls"]

    for prefix, reference in (("tool", calling), ("param", param)):
        expected = None if reference is None else [_dict(reference)[n] for n in ("tp", "fp", "fn")]
        found = [ours[f"{prefix}_{n}"] for n in ("tp", "fp", "fn")]
        assert found == (expected or [None, None, None])

    if calling is not None:
        assert round(float(ours["tool_f1"]), 4) == _dict(calling)["f1"]  # type: ignore[arg-type]

    assert ours["non_executable"] == _call(upstream, "find_non_executable_code", row)
    assert ours["cache_em"] == _call(upstream, "cache_summary_exact_match", row)


@needs_clone
@pytest.mark.parametrize(
    "output",
    [
        "",
        "no code",
        "<CODE>a()</CODE>",
        "<CODE>\n a() \n</CODE>",
        "<CODE>unterminated",
        "<code>lower()</code>",
    ],
)
def test_extraction_matches_upstreams_on_one_block(output: str) -> None:
    expected = _call(_row_functions(), "extract_code_from_generated_plan", output)

    assert extract_code(output)[0] == expected


@needs_clone
def test_extraction_departs_from_upstreams_on_several_blocks() -> None:
    """Upstream joins every block, which runs a trace's drafts; the last wins here."""
    output = "<CODE>\n a() \n</CODE>x<CODE>b()</CODE>"

    assert _call(_row_functions(), "extract_code_from_generated_plan", output) == "a()\nb()"
    assert extract_code(output)[0] == "b()"


# ---------------------------------------------------------- aggregation parity


def _records() -> list[dict[str, object]]:
    """Hand-made per-turn records over two domains and three files, CASES reused."""
    out = []

    for index, (gold, code, error, gold_after, after) in enumerate(CASES * 3):
        domain = ("hotel", "flighthotel")[index % 2]
        out.append(
            {
                "domain": domain,
                "file_index": 1 + index % 3,
                "ID": str(index),
                "gold": gold,
                "code": code,
                "error": error,
                "gold_after": gold_after,
                "after": after,
            }
        )

    return out


def _run_upstream_steps(tmp_path: Path, planning: Path, root: Path | None = None) -> Path:
    """Run upstream's steps 2 and 3 on a planning directory; return the eval directory."""
    env = _upstream_env(tmp_path, root)
    evaluation = tmp_path / "eval_output"
    metrics = tmp_path / "metrics.csv"
    subprocess.run(
        [sys.executable, str(CLONE / "evaluation" / "generate_evaluation_metrics.py")],
        env={**env, "INPUT_DIR": str(planning), "OUTPUT_DIR": str(evaluation)},
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(CLONE / "evaluation" / "compute_aggregate_metrics.py"),
            "--input_dir",
            str(evaluation),
            "--output_csv",
            str(metrics),
        ],
        env=env,
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return evaluation


@needs_clone
def test_aggregation_matches_upstreams_steps_2_and_3(tmp_path: Path) -> None:
    planning = tmp_path / "processed_output"
    ours: dict[str, list[dict[str, object]]] = {}

    for record in _records():
        meta = turn_metrics(
            str(record["gold"]),
            str(record["code"]),
            str(record["error"]),
            _dict(record["gold_after"]),
            _dict(record["after"]),
        )
        meta.update(domain=record["domain"], hf_split="test", file_index=record["file_index"])
        ours.setdefault(str(record["domain"]), []).append(meta)
        name = f"file_{record['file_index']}_planning.csv"
        path = planning / str(record["domain"]) / "test" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()

        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)

            if new:
                writer.writerow(
                    [
                        "ID",
                        "Filled_Plan",
                        "role",
                        "generated_code",
                        "code_error",
                        "cache_query_history",
                        "entire_planner_cache_history",
                    ]
                )

            writer.writerow(
                [
                    record["ID"],
                    record["gold"],
                    "user",
                    record["code"],
                    record["error"],
                    str(record["gold_after"]),
                    str(record["after"]),
                ]
            )

    _run_upstream_steps(tmp_path, planning)
    theirs = pd.read_csv(tmp_path / "metrics.csv").set_index("domain")

    for domain, rows in ours.items():
        mine = upstream_metrics(rows)

        for name in FIELDS:
            assert mine[name] == pytest.approx(float(theirs.loc[domain, name]), abs=1e-9), name


# ------------------------------------------------------------------ end to end


def _perturb_argument(plan: str) -> str:
    """Gold with one argument value of one T1 tool call changed."""
    tree = ast.parse(plan)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in t1_build.T1_TOOLS
        ):
            for keyword in node.keywords:
                value = keyword.value

                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    value.value += " (changed)"
                    return ast.unparse(tree)

                if isinstance(value, ast.List) and value.elts:
                    first = value.elts[0]

                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        first.value += " (changed)"
                        return ast.unparse(tree)

    return plan


def _drop_saves(plan: str) -> str:
    """Gold without its save_to_cache lines."""
    return "\n".join(line for line in plan.splitlines() if "save_to_cache(" not in line)


PREDICTORS = {
    "oracle": lambda plan: "" if plan is None else f"<CODE>\n{plan.strip()}\n</CODE>",
    "dummy": lambda plan: '<CODE>print("No planning needed")</CODE>',
    "perturbed_argument": lambda plan: (
        "" if plan is None else f"<CODE>{_perturb_argument(plan)}</CODE>"
    ),
    "perturbed_no_save": lambda plan: "" if plan is None else f"<CODE>{_drop_saves(plan)}</CODE>",
}


def _free_names(code: str) -> set[str]:
    """Names the code reads before binding them, beyond the exec namespace and builtins."""
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    bound: set[str] = set(GLOBALS) | set(dir(__builtins__))
    free: set[str] = set()

    for statement in tree.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                free |= {node.id} - bound

        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)

    return free


@pytest.mark.integration
@needs_clone
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_upstreams_pipeline_agrees_turn_by_turn(tmp_path: Path) -> None:
    """All three upstream steps on file 1 of every domain's test split.

    Upstream scores every user turn and runs on Python 3.12 here, so with its
    file-wide semantics. Two kinds of turn may disagree, and each one that
    does is named: leak, where our session replay and upstream's file-wide
    replay disagree on gold; carried, where the generated code reads a name
    an earlier turn bound, which upstream takes from the generation's own
    earlier code and we from gold. Tool and parameter counts depend on the
    code alone and must agree on every turn.
    """
    files = [dialogue_file(domain, "test", 1) for domain in DOMAINS]
    root = snapshot(files + sorted(set(ONTOLOGY.values())))
    tools = Tools(str(root))
    leaks: set[str] = set()
    dialogues: dict[str, list[list[str | None]]] = {}

    for name in files:
        domain = name.split("/")[0]
        entries = read_dialogues(root / name)
        records = {s: replay_file(tools, [e.rows for e in entries], s) for s in ("session", "file")}

        for index, entry in enumerate(entries):
            rows = entry.rows
            dialogues[f"{domain}:{entry.id}"] = rows

            for turn, row in enumerate(user_turns(rows)):
                prefix = range(row + 1)
                session, file = records["session"][index], records["file"][index]
                differs = any(
                    (session[i] or {}).get(k) != (file[i] or {}).get(k)
                    for i in prefix
                    for k in ("before", "after", "error")
                )
                raised = any(
                    (session[i] or {}).get("error") not in (None, "", "success") for i in prefix
                )

                if differs or raised:
                    leaks.add(f"{domain}:{entry.id}:{turn}")

    executor = Executor(str(root))
    report: dict[str, dict[str, list[str]]] = {}

    try:
        for predictor, answer in PREDICTORS.items():
            ours: dict[str, dict[str, object]] = {}
            codes: dict[str, str] = {}
            inputs = tmp_path / predictor / "input"

            for name in files:
                domain = name.split("/")[0]
                frame = pd.read_csv(root / name, sep=",", quoting=csv.QUOTE_MINIMAL, dtype=str)
                frame["generated_plan"] = ""

                for key, rows in dialogues.items():
                    if not key.startswith(domain + ":"):
                        continue

                    template_id = key.split(":")[1]
                    positions = list(frame.index[frame["ID"] == template_id])

                    for turn, row in enumerate(user_turns(rows)):
                        output = answer(rows[row][1])
                        frame.loc[positions[row], "generated_plan"] = output
                        code = extract_code(output)[0]
                        run = executor.execute(
                            replay_plans(rows[:row]), rows[row][1], code, 60, strict=False
                        )
                        ours[f"{key}:{turn}"] = turn_metrics(
                            rows[row][1], code, run.error, run.gold_after, run.after, run.executed
                        ) | {"error": run.error}
                        codes[f"{key}:{turn}"] = code

                path = inputs / domain / "test" / Path(name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)

            env = _upstream_env(tmp_path, root)
            processed = tmp_path / predictor / "processed_output"
            subprocess.run(
                [sys.executable, str(CLONE / "evaluation" / "process_model_output.py")],
                env={**env, "INPUT_DIR": str(inputs), "OUTPUT_DIR": str(processed)},
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )
            evaluation = _run_upstream_steps(tmp_path / predictor, processed, root)
            classes: dict[str, list[str]] = {"agree": [], "leak": [], "carried": []}

            for path in sorted(evaluation.rglob("*_eval.csv")):
                domain = path.parts[-3]
                frame = pd.read_csv(path)
                turns: Counter[object] = Counter()

                for _, row in frame.iterrows():
                    if row["role"] != "user":
                        continue

                    key = f"{domain}:{row['ID']}:{turns[row['ID']]}"
                    turns[row["ID"]] += 1
                    mine = ours[key]

                    for kind, column in (
                        ("tool", "tool_calling_metrics"),
                        ("param", "tool_param_metrics"),
                    ):
                        cell = row[column]
                        theirs = ast.literal_eval(cell) if isinstance(cell, str) else None
                        wanted = None if theirs is None else [theirs[n] for n in ("tp", "fp", "fn")]
                        found = [mine[f"{kind}_{n}"] for n in ("tp", "fp", "fn")]
                        assert found == (wanted or [None, None, None]), (predictor, key, column)

                    agree = (
                        mine["non_executable"] == int(row["non_executable_code"])
                        and mine["cache_em"] == float(row["cache_summary_exact_match"])
                        and mine["error"] == str(row["code_error"])
                    )

                    if agree:
                        classes["agree"].append(key)
                    elif key in leaks:
                        classes["leak"].append(key)
                    elif _free_names(codes[key]):
                        classes["carried"].append(key)
                    else:
                        pytest.fail(
                            f"{predictor} {key}: ours {mine}, upstream "
                            f"non_executable={row['non_executable_code']} "
                            f"cache_em={row['cache_summary_exact_match']} "
                            f"error={row['code_error']!r}, and neither a leak nor carried state"
                        )

            assert sum(map(len, classes.values())) == len(ours)
            report[predictor] = classes
    finally:
        executor.close()

    for predictor, classes in report.items():
        print(f"{predictor}: {len(classes['agree'])} turns agree")

        for name in ("leak", "carried"):
            print(f"  {name} ({len(classes[name])}): {', '.join(sorted(classes[name]))}")

    # Gold's own code carries nothing gold would not; the do-nothing code
    # reads no name. Either can only disagree where gold leaks.
    assert not report["oracle"]["carried"]
    assert not report["dummy"]["carried"]


# ------------------------------------------------------------ oracle and dummy


class Fake:
    """Answers each item in turn; evaluate_split rolls them out in order."""

    def __init__(self, outputs: list[str]) -> None:
        """Remember the answers, in item order."""
        self.outputs = iter(outputs)

    def generate(self, messages: list[dict[str, object]], image: object = None) -> str:
        """The next answer."""
        return next(self.outputs)


@pytest.mark.integration
def test_the_oracle_and_the_do_nothing_floor(tmp_path: Path) -> None:
    """The gold target scores every metric at its maximum; doing nothing, the floor.

    Rolled out through evaluate_split into the per-item JSONL skillpng eval
    writes, and read back with the report, which is printed.
    """
    from skillpng.benchmarks.t1.t1_report import build_report, render
    from skillpng.evaluate import evaluate_split
    from skillpng.results import PredictionLog

    bench = T1(BenchmarkCfg(name="t1", path=HF_REPO, system_prompt=SYSTEM_PROMPT))
    summaries: dict[str, dict[str, object]] = {}
    dummy = '<CODE>print("No planning needed")</CODE>'

    for split, name, answer in (
        ("test", "oracle", None),
        ("test", "dummy", dummy),
        ("val", "dummy", dummy),
    ):
        items = bench.items(split)
        outputs = [item.target if answer is None else answer for item in items]
        path = tmp_path / f"{split}-{name}.jsonl"

        with PredictionLog(path) as log:
            score, _ = evaluate_split(Fake(outputs), bench, None, items, predictions=log)  # type: ignore[arg-type]

        report = build_report(path)
        print(f"\n#### {split} {name}: mean score {score:.4f}\n" + render(report))
        summaries[f"{split}-{name}"] = _dict(_dict(report["conditions"])[str(path)])

    bench.close()
    oracle, floor = summaries["test-oracle"], summaries["test-dummy"]
    macro = _dict(oracle["macro"])

    assert oracle["headline"] == 1.0
    assert _dict(oracle["timeouts"])["count"] == 0

    for name in (
        "code_success_rate",
        "cache_summary_exact_match_avg",
        "tool_calling_micro_precision",
        "tool_calling_micro_recall",
        "tool_calling_micro_f1",
        "tool_calling_micro_accuracy",
        "tool_param_micro_precision",
        "tool_param_micro_recall",
        "tool_param_micro_f1",
        "tool_param_micro_accuracy",
    ):
        assert macro[name] == 1.0, name

    # Every seek_only and cache_read_only item, plus 2 tools items whose gold
    # leaves the summary values unchanged.
    assert floor["headline"] == 347 / 1400
    assert summaries["val-dummy"]["headline"] == 44 / 200
    assert _dict(floor["macro"])["code_success_rate"] == 0.0
    assert floor["code_success_per_item"] == 0.0
    assert _dict(floor["no_calls"])["count"] == 1400

    for name in ("tool_calling_micro_f1", "tool_param_micro_f1"):
        assert _dict(floor["macro"])[name] == 0.0


# ----------------------------------------------------------- environment drift


@pytest.mark.integration
@needs_clone
def test_gold_replay_matches_upstreams_pins(tmp_path: Path) -> None:
    """Every dialogue prefix the split uses, replayed here and under upstream's pins.

    The pinned side runs t1_exec.py by path with Python 3.11, pydantic 2.10.6
    and pandas < 3, against the clone's own t1 package.
    """
    entries = [
        entry
        for split in SPLITS
        for entry in json.loads((t1_build.SPLIT_ROOT / split / "items.json").read_text())
    ]
    files = sorted({dialogue_file(e["domain"], e["hf_split"], e["file_index"]) for e in entries})
    root = snapshot(files + sorted(set(ONTOLOGY.values())))
    dialogues = {f"{name}:{d.id}": d.rows for name in files for d in read_dialogues(root / name)}
    prefixes = {}

    for e in entries:
        rows = dialogues[
            f"{dialogue_file(e['domain'], e['hf_split'], e['file_index'])}:{e['template_id']}"
        ]
        prefixes[e["id"]] = rows[: user_turns(rows)[e["turn"]] + 1]

    script = Path(t1_build.__file__).parent / "t1_exec.py"
    outputs = {}

    for side, package, command in (
        ("ours", VENDORED, [sys.executable]),
        (
            "pinned",
            "t1",
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.11",
                "--with",
                "pydantic==2.10.6",
                "--with",
                "pandas<3",
                "--with",
                "jinja2",
                "python",
            ],
        ),
    ):
        request = tmp_path / f"{side}.json"
        request.write_text(
            json.dumps({"root": str(root), "package": package, "prefixes": prefixes})
        )
        env = {**os.environ, "PYTHONPATH": str(CLONE / "src") if side == "pinned" else ""}
        subprocess.run(
            [*command, str(script), str(request)],
            env=env,
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        outputs[side] = json.loads(Path(str(request) + ".out").read_text())

    assert len(outputs["ours"]) == len(prefixes) == 2000
    assert outputs["ours"] == outputs["pinned"]
