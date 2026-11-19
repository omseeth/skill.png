"""T1 (Chakraborty et al., NeurIPS 2025): multi-turn travel planning as tool-calling code.

**The split is fixed, not computed.** data/t1_id_split/{train,val,test}/items.json
lists the exact 400 / 200 / 1400 user turns this project runs on, drawn from
T1's train, validation and test splits respectively; see t1_build. An item is
one user turn, scored teacher-forced: the model always sees the gold
conversation so far and the gold cache summary, never its own earlier output.

Every item is one direct chat exchange: the system turn from
benchmark.system_prompt, then one user turn filled from the shipped zero-shot
template data/t1_prompt.txt (upstream's prompt without its few-shot block,
with a code-only OUTPUT FORMAT that states the conventions the block showed).
The model answers with Python inside <CODE> </CODE> tags, which is also the
shape the training target is wrapped in.

Scoring executes that code, which is why this adapter overrides rollout:

- Code is the last <CODE> block, stripped, after anything up to the last
  </think> is dropped, so a reasoning trace's drafts are never executed.
  Upstream's evaluation joins every block; on one block the two agree.
- It runs in the executor's worker process against the item's teacher-forced
  state: the gold session locals and the gold cache as they stand before the
  turn. See t1_exec.
- The per-turn components replicate the row functions of upstream's
  evaluation/generate_evaluation_metrics.py on top of its vendored
  eval_metrics. Every component rides in Result.meta; t1_report aggregates
  them the way upstream does. BLEU on seek_information text is not computed.

Three things are protocol, and changing any of them makes scores incomparable
with earlier runs: the prompt template (digest-checked at load), the system
prompt in configs/t1.yaml, and the replay semantics of t1_exec.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

from ...config import BenchmarkCfg
from ..base import Benchmark, Item, Result, inject_image
from ..registry import BENCHMARKS
from .t1_build import (
    CACHE_SLOT,
    CONVERSATION_SLOT,
    HF_REPO,
    HF_REVISION,
    ITEM_FIELDS,
    PROMPT_PATH,
    SPLIT_ROOT,
    SPLITS,
    UPSTREAM_COMMIT,
    dialogue_file,
    digest,
    gold_kind,
    ontology_root,
    read_dialogues,
    snapshot,
    upstream_digest,
    user_turns,
)
from .t1_exec import SUCCESS, Executor, Row, replay_plans
from .upstream.evaluation.eval_metrics import (
    calculate_tool_calling_metrics,
    calculate_tool_param_metrics,
    extract_tool_calls,
)

log = logging.getLogger(__name__)

METRICS = ("cache_em", "tool_f1", "code_success")
METRIC_NAMES = {
    "cache_em": "cache_summary_exact_match",
    "tool_f1": "tool_calling_f1",
    "code_success": "code_success",
}

# The shape the gold plan is trained in, matching the layout of upstream's
# few-shot examples. End-of-turn tokens stay with model.append_eos and
# model.target_suffix, as for every other benchmark.
CODE_TEMPLATE = "<CODE>\n{code}\n</CODE>"
# Bumped whenever the built items change shape, so an old item cache is
# never read as a new one.
BUILDER_VERSION = 1
# Where built items are cached; None means <HF_HOME>/skillpng/t1.
ITEM_CACHE_ROOT: Path | None = None

# A block holds no second opener, so a trace that only names the tag ("wrap
# it in <CODE> tags") never swallows the real block after it.
_CODE = re.compile(r"<CODE>((?:(?!<CODE>).)*?)</CODE>", re.DOTALL)
_THINK_END = "</think>"
_SLOTS = re.compile(re.escape(CONVERSATION_SLOT) + "|" + re.escape(CACHE_SLOT))


@dataclass(frozen=True)
class T1Options:
    """benchmark.options for this adapter. Unknown keys are rejected, as in the config schema."""

    metric: str = "cache_em"
    # Seconds generated code may run in the worker before it is killed and
    # the turn recorded as a timeout. Gold replay is not counted against it.
    exec_timeout_s: float = 30.0

    @classmethod
    def parse(cls, options: dict[str, object]) -> T1Options:
        """Validate the raw options mapping and build the options object.

        Args:
            options: Raw benchmark.options mapping from the config.

        Returns:
            Parsed options with a metric known to METRICS and a positive timeout.
        """
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(options) - known)

        if unknown:
            raise ValueError(f"unknown t1 option(s) {unknown}; expected {sorted(known)}")

        parsed = cls(**options)  # type: ignore[arg-type]

        if parsed.metric not in METRICS:
            raise ValueError(f"metric={parsed.metric!r}; expected one of {METRICS}")

        timeout = parsed.exec_timeout_s

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"exec_timeout_s={timeout!r}; must be a positive number of seconds")

        return parsed


@BENCHMARKS.register("t1")
class T1(Benchmark):
    def __init__(self, cfg: BenchmarkCfg) -> None:
        """Parse options and load the prompt; items and the worker come lazily.

        Args:
            cfg: Benchmark configuration for this instance.
        """
        super().__init__(cfg)
        self.repo_id = _repo_id(cfg.path)
        self.opts = T1Options.parse(cfg.options)
        self.template = load_template()
        self._splits: dict[str, list[Item]] | None = None
        self._executor: Executor | None = None

    @property
    def metric_name(self) -> str:
        """Name the selected metric goes by in logs and results.

        Args:
            None.

        Returns:
            The reporting name of the metric option.
        """
        return METRIC_NAMES[self.opts.metric]

    @property
    def executor(self) -> Executor:
        """The worker handle, created on first use; the worker itself starts later still.

        Args:
            None.

        Returns:
            The executor bound to the ontology of the pinned revision.
        """
        if self._executor is None:
            self._executor = Executor(str(ontology_root(self.repo_id)))
            atexit.register(self._executor.close)

        return self._executor

    def close(self) -> None:
        """Stop the worker, if one is running.

        Args:
            None.
        """
        if self._executor is not None:
            self._executor.close()

    def items(self, split: str) -> list[Item]:
        """Items of one split, materializing every split on first use.

        Args:
            split: Name of the split to load.

        Returns:
            Items in that split, in manifest order.
        """
        if self._splits is None:
            self._splits = self._build_splits()

        if split not in self._splits:
            raise KeyError(f"unknown split {split!r}; have {sorted(self._splits)}")

        return self._splits[split]

    def build_messages(
        self, item: Item, with_image: bool, site: str = "pre_query"
    ) -> list[dict[str, object]]:
        """The configured system turn plus the filled zero-shot template.

        The text is byte-identical with and without the image; only the image
        block differs, which is what makes the paired delta a paired delta.

        Args:
            item: Item to build the prompt for.
            with_image: Whether to reserve an image slot.
            site: Injection site for the image, pre_query or post_query.

        Returns:
            Chat messages for the item.
        """
        text = fill_prompt(self.template, str(item.meta["conversation"]), str(item.meta["cache"]))
        user = {"role": "user", "content": [{"type": "text", "text": text}]}
        messages = [*self.system_messages(), user]

        return inject_image(messages, site) if with_image else messages

    def score(self, output: str, item: Item) -> float:
        """Execute the output's code and score it on the selected metric.

        Args:
            output: Text generated by the model.
            item: Item the output was generated for.

        Returns:
            The selected component of the scoring breakdown.
        """
        return float(self.breakdown(output, item)[self.opts.metric])  # type: ignore[arg-type]

    def breakdown(self, output: str, item: Item) -> dict[str, object]:
        """Every per-turn component for one output, plus the keys a report groups by.

        Args:
            output: Text generated by the model.
            item: Item the output was generated for.

        Returns:
            The extraction, the execution outcome and the component metrics.
        """
        code, blocks, thought = extract_code(output)
        gold = item.ground_truth if isinstance(item.ground_truth, str) else None
        replay = item.meta["replay"]
        assert isinstance(replay, list)
        run = self.executor.execute(replay, gold, code, float(self.opts.exec_timeout_s))

        if run.before != item.meta["cache"] or run.gold_error != SUCCESS:
            raise RuntimeError(
                f"t1 item {item.id!r}: replaying its gold gives pre-turn cache "
                f"{run.before[:200]!r} and gold error {run.gold_error!r}, but the item was "
                "built with the shipped cache digest and a gold plan that runs. The replay "
                "has drifted; rebuild the item cache deliberately."
            )

        metrics = turn_metrics(gold, code, run.error, run.gold_after, run.after, run.executed)
        keys = ("domain", "hf_split", "file_index", "template_id", "turn", "gold_kind")

        return {
            **{name: item.meta[name] for name in keys},
            "code": code,
            "n_blocks": blocks,
            "think_stripped": thought,
            "error": run.error,
            "error_type": run.error_type,
            "timeout": run.timeout,
            "executed": run.executed,
            **metrics,
        }

    async def rollout(
        self, item: Item, model: object, image: object = None, site: str = "pre_query"
    ) -> Result:
        """Generate, execute, and carry every component through Result.meta.

        score() returns one number and evaluate_split averages exactly that, so
        everything else this benchmark measures rides here instead.

        Args:
            item: Item to roll out.
            model: Model used to generate the output.
            image: Image to inject, or None for a text-only rollout.
            site: Injection site for the image, pre_query or post_query.

        Returns:
            Result holding the output, the headline score and the breakdown.
        """
        messages = self.build_messages(item, with_image=image is not None, site=site)
        output = model.generate(messages, image)  # type: ignore[attr-defined]
        scores = self.breakdown(output, item)

        return Result(
            item_id=item.id,
            output=output,
            score=float(scores[self.opts.metric]),  # type: ignore[arg-type]
            meta=scores,
        )

    def _build_splits(self) -> dict[str, list[Item]]:
        """The manifest's turns, from the item cache or built in the worker.

        Args:
            None.

        Returns:
            Split name to its items, capped by cfg.limit when set.
        """
        entries = {name: _split_entries(name) for name in SPLITS}
        path = item_cache_dir() / cache_key(self.repo_id) / "items.json"
        built = _read_cache(path)

        if built is None:
            log.info("t1: building %d items into %s", sum(map(len, entries.values())), path)
            built = self._build(entries)
            _write_cache(path, built)

        splits = {
            name: [_to_item(entry, built.get(str(entry["id"]))) for entry in split]
            for name, split in entries.items()
        }

        if self.cfg.limit:
            # A smoke-run cap: the prefix of each split in manifest order, so
            # the same items are taken every time.
            splits = {name: split[: self.cfg.limit] for name, split in splits.items()}

        for name, split_items in splits.items():
            log.info("t1 %s: %d items", name, len(split_items))

        return splits

    def _build(self, entries: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, object]]:
        """Build every manifest turn by replaying its dialogue in the worker.

        Args:
            entries: Split name to its manifest entries.

        Returns:
            Item id to its built record.
        """
        wanted = [entry for split in entries.values() for entry in split]
        files = sorted(
            {
                dialogue_file(str(e["domain"]), str(e["hf_split"]), int(e["file_index"]))  # type: ignore[call-overload]
                for e in wanted
            }
        )
        root = snapshot(files, self.repo_id)
        dialogues: dict[str, list[Row]] = {}

        for name in files:
            for dialogue in read_dialogues(root / name):
                dialogues[f"{name}:{dialogue.id}"] = dialogue.rows

        built: dict[str, dict[str, object]] = {}

        for count, entry in enumerate(wanted, 1):
            built[str(entry["id"])] = self._build_item(entry, dialogues)

            if count % 250 == 0:
                log.info("t1: built %d of %d items", count, len(wanted))

        return built

    def _build_item(
        self, entry: dict[str, object], dialogues: dict[str, list[Row]]
    ) -> dict[str, object]:
        """Build one turn, refusing it unless it still satisfies eligibility rules 1 and 2.

        Args:
            entry: The item's manifest entry.
            dialogues: File path and ID to the dialogue's rows.

        Returns:
            The record an Item is made from.
        """
        item_id = str(entry["id"])
        name = dialogue_file(str(entry["domain"]), str(entry["hf_split"]), int(entry["file_index"]))  # type: ignore[call-overload]
        rows = dialogues.get(f"{name}:{entry['template_id']}")
        turns = user_turns(rows) if rows is not None else []
        turn = int(entry["turn"])  # type: ignore[call-overload]

        if rows is None or turn >= len(turns):
            raise ValueError(
                f"t1 item {item_id!r} is not in {name} at {HF_REVISION}; a partial split is "
                "not this split"
            )

        index = turns[turn]
        plan = rows[index][1]

        if plan is None:
            raise ValueError(f"t1 item {item_id!r} has no gold plan (eligibility rule 1)")

        result = self.executor.build(rows[: index + 1])
        failed = [error for error in result.errors if error[1] != SUCCESS]

        if failed or result.gold_error != SUCCESS:
            raise ValueError(
                f"t1 item {item_id!r}: a gold plan of its dialogue raises under session replay "
                f"({failed or result.gold_error}); eligibility rule 2 says it cannot ship"
            )

        return {
            "id": item_id,
            "target": build_target(plan),
            "ground_truth": plan,
            "group": f"{entry['domain']}:{entry['template_id']}",
            "meta": {
                # evaluate.py copies meta["question"] into the per-item log.
                "question": result.question,
                "domain": entry["domain"],
                "hf_split": entry["hf_split"],
                "file_index": entry["file_index"],
                "template_id": entry["template_id"],
                "turn": entry["turn"],
                "gold_kind": gold_kind(plan),
                "conversation": result.conversation,
                "cache": result.cache,
                # The earlier gold plans, in order: a rollout replays them to
                # rebuild the teacher-forced state without the CSVs.
                "replay": replay_plans(rows[:index]),
            },
        }


def load_template(path: Path | None = None, manifest: dict[str, object] | None = None) -> str:
    """Read the shipped zero-shot template and check it against the manifest's digest.

    Args:
        path: Template file, or None for the shipped one.
        manifest: Split manifest, or None for the shipped one.

    Returns:
        The template, with each sentinel exactly once.
    """
    path = PROMPT_PATH if path is None else path
    manifest = _manifest() if manifest is None else manifest

    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"t1 prompt template is missing {path}; build it with "
            "`python -m skillpng.benchmarks.t1.t1_build prompt`"
        ) from exc

    sha = hashlib.sha256(raw).hexdigest()

    if sha != manifest.get("prompt_sha256"):
        raise ValueError(
            f"{path}: sha256 is {sha}, the manifest records {manifest.get('prompt_sha256')}. "
            "The template carries meaningful trailing whitespace, so an editor that strips "
            "it changes every prompt; restore the file or rebuild it with t1_build prompt."
        )

    template = raw.decode()

    for slot in (CONVERSATION_SLOT, CACHE_SLOT):
        if template.count(slot) != 1:
            raise ValueError(f"{path}: {slot} occurs {template.count(slot)} times, expected once")

    return template


def fill_prompt(template: str, conversation: str, cache: str) -> str:
    """Fill the template's two sentinels in one pass.

    Never str.format: the template holds literal braces, and so do
    conversations. One pass also means a sentinel inside a conversation is
    left as it is.

    Args:
        template: The zero-shot template.
        conversation: The conversation so far, as inference.py formats it.
        cache: The pre-turn cache summary dict, as a string.

    Returns:
        The user turn's text.
    """
    values = {CONVERSATION_SLOT: conversation, CACHE_SLOT: cache}

    return _SLOTS.sub(lambda match: values[match.group(0)], template)


def build_target(plan: str) -> str:
    """The gold continuation: the stripped plan inside one code block.

    Args:
        plan: The turn's gold Filled_Plan.

    Returns:
        The tagged code the loss is taken on.
    """
    return CODE_TEMPLATE.format(code=plan.strip())


def extract_code(output: str) -> tuple[str, int, bool]:
    """The code a generation holds: its last <CODE> block.

    Anything up to the last </think> is dropped and the last block wins, so
    draft code in a reasoning trace never runs. Upstream joins every block
    instead, which would run the drafts of a trace with no </think>: Gemma's,
    once vLLM strips its channel tokens. On a single block the two agree.

    Args:
        output: Text generated by the model.

    Returns:
        The last <CODE> block after the last </think>, stripped ("" when
        there is none), the number of blocks, and whether a reasoning trace
        was dropped.
    """
    thought = _THINK_END in output

    if thought:
        output = output.rsplit(_THINK_END, 1)[1]

    blocks = _CODE.findall(output)
    code = blocks[-1].strip() if blocks else ""

    return code, len(blocks), thought


def gold_calls(plan: str | None) -> list[dict[str, object]] | None:
    """The gold tool calls, as generate_evaluation_metrics.extract_actual_tool_calls.

    Args:
        plan: The turn's gold plan, or None when it has none.

    Returns:
        The calls, or None without a plan.
    """
    return None if plan is None else list(extract_tool_calls(plan))


def generated_calls(code: str) -> tuple[list[dict[str, object]] | None, str]:
    """The generated tool calls, as generate_evaluation_metrics.extract_generated_tool_calls.

    Upstream catches only SyntaxError; any exception from parsing is caught
    here, treated the same way, and reported.

    Args:
        code: The extracted code.

    Returns:
        The calls, None when there is no code, it does not parse, or its only
        call is print, and the parse error's type name ("" when it parsed).
    """
    if not code:
        return None, ""

    try:
        calls = list(extract_tool_calls(code))
    except Exception as error:
        return None, type(error).__name__

    if len(calls) == 1 and "print" in calls[0]:
        return None, ""

    if len(calls) > 1:
        calls = [call for call in calls if "print" not in call]

    return calls, ""


def non_executable(
    actual: list[dict[str, object]] | None, generated: list[dict[str, object]] | None, error: str
) -> int:
    """Upstream's find_non_executable_code for one user turn.

    Args:
        actual: Gold calls.
        generated: Generated calls.
        error: "success" or the generated code's error.

    Returns:
        1 when gold has calls and the generation has none without erroring,
        or errored; else 0.
    """
    if actual and not generated and error == SUCCESS:
        return 1

    if actual and error != SUCCESS:
        return 1

    return 0


def cache_exact_match(actual: dict[str, object] | None, generated: dict[str, object] | None) -> int:
    """Upstream's cache_summary_exact_match for one user turn.

    Keys are ignored: the summary values are compared as multisets.

    Args:
        actual: Post-turn summary after gold.
        generated: Post-turn summary after the generated code.

    Returns:
        1 when both are empty or their values agree, else 0.
    """
    if not actual and not generated:
        return 1

    if actual and generated and Counter(actual.values()) == Counter(generated.values()):
        return 1

    return 0


def turn_metrics(
    gold: str | None,
    code: str,
    error: str,
    gold_after: dict[str, object] | None,
    after: dict[str, object] | None,
    executed: bool = True,
) -> dict[str, object]:
    """Every per-turn component, replicating upstream's row functions.

    Tool and parameter counts exist only when gold and generated calls both
    do, in upstream's sense: None is missing, an empty list is not. A
    generation that did not finish executing has no post-turn cache, so its
    cache match is 0 rather than a comparison.

    Args:
        gold: The turn's gold plan, or None.
        code: The extracted generated code.
        error: "success", the code's error, or the executor's failure.
        gold_after: Post-turn summary after gold.
        after: Post-turn summary after the generated code, None without one.
        executed: Whether the generated code ran to an outcome.

    Returns:
        tp/fp/fn for tool calls and parameters (None where upstream has no
        row), per-turn tool F1, non_executable, code_success and cache_em,
        plus the call counts and the parse error.
    """
    actual = gold_calls(gold)
    generated, parse_error = generated_calls(code)
    out: dict[str, object] = {
        "n_gold_calls": None if actual is None else len(actual),
        "n_gen_calls": None if generated is None else len(generated),
        "no_calls": not generated,
        "parse_error": parse_error,
    }

    for prefix, function in (
        ("tool", calculate_tool_calling_metrics),
        ("param", calculate_tool_param_metrics),
    ):
        both = actual is not None and generated is not None
        counts = function(actual, generated) if both else None

        for name in ("tp", "fp", "fn"):
            out[f"{prefix}_{name}"] = None if counts is None else int(counts[name])

    tp, fp, fn = (int(out[f"tool_{name}"] or 0) for name in ("tp", "fp", "fn"))  # type: ignore[call-overload]
    failed = non_executable(actual, generated, error)
    out["tool_f1"] = _f1(tp, fp, fn)
    out["non_executable"] = failed
    out["code_success"] = float(1 - failed)
    out["cache_em"] = float(cache_exact_match(gold_after, after)) if executed else 0.0

    return out


def item_cache_dir() -> Path:
    """Where built items are cached: shared storage on a cluster, via HF_HOME.

    Args:
        None.

    Returns:
        ITEM_CACHE_ROOT when set, else <HF_HOME>/skillpng/t1.
    """
    if ITEM_CACHE_ROOT is not None:
        return ITEM_CACHE_ROOT

    from huggingface_hub import constants

    return Path(constants.HF_HOME) / "skillpng" / "t1"


def cache_key(repo_id: str = HF_REPO) -> str:
    """Key of the item cache: everything a built item depends on.

    Args:
        repo_id: Hub dataset repo the items are read from.

    Returns:
        A short digest over the repo, the revision, the vendored commit and
        code, BUILDER_VERSION, and the shipped manifest and items files.
    """
    parts = [repo_id, HF_REVISION, UPSTREAM_COMMIT, upstream_digest(), str(BUILDER_VERSION)]

    files = [SPLIT_ROOT / "split_manifest.json"] + [SPLIT_ROOT / s / "items.json" for s in SPLITS]

    for path in files:
        parts.append(hashlib.sha256(path.read_bytes()).hexdigest())

    return digest("\n".join(parts))


def _read_cache(path: Path) -> dict[str, dict[str, object]] | None:
    """Built items from the cache, or None when there are none.

    Args:
        path: The cache file.

    Returns:
        Item id to its built record, or None.
    """
    try:
        document = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None

    items: dict[str, dict[str, object]] = document["items"]

    return items


def _write_cache(path: Path, items: dict[str, dict[str, object]]) -> None:
    """Write the built items atomically: two jobs may build at once.

    Args:
        path: The cache file.
        items: Item id to its built record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")

    with os.fdopen(handle, "w") as out:
        json.dump({"builder_version": BUILDER_VERSION, "items": items}, out)

    os.replace(temporary, path)


def _to_item(entry: dict[str, object], record: dict[str, object] | None) -> Item:
    """Check one built record against its manifest entry and make it an Item.

    Args:
        entry: The item's manifest entry.
        record: Its built record, or None when the build lacks it.

    Returns:
        The Item.
    """
    item_id = str(entry["id"])

    if record is None:
        raise ValueError(f"t1 item {item_id!r} is missing from the built items; delete the cache")

    meta = record["meta"]
    assert isinstance(meta, dict)
    plan = str(record["ground_truth"])
    found = {
        **{name: meta.get(name) for name in ITEM_FIELDS if name in meta},
        "id": record["id"],
        "plan_digest": digest(plan),
        "cache_digest": digest(str(meta["cache"])),
    }
    expected = {name: entry.get(name) for name in ITEM_FIELDS}

    if found != expected:
        raise ValueError(
            f"t1 item {item_id!r} builds {found}, the manifest says {expected}. A Hub-side "
            "change or a replay drift; the split is fixed, so it is refused rather than "
            "rebuilt quietly."
        )

    return Item(
        id=item_id,
        target=str(record["target"]),
        ground_truth=plan,
        group=str(record["group"]),
        meta=dict(meta),
    )


def _manifest() -> dict[str, object]:
    """The shipped split manifest.

    Args:
        None.

    Returns:
        The manifest mapping.
    """
    path = SPLIT_ROOT / "split_manifest.json"

    try:
        document: dict[str, object] = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"t1 id split is missing {path}; build it with "
            "`python -m skillpng.benchmarks.t1.t1_build split`"
        ) from exc

    return document


def _split_entries(split: str) -> list[dict[str, object]]:
    """The entries of one manifest split, in file order, refusing duplicates.

    Args:
        split: Split name, which is also its directory under SPLIT_ROOT.

    Returns:
        Manifest entries listed by that split's items.json.
    """
    path = SPLIT_ROOT / split / "items.json"

    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"t1 id split is missing {path}") from exc

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list of items")

    ids = [str(entry.get("id", "")) for entry in raw]

    if "" in ids or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: every item needs an id, and no id may repeat")

    return [dict(entry) for entry in raw]


def _repo_id(path: str | None) -> str:
    """benchmark.path names the Hub repo; the hf: prefix is optional, empty means HF_REPO.

    Args:
        path: Configured benchmark path, possibly None or hf:-prefixed.

    Returns:
        Hub repo id to download from, defaulting to HF_REPO.
    """
    text = (path or "").strip()

    if text.startswith("hf:"):
        text = text[3:].strip()

    return text or HF_REPO


def _f1(tp: int, fp: int, fn: int) -> float:
    """F1 from counts, as upstream's calculate_tool_calling_metrics, unrounded.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        F1, or 0.0 when precision and recall are both zero.
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
