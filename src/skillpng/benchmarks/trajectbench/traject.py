"""TRAJECT-Bench (parallel, simple): a set of tool calls emitted in one sweep.

**The splits are fixed, not computed.** data/traject_id_split/{train,val,test}/items.json
lists the exact 175 / 89 / 611 rows this project runs on, grouped so that no
(domain, gold provider set) straddles a boundary. They are 875 of the corpus'
1800 rows: a row is dropped when its gold requires a value the query does not
contain, because the schema default is stripped and so no prompt supplies it.
See traject_build.filter_reachable.

Every item is one direct chat exchange: the system turn from
benchmark.system_prompt, then a candidate tool pool and the query. The model
answers with a JSON list of calls inside <answer> tags, which is also the
shape the training target is wrapped in. Nothing is executed.

Three things are protocol rather than preference, and changing any of them
makes scores incomparable with earlier runs:

- The pool is the gold tools plus N_DISTRACTORS same-domain distractors,
  ordered by a hash of the row id, so it is identical across runs, splits and
  the image / no-image conditions. The manifest carries a pool digest.
- Parameter defaults and the code field never reach a prompt: both carry gold
  argument values verbatim, and they are already absent from the shipped tool
  index. Rendering is one function that cannot tell gold from distractor.
- Calls and their parameters are sorted by name before serialization, because
  a parallel trajectory is an unordered set and a sequence target would
  otherwise penalize permutations.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from ...config import BenchmarkCfg
from ..base import Benchmark, Item, Message, Result, _unit_hash, inject_image
from ..registry import BENCHMARKS
from .traject_build import (
    CONFIGS,
    HF_REPO,
    HF_SPLIT,
    SPLIT_ROOT,
    SPLITS,
    TOOL_INDEX,
    index_digest,
    normalize_name,
    row_id,
)

log = logging.getLogger(__name__)

METRICS = ("full_em", "tool_em", "tool_inclusion", "required_em")
METRIC_NAMES = {
    "full_em": "full_exact_match",
    "tool_em": "tool_set_exact_match",
    "tool_inclusion": "tool_inclusion",
    "required_em": "required_param_exact_match",
}

# The shape the gold calls are trained and scored in, and the size of the
# candidate pool the query is answered against. Both are protocol.
ANSWER_TEMPLATE = "<answer>{calls}</answer>"
N_DISTRACTORS = 10
CALL_KEYS = ("tool name", "required parameters", "optional parameters")

# A block holds no second opener, so a trace that only names the tag ("wrap
# it in <answer> tags") never swallows the real block after it.
_ANSWER_TAG = re.compile(r"<answer>((?:(?!<answer>).)*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_END = "</think>"
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s.\-/:]")


@dataclass(frozen=True)
class TrajectOptions:
    """benchmark.options for this adapter. Unknown keys are rejected, as in the config schema."""

    metric: str = "full_em"
    n_distractors: int = N_DISTRACTORS
    # Distractors drawn from the gold tools' own provider before the uniform
    # fill. Sibling endpoints share parameter names and near-identical
    # descriptions, which is the similar-tool confusion the benchmark targets.
    hard_negatives: int = 0
    # None renders descriptions whole, which the measured budgets allow. A
    # count truncates them, identically for gold and distractor tools.
    description_chars: int | None = None
    render_output_info: bool = False

    @classmethod
    def parse(cls, options: dict) -> TrajectOptions:
        """Validate the raw options mapping and build the options object.

        Args:
            options: Raw benchmark.options mapping from the config.

        Returns:
            Parsed options with a metric known to METRICS and sane counts.
        """
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(options) - known)

        if unknown:
            raise ValueError(f"unknown traject option(s) {unknown}; expected {sorted(known)}")

        parsed = cls(**options)

        if parsed.metric not in METRICS:
            raise ValueError(f"metric={parsed.metric!r}; expected one of {METRICS}")

        if parsed.n_distractors < 0:
            raise ValueError(f"n_distractors={parsed.n_distractors}; must not be negative")

        if not 0 <= parsed.hard_negatives <= parsed.n_distractors:
            raise ValueError(
                f"hard_negatives={parsed.hard_negatives}; must be between 0 and "
                f"n_distractors={parsed.n_distractors}"
            )

        if parsed.description_chars is not None and parsed.description_chars < 1:
            raise ValueError(f"description_chars={parsed.description_chars}; must be at least 1")

        return parsed


@BENCHMARKS.register("traject_bench")
class TrajectBench(Benchmark):
    def __init__(self, cfg: BenchmarkCfg) -> None:
        """Resolve the mirror repo and options; the corpus is loaded lazily.

        Args:
            cfg: Benchmark configuration for this instance.
        """
        super().__init__(cfg)
        self.repo_id = _repo_id(cfg.path)
        self.opts = TrajectOptions.parse(cfg.options)
        self.index = load_index()
        self._splits: dict[str, list[Item]] | None = None
        self._verify_pool = True

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
    def n_distractors(self) -> int:
        """Distractor tools added to each item's gold tools.

        Returns:
            The n_distractors option the pool is built with.
        """
        return self.opts.n_distractors

    def items(self, split: str) -> list[Item]:
        """Items of one split, materializing every split on first use.

        Args:
            split: Name of the split to load.

        Returns:
            Items in that split, in manifest order.
        """
        if self._splits is None:
            # items() is called once per split by the training loop, and every
            # split is materialized from the same nine-config scan.
            self._splits = self._build_splits()

        if split not in self._splits:
            raise KeyError(f"unknown split {split!r}; have {sorted(self._splits)}")

        return self._splits[split]

    def build_messages(
        self, item: Item, with_image: bool, site: str = "pre_query"
    ) -> list[Message]:
        """The configured system turn plus the tool pool and the query.

        The text is byte-identical with and without the image; only the image
        block differs, which is what makes the paired delta a paired delta.

        Args:
            item: Item to build the prompt for.
            with_image: Whether to reserve an image slot.
            site: Injection site for the image, pre_query or post_query.

        Returns:
            Chat messages for the item.
        """
        messages = [*self.system_messages(), self._query(item)]

        return inject_image(messages, site) if with_image else messages

    def score(self, output: str, item: Item) -> float:
        """Score one output on the selected metric.

        Args:
            output: Text generated by the model.
            item: Item the output was generated for.

        Returns:
            The selected component of the scoring breakdown.
        """
        return float(self.breakdown(output, item)[self.opts.metric])

    def breakdown(self, output: str, item: Item) -> dict:
        """Every component metric for one output, plus the stratification keys.

        Args:
            output: Text generated by the model.
            item: Item the output was generated for.

        Returns:
            Component metrics and the keys a report groups them by.
        """
        gold = item.ground_truth if isinstance(item.ground_truth, list) else []
        scores = score_calls(extract_calls(output), gold)
        scores.update(
            {
                "domain": item.meta.get("domain", ""),
                "tool_count": item.meta.get("tool_count", 0),
                "has_unseen_values": item.meta.get("has_unseen_values", False),
                "pool_size": len(item.meta.get("pool", [])),
            }
        )

        return scores

    async def rollout(
        self, item: Item, model: object, image: object = None, site: str = "pre_query"
    ) -> Result:
        """Generate, score, and carry the component metrics through Result.meta.

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
            score=float(scores[self.opts.metric]),
            meta=scores,
        )

    def _query(self, item: Item) -> Message:
        """The user turn: the rendered candidate pool, then the query verbatim.

        Args:
            item: Item to build the prompt for.

        Returns:
            One user message carrying the tool pool and the query.
        """
        pool = [self.index["by_key"][key] for key in item.meta["pool"]]
        tools = "\n\n".join(
            render_tool(tool, self.opts.description_chars, self.opts.render_output_info)
            for tool in pool
        )
        user = f"## Tools\n{tools}\n\n## Query\n{item.meta['question']}"

        return {"role": "user", "content": [{"type": "text", "text": user}]}

    def _build_splits(self) -> dict[str, list[Item]]:
        """The manifest's ids, materialized from the mirror in manifest order.

        Args:
            None.

        Returns:
            Split name to its items, capped by cfg.limit when set.
        """
        manifest = _manifest()
        defaults = manifest.get("pool_defaults") or {}
        # The pool the prompts are built from, on the record for every run. A
        # run directory can be named for one pool and built from another, and
        # nothing else in a finished run says which one it was.
        log.info(
            "traject pool: n_distractors=%d hard_negatives=%d description_chars=%s "
            "render_output_info=%s metric=%s",
            self.opts.n_distractors,
            self.opts.hard_negatives,
            self.opts.description_chars,
            self.opts.render_output_info,
            self.opts.metric,
        )
        # The shipped digests pin one pool. Raising n_distractors is an
        # advertised knob, so the check applies to the shipped settings and
        # steps aside, loudly, for a deliberately different pool.
        self._verify_pool = (
            int(defaults.get("n_distractors", -1)) == self.opts.n_distractors
            and int(defaults.get("hard_negatives", -1)) == self.opts.hard_negatives
        )

        if not self._verify_pool:
            log.warning(
                "traject: pool settings (n_distractors=%d, hard_negatives=%d) differ from the "
                "manifest's %s, so the shipped pool digests are not checked",
                self.opts.n_distractors,
                self.opts.hard_negatives,
                defaults,
            )
        entries = {name: _split_entries(name) for name in SPLITS}
        wanted = {entry["id"] for split in entries.values() for entry in split}
        records = _records_by_id(self.repo_id, wanted)
        missing = sorted(wanted - set(records))

        if missing:
            raise ValueError(
                f"{len(missing)} of {len(wanted)} manifest id(s) are not in {self.repo_id!r} "
                f"(e.g. {missing[:3]}). An id is config:row_index over the nine parallel "
                "*_simple configs; another mirror does not carry the same rows, and a partial "
                "split is not this split."
            )

        splits = {
            name: [self._to_item(entry, records[entry["id"]]) for entry in split]
            for name, split in entries.items()
        }

        if self.cfg.limit:
            # A smoke-run cap: the prefix of each split in manifest order, so
            # the same items are taken every time.
            splits = {name: split[: self.cfg.limit] for name, split in splits.items()}

        for name, split_items in splits.items():
            log.info("traject %s: %d items", name, len(split_items))

        return splits

    def _to_item(self, entry: dict, record: dict) -> Item:
        """Turn one manifest entry and its mirror row into an Item.

        Args:
            entry: Manifest entry naming the id, group and reporting fields.
            record: Parsed row of the mirror dataset.

        Returns:
            Item carrying the pool, the gold calls and the tagged target.
        """
        item_id = entry["id"]
        calls = canonical_calls(record["calls"])

        if len(calls) != entry["tool_count"]:
            # The split is fixed, so a row that no longer matches the manifest
            # is a broken corpus, not a row to drop.
            raise ValueError(
                f"traject record {item_id!r} has {len(calls)} calls, manifest says "
                f"{entry['tool_count']}"
            )

        gold_keys = [normalize_name(call["tool name"]) for call in calls]
        unknown = sorted(key for key in gold_keys if key not in self.index["by_key"])

        if unknown:
            raise ValueError(
                f"traject record {item_id!r} names gold tool(s) {unknown} that the shipped "
                "tool index does not carry. A silent join failure would surface as "
                "inexplicably low argument accuracy, so it is refused here instead."
            )

        pool = candidate_pool(
            item_id,
            entry["domain"],
            gold_keys,
            self.index,
            self.opts.n_distractors,
            self.opts.hard_negatives,
        )
        digest = pool_digest(pool)

        if self._verify_pool and entry.get("pool_digest") and entry["pool_digest"] != digest:
            raise ValueError(
                f"traject record {item_id!r} builds pool digest {digest}, manifest says "
                f"{entry['pool_digest']}. The sampler has drifted; rebuild the manifest "
                "deliberately rather than changing the experiment quietly."
            )

        recoverable = recoverable_flags(record["query"], calls)
        _check_recoverable(item_id, entry, calls, recoverable)

        return Item(
            id=item_id,
            # The loss is taken on the shape the prompt asks for and the
            # scorer reads back; ground_truth stays the parsed calls.
            target=build_target(calls),
            ground_truth=calls,
            group=entry["group"],
            meta={
                # evaluate.py copies meta["question"] into the per-item log.
                "question": record["query"],
                "domain": entry["domain"],
                "task_name": record["task_name"],
                "tool_count": entry["tool_count"],
                "has_unseen_values": entry["has_unseen_values"],
                "pool": pool,
                "pool_digest": digest,
            },
        )


def load_index(path: Path | None = None) -> dict:
    """Read the shipped tool index and check it against its own digest.

    Args:
        path: Path to the index file, or None for the shipped one.

    Returns:
        The document plus by_key, by_domain and by_parent lookups.
    """
    # Resolved here rather than as a default argument, so a test can point
    # the module constant somewhere else.
    path = TOOL_INDEX if path is None else path
    document: dict = {}

    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"traject tool index is missing {path}; build it with "
            "`python -m skillpng.benchmarks.trajectbench.traject_build index`"
        ) from exc

    tools = document["tools"]
    digest = index_digest(tools)

    if digest != document["digest"]:
        raise ValueError(
            f"{path}: index digest is {digest}, the file claims {document['digest']}. "
            "The rendered pool would differ from the one the manifest was built against."
        )

    by_key: dict = {}
    by_domain: dict = {}
    by_parent: dict = {}

    for tool in tools:
        by_key[tool["key"]] = tool
        by_domain.setdefault(str(tool["domain name"]), []).append(tool["key"])
        by_parent.setdefault(normalize_name(tool["parent tool name"]), []).append(tool["key"])

    document["by_key"] = by_key
    document["by_domain"] = {name: sorted(keys) for name, keys in by_domain.items()}
    document["by_parent"] = {name: sorted(keys) for name, keys in by_parent.items()}

    return document


def render_tool(tool: dict, description_chars: int | None = None, output_info: bool = False) -> str:
    """Render one endpoint for the prompt, showing nothing that names a gold value.

    This function is the leak boundary and it takes no argument saying whether
    the tool is gold, so gold and distractor cannot be rendered differently.
    Parameter defaults and the code field are already absent from the index.

    Args:
        tool: One entry of the shipped tool index.
        description_chars: Character budget per description, or None for whole.
        output_info: Whether the endpoint's output summary is rendered too.

    Returns:
        The endpoint as a markdown block.
    """
    def clip(value: object) -> str:
        """Collapse a description and cut it to the configured budget.

        Args:
            value: Description text of a tool or a parameter.

        Returns:
            The cleaned description, truncated when over budget.
        """
        text = re.sub(r"\s+", " ", str(value or "")).strip()

        if description_chars is None or len(text) <= description_chars:
            return text

        return text[:description_chars].rstrip() + "..."

    lines = [
        f"### {tool['tool name']}",
        f"Provider: {tool['parent tool name']} — {clip(tool['parent tool description'])}",
        f"API: {tool['API name']}",
        f"Description: {clip(tool['tool description'])}",
    ]

    for label, key in (("Required parameters", "required"), ("Optional parameters", "optional")):
        parameters = tool.get(key) or []

        if not parameters:
            lines.append(f"{label}: none")
            continue

        lines.append(f"{label}:")
        lines += [_render_parameter(parameter, clip) for parameter in parameters]

    if output_info and tool.get("output_info"):
        lines.append(f"Output: {clip(tool['output_info'].get('output_info_summary'))}")

    return "\n".join(lines)


def _render_parameter(parameter: dict, clip: object) -> str:
    """Render one parameter, leaving out the parts the schema does not fill in.

    Args:
        parameter: One parameter of an index entry.
        clip: The description shortener the surrounding render is using.

    Returns:
        The parameter as one bullet line.
    """
    line = f"- {parameter['name']}"
    kind = str(parameter.get("type") or "").strip()

    if kind:
        line += f" ({kind})"

    description = clip(parameter.get("description"))  # type: ignore[operator]

    return f"{line}: {description}" if description else line


def candidate_pool(
    item_id: str,
    domain: str,
    gold_keys: list[str],
    index: dict,
    n_distractors: int = N_DISTRACTORS,
    hard_negatives: int = 0,
) -> list[str]:
    """The gold tools plus same-domain distractors, ordered from the row id.

    Every draw is seeded from the row id rather than from module random state,
    which the training loop reseeds: the pool must be identical across runs,
    across splits and across the image and no-image conditions.

    Args:
        item_id: Row id the per-instance seed is derived from.
        domain: Domain the distractors are drawn from.
        gold_keys: Index keys of the row's gold tools.
        index: The loaded tool index with its lookups.
        n_distractors: Number of distractors to add to the gold tools.
        hard_negatives: How many distractors are drawn from the hard tiers.

    Returns:
        The pool's index keys, in the order they are rendered.
    """
    gold = set(gold_keys)
    available = [key for key in index["by_domain"].get(domain, []) if key not in gold]

    if len(available) < n_distractors:
        raise ValueError(
            f"domain {domain!r} offers {len(available)} non-gold tools for {item_id!r}, "
            f"fewer than n_distractors={n_distractors}"
        )

    def shuffled(keys: list[str], tag: str) -> list[str]:
        """Order keys by a stable hash of the row id and the key.

        Args:
            keys: Index keys to order.
            tag: Draw label, so two draws of one row do not share an order.

        Returns:
            The keys in a fixed, row-specific order.
        """
        return sorted(keys, key=lambda key: _unit_hash(f"{item_id}|{tag}|{key}"))

    chosen: list[str] = []

    if hard_negatives:
        # Siblings first: same provider, same domain, not gold. Connected
        # tools are the benchmark's own similarity signal but are empty for
        # most rows, so they are the second tier rather than the only one.
        siblings = {
            key
            for gold_key in gold_keys
            for key in index["by_parent"].get(
                normalize_name(index["by_key"][gold_key]["parent tool name"]), []
            )
            if key not in gold and index["by_key"][key]["domain name"] == domain
        }
        connected = {
            key
            for gold_key in gold_keys
            for key in index["by_key"][gold_key].get("connected", [])
            if key not in gold
            and key not in siblings
            and key in index["by_key"]
            and index["by_key"][key]["domain name"] == domain
        }
        hard = shuffled(sorted(siblings), "sibling") + shuffled(sorted(connected), "connected")
        chosen = hard[:hard_negatives]

    taken = set(chosen)
    fill = [key for key in shuffled(available, "distractor") if key not in taken]
    chosen += fill[: n_distractors - len(chosen)]
    # Ordering is fixed but uncorrelated with gold membership: position bias
    # over a twenty-tool list can swamp the effect being measured.
    return shuffled(sorted(gold) + chosen, "order")


def pool_digest(pool: list[str]) -> str:
    """Fingerprint one ordered pool, so a silent sampler drift becomes an error.

    Args:
        pool: The pool's index keys, in render order.

    Returns:
        The first 16 hex chars of the sha256 of the joined keys.
    """
    import hashlib

    return hashlib.sha256("\n".join(pool).encode()).hexdigest()[:16]


def canonical_calls(calls: object) -> list[dict]:
    """Reduce calls to the three scored keys, sorted by name at both levels.

    Gold parallel calls are an unordered set, so the same canonicalization runs
    over the target and over model output before they are compared.

    Args:
        calls: Parsed calls from a mirror row or from a model output.

    Returns:
        The calls with only CALL_KEYS, sorted by tool name then parameter name.
    """
    out: list[dict] = []

    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict):
            continue

        name = re.sub(r"\s+", " ", str(call.get("tool name", ""))).strip()

        if not name:
            continue

        out.append(
            {
                "tool name": name,
                "required parameters": _parameters(call, "required"),
                "optional parameters": _parameters(call, "optional"),
            }
        )

    return sorted(out, key=lambda call: call["tool name"])


def serialize_calls(calls: list[dict]) -> str:
    """Serialize canonical calls deterministically, byte for byte.

    One function serves the target builder and the scorer, so the two can never
    disagree about what the answer is meant to look like.

    Args:
        calls: Canonical calls to serialize.

    Returns:
        The JSON list the answer tags wrap.
    """
    return json.dumps(calls, ensure_ascii=False, separators=(", ", ": "))


def build_target(calls: list[dict]) -> str:
    """The gold continuation: the canonical calls inside answer tags.

    Args:
        calls: Canonical gold calls of one row.

    Returns:
        The tagged JSON the loss is taken on.
    """
    return ANSWER_TEMPLATE.format(calls=serialize_calls(calls))


def extract_calls(output: str) -> list[dict] | None:
    """Read the answer back: last tag, then last fence, then last JSON value.

    Reasoning checkpoints emit thousands of tokens before answering, so
    anything up to the last </think> is dropped and the last tagged block
    wins; a draft inside a reasoning trace is never scored.

    Args:
        output: Text generated by the model.

    Returns:
        The canonical calls, or None when nothing parsed as the schema.
    """
    output = output.rsplit(_THINK_END, 1)[-1]

    for candidate in _answer_candidates(output):
        parsed = _loads(candidate)

        if isinstance(parsed, dict):
            # The benchmark's own prompt asks for {"tool list": [...]}; accept
            # it so a model that answers in that shape is not scored as a
            # parse failure.
            for key in ("tool list", "tool_list", "calls", "tools"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break

        if isinstance(parsed, list):
            calls = canonical_calls(parsed)

            if calls:
                return calls

    return None


def score_calls(pred: list[dict] | None, gold: list[dict]) -> dict:
    """Every component of the metric, from one prediction and its gold calls.

    Args:
        pred: Canonical predicted calls, or None when the output did not parse.
        gold: Canonical gold calls of the item.

    Returns:
        The component metrics, all floats in [0,1].
    """
    empty = {
        "parse_success": 0.0,
        "tool_em": 0.0,
        "tool_inclusion": 0.0,
        "required_em": 0.0,
        "optional_em": 0.0,
        "usage_em": 0.0,
        "full_em": 0.0,
        "n_pred": 0,
        "n_gold": len(gold),
    }

    if pred is None:
        return empty

    gold_by_name = {call["tool name"]: call for call in gold}
    pred_by_name = {call["tool name"]: call for call in pred}
    shared = sorted(set(gold_by_name) & set(pred_by_name))
    tool_em = float(set(gold_by_name) == set(pred_by_name))
    inclusion = len(shared) / max(1, len(gold_by_name))
    required = [
        _params_match(pred_by_name[name], gold_by_name[name], "required parameters")
        for name in shared
    ]
    optional = [
        _params_match(pred_by_name[name], gold_by_name[name], "optional parameters")
        for name in shared
    ]
    required_em = _mean(required)

    return {
        "parse_success": 1.0,
        "tool_em": tool_em,
        "tool_inclusion": inclusion,
        "required_em": required_em,
        "optional_em": _mean(optional),
        # The reference metric: required and optional both matching, averaged
        # over the tools the prediction and the gold set share.
        "usage_em": _mean([a and b for a, b in zip(required, optional, strict=True)]),
        "full_em": float(tool_em == 1.0 and required_em == 1.0),
        "n_pred": len(pred_by_name),
        "n_gold": len(gold_by_name),
    }


def recoverable_flags(query: str, calls: list[dict]) -> dict:
    """Which gold parameter values the query actually contains.

    This is what the reachability filter and its manifest check are built
    on: a row ships only if every required gold value occurs in its query,
    so a wrong answer is a statement about the model rather than about gold
    the prompt never carried.

    Args:
        query: The row's query, verbatim.
        calls: Canonical gold calls of the row.

    Returns:
        Tool name to the names of its parameters whose value is in the query.
    """
    haystack = _normalize_text(query)
    out: dict = {}

    for call in calls:
        names: list[str] = []

        for key in ("required parameters", "optional parameters"):
            for parameter in call[key]:
                value = _normalize_text(str(parameter["value"]))

                if value and value in haystack:
                    names.append(str(parameter["name"]))

        out[call["tool name"]] = names

    return out


def normalize_value(value: object) -> object:
    """Fold one parameter value the way the benchmark's own metrics do.

    Mirrors utils/metrics.py normalize_value: literal evaluation, then
    whitespace, location, date, boolean and numeric normalization in that
    order. Reported numbers then mean what the paper means by them.

    Args:
        value: Raw parameter value from gold or from a model output.

    Returns:
        The normalized value, which may be a str, int, float or bool.
    """
    if value is None:
        return None

    text = str(value).strip()

    try:
        result: object = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        result = text

    if isinstance(result, str):
        result = re.sub(r"\s+", " ", re.sub(r"\s*,\s*", ",", result)).strip()
        result = _normalize_location(result)
        result = _normalize_date(result)
        result = _normalize_boolean(result)

    if isinstance(result, str):
        if result.isdigit():
            return int(result)
        try:
            return float(result)
        except ValueError:
            return result

    return result


def _repo_id(path: str | None) -> str:
    """benchmark.path names the Hub repo; the hf: prefix is optional, empty means the mirror.

    Args:
        path: Configured benchmark path, possibly None or hf:-prefixed.

    Returns:
        Hub repo id to load, defaulting to HF_REPO.
    """
    text = (path or "").strip()

    if text.startswith("hf:"):
        text = text[3:].strip()

    return text or HF_REPO


def _manifest() -> dict:
    """The shipped split manifest, which records how the digests were built.

    Args:
        None.

    Returns:
        The manifest mapping.
    """
    path = SPLIT_ROOT / "split_manifest.json"

    try:
        document: dict = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"traject id split is missing {path}") from exc

    return document


def _check_recoverable(item_id: str, entry: dict, calls: list[dict], usable: dict) -> None:
    """Refuse a row whose recoverable-parameter counts differ from the manifest.

    The flags are recomputed from the query rather than shipped per parameter,
    so the counts are what proves the recomputation still agrees. The split
    also drops every row with an unreachable required value, so equality of
    n_required and n_required_recoverable is checked as an invariant.

    Args:
        item_id: Row id being checked.
        entry: Manifest entry for that row.
        calls: Canonical gold calls of the row.
        usable: Tool name to the parameter names found in the query.
    """
    if "n_required" not in entry:
        return

    counts = {"n_required": 0, "n_required_recoverable": 0, "n_optional": 0,
              "n_optional_recoverable": 0}

    for call in calls:
        found = set(usable.get(call["tool name"], []))

        for kind in ("required", "optional"):
            for parameter in call[f"{kind} parameters"]:
                counts[f"n_{kind}"] += 1
                counts[f"n_{kind}_recoverable"] += int(parameter["name"] in found)

    expected = {key: entry[key] for key in counts if key in entry}
    actual = {key: counts[key] for key in expected}

    if actual != expected:
        raise ValueError(
            f"traject record {item_id!r} recomputes parameter counts {actual}, manifest says "
            f"{expected}. Either the mirror row or the recoverability rule has changed; "
            "rebuild the manifest deliberately."
        )

    if counts["n_required"] != counts["n_required_recoverable"]:
        raise ValueError(
            f"traject record {item_id!r} has {counts['n_required']} required parameters but "
            f"only {counts['n_required_recoverable']} of them occur in the query. The split "
            "is filtered to reachable rows, so this one cannot be answered from any prompt; "
            "rebuild the split."
        )


def _split_entries(split: str) -> list[dict]:
    """The entries of one manifest split, in file order and de-duplicated.

    Args:
        split: Split name, which is also its directory under SPLIT_ROOT.

    Returns:
        Manifest entries listed by that split's items.json.
    """
    path = SPLIT_ROOT / split / "items.json"

    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"traject id split is missing {path}") from exc

    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of items, got {type(raw).__name__}")

    out: list[dict] = []
    seen: set[str] = set()

    for entry in raw:
        item_id = str(entry.get("id", "")).strip() if isinstance(entry, dict) else ""

        if not item_id:
            raise ValueError(f"{path}: item without an `id` field")

        if item_id not in seen:
            seen.add(item_id)
            out.append(dict(entry))

    if not out:
        raise ValueError(f"{path}: no ids")

    return out


def _load(repo_id: str, config: str) -> list[dict]:
    """One config of the Hub dataset, at the only split it carries.

    Args:
        repo_id: Hub repo id to load.
        config: Config name to load.

    Returns:
        The rows of that config's test split, 200 of them.
    """
    return [dict(row) for row in load_dataset(repo_id, config)[HF_SPLIT]]


def _records_by_id(repo_id: str, wanted: set) -> dict:
    """The rows the manifest names, keyed by id and already parsed.

    Args:
        repo_id: Hub repo id to scan.
        wanted: Item ids the manifest asks for.

    Returns:
        Item id to its parsed row, for the ids that were found.
    """
    out: dict = {}

    for config in sorted(CONFIGS):
        for index, row in enumerate(_load(repo_id, config)):
            item_id = row_id(config, index)

            if item_id not in wanted or item_id in out:
                continue

            try:
                calls = json.loads(row["tool_list"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"traject record {item_id!r} has an unparseable tool_list ({exc}); the id "
                    "split names it, so it cannot be silently dropped"
                ) from exc

            out[item_id] = {
                "query": str(row["query"]),
                "task_name": str(row.get("task_name", "")),
                "calls": calls,
            }

    return out


def _parameters(call: dict, kind: str) -> list[dict]:
    """The name and value of every parameter of one kind, sorted by name.

    Args:
        call: One raw or predicted tool call.
        kind: Either required or optional.

    Returns:
        The parameters reduced to name and value, sorted by name.
    """
    raw = call.get(f"{kind} parameters")

    if raw is None:
        raw = call.get(f"{kind}_parameters")

    out: list[dict] = []

    for parameter in raw if isinstance(raw, list) else []:
        if not isinstance(parameter, dict) or "name" not in parameter:
            continue

        out.append({"name": str(parameter["name"]).strip(), "value": parameter.get("value")})

    return sorted(out, key=lambda parameter: parameter["name"])


def _answer_candidates(output: str) -> list[str]:
    """Every place an answer might be, most authoritative last-first.

    Args:
        output: Text generated by the model.

    Returns:
        Candidate JSON fragments, in the order they should be tried.
    """
    out: list[str] = []
    tagged = _ANSWER_TAG.findall(output)

    if tagged:
        out.append(str(tagged[-1]))

    fenced = _FENCE.findall(output)

    if fenced:
        out.append(str(fenced[-1]))

    out += _balanced(output)

    return out


def _balanced(output: str) -> list[str]:
    """The last balanced JSON array and object in the text, longest first.

    Args:
        output: Text generated by the model.

    Returns:
        Candidate fragments, empty when the text has no bracketed span.
    """
    out: list[str] = []

    for opener, closer in (("[", "]"), ("{", "}")):
        start = output.find(opener)
        end = output.rfind(closer)

        if 0 <= start < end:
            out.append(output[start : end + 1])

    return out


def _loads(text: str) -> object:
    """Parse one candidate fragment as JSON, then as a Python literal.

    Args:
        text: Candidate fragment from the model output.

    Returns:
        The parsed value, or None when neither parser accepted it.
    """
    stripped = text.strip()

    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except ValueError:
        pass

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _params_match(pred: dict, gold: dict, key: str) -> bool:
    """Whether two calls agree on one kind of parameter after normalization.

    Empty values are dropped before comparison, as the benchmark's own metric
    does, so a parameter supplied as the empty string is not a disagreement.

    Args:
        pred: One predicted call.
        gold: The gold call of the same tool.
        key: Which parameter list to compare.

    Returns:
        True when the two normalized name and value mappings are equal.
    """
    return _normalized_params(pred.get(key)) == _normalized_params(gold.get(key))


def _normalized_params(raw: object) -> list[tuple]:
    """Normalize a parameter list into sorted name and value pairs.

    Args:
        raw: A call's parameter list, possibly missing.

    Returns:
        Non-empty parameters as normalized pairs, sorted by name.
    """
    out: list[tuple] = []

    for parameter in raw if isinstance(raw, list) else []:
        if not isinstance(parameter, dict) or "name" not in parameter:
            continue

        value = parameter.get("value")

        if value is None or (isinstance(value, str) and not value.strip()):
            continue

        out.append((str(parameter["name"]).strip(), normalize_value(value)))

    return sorted(out, key=lambda pair: pair[0])


def _mean(values: list[bool]) -> float:
    """Average a list of outcomes, scoring an empty intersection as zero.

    Args:
        values: Per-tool outcomes over the shared tools.

    Returns:
        The mean, or 0.0 when there is nothing to average.
    """
    return float(sum(values) / len(values)) if values else 0.0


def _normalize_text(text: str) -> str:
    """Fold text for the recoverability test: casefold, drop punctuation, collapse.

    Args:
        text: Query text or one gold parameter value.

    Returns:
        The folded text a containment test runs on.
    """
    folded = str(text).strip().casefold()

    try:
        number = float(folded)
        folded = str(int(number)) if number == int(number) else str(number)
    except (ValueError, OverflowError):
        pass

    return re.sub(r"\s+", " ", _PUNCT.sub(" ", folded)).strip()


_COUNTRIES = {
    "FR": "France", "US": "United States", "USA": "United States", "UK": "United Kingdom",
    "GB": "United Kingdom", "DE": "Germany", "Deutschland": "Germany", "IT": "Italy",
    "Italia": "Italy", "ES": "Spain", "España": "Spain", "JP": "Japan", "CN": "China",
    "AU": "Australia", "CA": "Canada", "IN": "India", "BR": "Brazil", "MX": "Mexico",
    "RU": "Russia", "KR": "South Korea", "NL": "Netherlands", "SE": "Sweden", "NO": "Norway",
    "DK": "Denmark", "FI": "Finland", "CH": "Switzerland", "AT": "Austria", "BE": "Belgium",
    "PT": "Portugal", "GR": "Greece", "TR": "Turkey", "PL": "Poland", "CZ": "Czech Republic",
    "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia", "SI": "Slovenia",
    "SK": "Slovakia", "LT": "Lithuania", "LV": "Latvia", "EE": "Estonia",
}


def _normalize_location(text: str) -> str:
    """Expand country codes inside a comma separated location, as the reference does.

    Args:
        text: A candidate location string.

    Returns:
        The string with known codes and names expanded, unchanged otherwise.
    """
    if "," not in text:
        return text

    lowered = {full.casefold(): full for full in _COUNTRIES.values()}
    parts = []

    for part in (piece.strip() for piece in text.split(",")):
        if part.upper() in _COUNTRIES:
            parts.append(_COUNTRIES[part.upper()])
        else:
            parts.append(lowered.get(part.casefold(), part))

    return ",".join(parts)


def _normalize_date(text: str) -> str:
    """Reformat an ISO-looking date so 2023-1-5 and 2023-01-05 agree.

    Args:
        text: A candidate date string.

    Returns:
        The date as YYYY-MM-DD, unchanged when it does not parse.
    """
    if not re.match(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return text

    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return text


def _normalize_boolean(text: str) -> object:
    """Fold the strings that stand for true and false, as the reference does.

    Args:
        text: A candidate boolean string.

    Returns:
        The boolean it denotes, or the text unchanged.
    """
    lowered = text.casefold().strip()

    if lowered in ("true", "1", "yes", "on"):
        return True

    if lowered in ("false", "0", "no", "off"):
        return False

    return text
