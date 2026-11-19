"""Build the artifacts the TRAJECT-Bench adapter ships: the tool index and the id split.

The splits are generated once, here, and committed under
data/traject_id_split/. Nothing recomputes them at runtime: the adapter reads
the manifest and materializes the rows it names from the Hub.

Run it as a module:

    python -m skillpng.benchmarks.trajectbench.traject_build index
    python -m skillpng.benchmarks.trajectbench.traject_build split

The grouping unit is (domain, sorted set of gold parent tool names). Within
every domain the gold-tool co-occurrence graph is one connected component, so
no unit finer than the domain can keep an individual tool out of two splits;
this one blocks the next structure down, the provider combination.

Rows are filtered before they are split: a row whose gold requires a value the
query does not contain is unanswerable, because the schema default is stripped
from the index and so the value appears in the prompt nowhere. See
filter_reachable.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ..base import Item, split_by_group

log = logging.getLogger(__name__)

HF_REPO = "bigboss24/TRAJECT-Bench"
# The nine parallel-simple configs, mapped to the domain name their gold calls
# carry. The paper's tenth parallel domain, Mapping, has no Hub config.
CONFIGS = {
    "parallel_ecommerce_simple": "eCommerce",
    "parallel_education_simple": "Education",
    "parallel_email_simple": "Email",
    "parallel_finance_simple": "Finance",
    "parallel_gaming_simple": "Gaming",
    "parallel_music_simple": "Music",
    "parallel_news_media_simple": "News_Media",
    "parallel_travel_simple": "Travel",
    "parallel_weather_simple": "Weather",
}
HF_SPLIT = "test"
EXPECTED_ROWS = 1800
# Rows surviving the reachability filter, of EXPECTED_ROWS. Pinned for the
# same reason as EXPECTED_ROWS: a different count is a different corpus.
EXPECTED_REACHABLE = 875
SPLITS = ("train", "val", "test")
FRACTIONS = {"train": 0.2, "val": 0.1, "test": 0.7}
# Every grouping is a valid partition; the seed picks which one. Searched over
# SEED_CANDIDATES for the assignment closest to FRACTIONS, then pinned here so
# the shipped manifest is reproducible. Searched over the reachable rows, so
# it moved when the filter was introduced.
SPLIT_SEED = 109
SEED_CANDIDATES = 200

DATA_ROOT = Path(__file__).parent.parent / "data"
SPLIT_ROOT = DATA_ROOT / "traject_id_split"
QUARANTINE = DATA_ROOT / "traject_quarantine.json"
UNREACHABLE = DATA_ROOT / "traject_unreachable.json"
TOOL_INDEX = DATA_ROOT / "traject_tool_index.json"
# The merged endpoint schemas live here on the Hub. The GitHub clone ships
# an older copy missing 139 of the 752 gold endpoints, so it is never the
# source: it stays a parity fixture for the reference metrics.
TOOLS_FILE = "tools/all_tools.json"
# Fields the index keeps per endpoint. default and code are dropped at
# build time rather than at render time, so the two fields that carry the
# gold answer cannot reach a prompt even by mistake. See render_tool.
INDEX_FIELDS = (
    "tool name",
    "parent tool name",
    "parent tool description",
    "API name",
    "tool description",
    "domain name",
)


def row_id(config: str, index: int) -> str:
    """The stable id of one row: its config and its position in that config.

    Args:
        config: Hub config the row was loaded from.
        index: Position of the row inside that config's test split.

    Returns:
        Row id used as manifest key, per-instance seed and Item id.
    """
    return f"{config}:{index}"


def normalize_name(name: object) -> str:
    """Fold a tool name to the form the gold calls and the index are joined on.

    Names are inconsistently formatted across the two sources: two gold names
    carry a double space the index does not. Casefolding and entity decoding
    are not needed for the current corpus but cost nothing and turn a future
    formatting drift into a match rather than a hard failure.

    Args:
        name: Raw tool name from a gold call or from the index.

    Returns:
        The name with entities decoded, whitespace and the separator
        normalized, and casefolded.
    """
    text = re.sub(r"\s+", " ", html.unescape(str(name))).strip().casefold()
    # A handful of providers carry a trailing space, which would otherwise
    # leave the key as "zerobounce : validate" and stop it reconstructing
    # from the provider and endpoint it is built out of.
    return re.sub(r"\s*:\s*", ": ", text)


def index_digest(tools: list[dict]) -> str:
    """Fingerprint the index payload so a Hub-side change is an error, not a drift.

    Args:
        tools: Index entries in the order they are shipped.

    Returns:
        The first 16 hex chars of the sha256 of the canonical dump.
    """
    payload = json.dumps(tools, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_index(repo_id: str = HF_REPO) -> None:
    """Merge the Hub endpoint schemas into the shipped index and write it out.

    Args:
        repo_id: Hub repo id holding the tools file.
    """
    path = hf_hub_download(repo_id, TOOLS_FILE, repo_type="dataset")
    raw = json.loads(Path(path).read_text())
    revision = Path(path).parent.parent.name
    tools: list[dict] = []
    seen: set[str] = set()

    for entry in raw:
        key = normalize_name(entry.get("tool name"))

        if not key or key in seen:
            # 95 of 1281 names carry duplicate entries that disagree on
            # descriptions and, for four of them, on the parameter lists.
            # First occurrence in file order wins; the digest pins the result.
            continue

        expected = f"{entry.get('parent tool name')}: {entry.get('API name')}"

        if normalize_name(expected) != key:
            raise ValueError(
                f"index entry {entry.get('tool name')!r} is not "
                f"'{{parent tool name}}: {{API name}}'; the pool and the target both "
                "assume that identity holds"
            )
        seen.add(key)
        tools.append(
            {
                "key": key,
                **{field: _text(entry.get(field)) for field in INDEX_FIELDS},
                "required": _parameters(entry.get("required_parameters")),
                "optional": _parameters(entry.get("optional_parameters")),
                # Names only: connect params describe how one endpoint feeds
                # another, which is a hint at the rest of the gold set.
                "connected": sorted(
                    {
                        normalize_name(item.get("tool name"))
                        for item in (entry.get("connected tools") or [])
                        if item.get("tool name")
                    }
                ),
                "output_info": entry.get("output_info") or {},
            }
        )

    tools.sort(key=lambda tool: tool["key"])
    domains = Counter(str(tool["domain name"]) for tool in tools)
    document = {
        "benchmark": "TRAJECT-Bench",
        "manifest_type": "tool_index",
        "source_repo": repo_id,
        "source_repo_type": "dataset",
        "source_url": f"https://huggingface.co/datasets/{repo_id}",
        "source_file": TOOLS_FILE,
        "source_revision": revision,
        "n_tools": len(tools),
        "domains": dict(sorted(domains.items())),
        "digest": index_digest(tools),
        "notes": [
            "Endpoint schemas merged from the Hub tools file, keyed on the ",
            "casefolded, whitespace-collapsed tool name, which is always ",
            "'{parent tool name}: {API name}'. Duplicate names keep their ",
            "first occurrence in file order. Parameter default values and the ",
            "code field are dropped here rather than at render time: both ",
            "carry gold argument values verbatim, so removing them from the ",
            "shipped artifact makes the leak unreachable. connect params are ",
            "dropped and only connected tool names are kept, for distractor ",
            "sampling; they are never rendered into a prompt.",
        ],
        "tools": tools,
    }
    TOOL_INDEX.parent.mkdir(parents=True, exist_ok=True)
    TOOL_INDEX.write_text(json.dumps(document, indent=1) + "\n")
    log.info(
        "traject index: %d endpoints, digest %s, revision %s",
        len(tools),
        document["digest"],
        revision,
    )


def _text(value: object) -> str:
    """Decode entities and collapse whitespace in one shipped text field.

    The Hub file carries HTML entities inside descriptions, so a raw render
    shows filter&#95;string where the model needs filter_string.

    Args:
        value: Raw text field of an endpoint schema.

    Returns:
        The decoded text with whitespace collapsed, empty when missing.
    """
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def _parameters(raw: object) -> list[dict]:
    """Keep only what a prompt may show about a parameter: name, type, description.

    Args:
        raw: Parameter list of one endpoint schema, possibly missing.

    Returns:
        The parameters with their default values removed.
    """
    out: list[dict] = []

    for parameter in raw if isinstance(raw, list) else []:
        out.append(
            {
                "name": _text(parameter.get("name")),
                "type": _text(parameter.get("type")),
                "description": _text(parameter.get("description")),
            }
        )

    return out


def group_key(domain: str, calls: list[dict]) -> str:
    """The split unit: the domain and the sorted set of gold parent tool names.

    Args:
        domain: Domain name the row's gold calls belong to.
        calls: Parsed gold tool calls of the row.

    Returns:
        Group key, stable under any reordering of the calls.
    """
    parents = sorted({str(call["parent tool name"]) for call in calls})

    return domain + "::" + "|".join(parents)


def normalize_value(value: object) -> str:
    """Fold a gold parameter value to the form value novelty is judged on.

    Args:
        value: Raw parameter value of a gold call.

    Returns:
        Casefolded value with whitespace collapsed, empty when it is blank.
    """
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def call_values(call: dict) -> list[str]:
    """Every non-blank parameter value of one gold call, required and optional.

    Args:
        call: One parsed gold tool call.

    Returns:
        Normalized values, in the order the call lists them.
    """
    out: list[str] = []

    for key in ("required parameters", "optional parameters"):
        for parameter in call.get(key) or []:
            value = normalize_value(parameter.get("value"))

            if value:
                out.append(value)

    return out


def load_records(repo_id: str = HF_REPO) -> tuple[list[dict], list[dict]]:
    """Every parallel-simple row, parsed and checked, plus the ones that failed.

    Args:
        repo_id: Hub repo id holding the nine configs.

    Returns:
        The usable records in canonical order, and the quarantined rows.
    """
    records: list[dict] = []
    quarantined: list[dict] = []

    for config in sorted(CONFIGS):
        domain = CONFIGS[config]
        rows = load_dataset(repo_id, config)[HF_SPLIT]
        log.info("traject: %s -> %d rows", config, len(rows))

        for index, row in enumerate(rows):
            item_id = row_id(config, index)

            try:
                calls = json.loads(row["tool_list"])
            except (TypeError, ValueError) as exc:
                quarantined.append({"id": item_id, "reason": f"tool_list parse: {exc}"})
                continue

            if not isinstance(calls, list) or not calls:
                quarantined.append({"id": item_id, "reason": "tool_list is not a non-empty list"})
                continue

            if len(calls) != int(row["tool_count"]):
                quarantined.append(
                    {
                        "id": item_id,
                        "reason": f"tool_count {row['tool_count']} != {len(calls)} parsed calls",
                    }
                )
                continue

            domains = {str(call.get("domain name")) for call in calls}

            if domains != {domain}:
                quarantined.append(
                    {"id": item_id, "reason": f"gold domains {sorted(domains)} != [{domain}]"}
                )
                continue

            records.append(
                {
                    "id": item_id,
                    "config": config,
                    "index": index,
                    "domain": domain,
                    "query": str(row["query"]),
                    "task_name": str(row["task_name"]),
                    "tool_count": len(calls),
                    # final_answer and executed_output are dropped: nothing is
                    # executed here, and neither is part of the target.
                    "calls": [
                        {
                            key: call[key]
                            for key in (
                                "tool name",
                                "required parameters",
                                "optional parameters",
                                "parent tool name",
                                "API name",
                                "domain name",
                            )
                            if key in call
                        }
                        for call in calls
                    ],
                }
            )

    return records, quarantined


def unreachable_required(record: dict) -> list[str]:
    """The required gold parameters of one row whose value the query omits.

    Args:
        record: One usable record as load_records returns it.

    Returns:
        Tool name and parameter name, joined by a dot, one per omission.
    """
    # Imported here for the same reason write_split does: the adapter imports
    # this module for the corpus definition, so the dependency runs one way.
    from .traject import canonical_calls, recoverable_flags

    calls = canonical_calls(record["calls"])
    usable = recoverable_flags(record["query"], calls)
    missing: list[str] = []

    for call in calls:
        found = set(usable.get(call["tool name"], []))

        for parameter in call["required parameters"]:
            if parameter["name"] not in found:
                missing.append(f"{call['tool name']}.{parameter['name']}")

    return missing


def filter_reachable(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop the rows asking for a required value no prompt can supply.

    The schema default is stripped from the index, so a required value absent
    from the query is in the prompt nowhere; the row scores zero on full_em
    for a correct model. Dropping them makes the headline a measurement of
    ability rather than of how much gold the corpus withheld. Optional
    parameters are not filtered on: they never enter full_em, and requiring
    them would cost most of the corpus.

    Args:
        records: Usable records in canonical order.

    Returns:
        The reachable records, and one entry per dropped row naming the cause.
    """
    reachable: list[dict] = []
    dropped: list[dict] = []

    for record in records:
        missing = unreachable_required(record)

        if missing:
            dropped.append(
                {
                    "id": record["id"],
                    "domain": record["domain"],
                    "unreachable_required": missing,
                }
            )
        else:
            reachable.append(record)

    return reachable, dropped


def assign_splits(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """Partition the records by group, using the shared splitter unchanged.

    Placeholder Items exist only to reach split_by_group, which is the one
    definition of how a group key becomes a split.

    Args:
        records: Usable records in canonical order.
        seed: Seed selecting which of the valid partitions is taken.

    Returns:
        Split name to the records assigned to it, in input order.
    """
    by_id = {record["id"]: record for record in records}
    items = [
        Item(id=record["id"], target="", group=group_key(record["domain"], record["calls"]))
        for record in records
    ]
    split_items = split_by_group(items, FRACTIONS, seed=seed)

    return {name: [by_id[item.id] for item in split] for name, split in split_items.items()}


def choose_seed(records: list[dict], candidates: int = SEED_CANDIDATES) -> tuple[int, float]:
    """The seed whose partition lands closest to the requested fractions.

    Args:
        records: Usable records in canonical order.
        candidates: Number of seeds to search, starting from zero.

    Returns:
        The best seed and its total absolute deviation from FRACTIONS.
    """
    best = (0, float("inf"))

    for seed in range(candidates):
        splits = assign_splits(records, seed)
        deviation = sum(
            abs(len(splits[name]) / len(records) - fraction)
            for name, fraction in FRACTIONS.items()
        )

        if deviation < best[1]:
            best = (seed, deviation)

    return best


def unseen_value_flags(splits: dict[str, list[dict]]) -> dict[str, bool]:
    """Mark the items none of whose gold parameter values occur anywhere in train.

    Literal identifiers recur heavily across rows, so this separates the items
    that test genuine extraction from the ones a memorized value would carry.

    Args:
        splits: Split name to its records.

    Returns:
        Item id to whether all of its gold values are unseen in train.
    """
    seen = {
        value
        for record in splits["train"]
        for call in record["calls"]
        for value in call_values(call)
    }
    flags: dict[str, bool] = {}

    for split in splits.values():
        for record in split:
            values = [value for call in record["calls"] for value in call_values(call)]
            # A row with no values at all has nothing novel to test on.
            flags[record["id"]] = bool(values) and not any(value in seen for value in values)

    return flags


def write_split(
    splits: dict[str, list[dict]], flags: dict[str, bool], seed: int, dropped: int
) -> None:
    """Write the manifest and the per-split item files under SPLIT_ROOT.

    Args:
        splits: Split name to its records.
        flags: Item id to its has_unseen_values flag.
        seed: Seed the partition was generated with.
        dropped: Rows the reachability filter removed before splitting.
    """
    # Imported here, not at module scope: the adapter imports this module for
    # the corpus definition, so the dependency only runs in this direction at
    # build time.
    from .traject import (
        N_DISTRACTORS,
        candidate_pool,
        canonical_calls,
        load_index,
        pool_digest,
        recoverable_flags,
    )

    index = load_index()

    for name in SPLITS:
        entries = []

        for record in splits[name]:
            calls = canonical_calls(record["calls"])
            gold_keys = [normalize_name(call["tool name"]) for call in calls]
            unknown = sorted(key for key in gold_keys if key not in index["by_key"])

            if unknown:
                raise ValueError(
                    f"{record['id']}: gold tool(s) {unknown} are not in the shipped tool "
                    "index; rebuild the index before the split"
                )

            pool = candidate_pool(
                record["id"], record["domain"], gold_keys, index, N_DISTRACTORS, 0
            )
            usable = recoverable_flags(record["query"], calls)
            counts = _recoverable_counts(calls, usable)
            entries.append(
                {
                    "id": record["id"],
                    "domain": record["domain"],
                    "group": group_key(record["domain"], record["calls"]),
                    "tool_count": record["tool_count"],
                    "has_unseen_values": flags[record["id"]],
                    "pool_digest": pool_digest(pool),
                    **counts,
                }
            )
        path = SPLIT_ROOT / name / "items.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2) + "\n")

    groups = {group_key(r["domain"], r["calls"]) for split in splits.values() for r in split}
    manifest = {
        "benchmark": "TRAJECT-Bench",
        "manifest_type": "id_split",
        "source_repo": HF_REPO,
        "source_repo_type": "dataset",
        "source_url": f"https://huggingface.co/datasets/{HF_REPO}",
        "source_id_field": "config:row_index",
        "source_configs": sorted(CONFIGS),
        "source_split_name": HF_SPLIT,
        "split_seed": seed,
        "source_rows": EXPECTED_ROWS,
        "filter": "every required gold value occurs in the query under normalization",
        "n_dropped_unreachable": dropped,
        "grouping": "(domain, sorted set of gold parent tool names)",
        "n_groups": len(groups),
        "counts": {name: len(splits[name]) for name in SPLITS},
        "pool_defaults": {"n_distractors": 10, "hard_negatives": 0},
        "item_fields": [
            "id",
            "domain",
            "group",
            "tool_count",
            "has_unseen_values",
            "pool_digest",
            "n_required",
            "n_required_recoverable",
            "n_optional",
            "n_optional_recoverable",
        ],
        "notes": [
            "This is a split manifest, not the full TRAJECT-Bench payload. ",
            "Materialize the rows from the nine parallel *_simple configs of ",
            f"{HF_REPO} before evaluation. An id is config:row_index over the ",
            "test split of that config, which is the only split each has. ",
            "Rows are grouped on (domain, sorted set of gold parent tool ",
            "names) and whole groups are assigned to splits, so no provider ",
            "combination straddles a boundary. Within each domain the gold ",
            "tool co-occurrence graph is a single connected component, so no ",
            "unit finer than the domain can keep an individual tool out of ",
            "two splits; the domain itself has only nine values and would ",
            "leave whole domains unseen, so this is the finest unit that ",
            "still blocks a shared structure. Every partition is valid and ",
            f"the seed picks one: seed {seed} was chosen from the first ",
            f"{SEED_CANDIDATES} as the assignment closest to 20/10/70. ",
            "has_unseen_values is true when none of the row's gold parameter ",
            "values, casefolded and whitespace-collapsed, occurs anywhere in ",
            "train; it is a reporting subset, not a filter. pool_digest is the ",
            "sha256 of the ordered candidate pool built under pool_defaults, ",
            "and the adapter refuses a mismatch under those settings so a ",
            "drifting sampler is an error rather than a quiet change of ",
            "experiment. n_*_recoverable counts the gold parameters whose ",
            "value occurs in the query under normalization. Rows with any ",
            "required value the query does not supply are dropped before the ",
            "split: the schema default is stripped from the index, so such a ",
            "value is in the prompt nowhere and the row is unanswerable. ",
            f"{dropped} of {EXPECTED_ROWS} rows go, listed by id and cause in ",
            "traject_unreachable.json, and n_required == ",
            "n_required_recoverable holds for every shipped item. Optional ",
            "parameters are not filtered on: they never enter full_em, and ",
            "n_optional_recoverable is only a bookkeeping count.",
        ],
    }
    (SPLIT_ROOT / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _recoverable_counts(calls: list[dict], usable: dict) -> dict:
    """Count the gold parameters, and how many of them the query supplies.

    Args:
        calls: Canonical gold calls of one row.
        usable: Tool name to the names of its parameters found in the query.

    Returns:
        The required and optional totals and their recoverable counts.
    """
    counts = {
        "n_required": 0,
        "n_required_recoverable": 0,
        "n_optional": 0,
        "n_optional_recoverable": 0,
    }

    for call in calls:
        found = set(usable.get(call["tool name"], []))

        for kind in ("required", "optional"):
            for parameter in call[f"{kind} parameters"]:
                counts[f"n_{kind}"] += 1
                counts[f"n_{kind}_recoverable"] += int(parameter["name"] in found)

    return counts


def build_split(repo_id: str = HF_REPO, seed: int | None = None, search: bool = False) -> None:
    """Load the corpus, partition it, and write the shipped split files.

    Args:
        repo_id: Hub repo id holding the nine configs.
        seed: Seed to use, or None for the pinned SPLIT_SEED.
        search: Whether to search for the seed closest to the fractions first.
    """
    records, quarantined = load_records(repo_id)

    if quarantined:
        QUARANTINE.parent.mkdir(parents=True, exist_ok=True)
        QUARANTINE.write_text(json.dumps(quarantined, indent=2) + "\n")
        raise ValueError(
            f"{len(quarantined)} row(s) could not be parsed or failed a consistency check; "
            f"they are listed in {QUARANTINE}. The corpus is meant to be complete, so this "
            "is a broken mirror rather than a set of rows to drop."
        )

    if len(records) != EXPECTED_ROWS:
        raise ValueError(
            f"{repo_id!r} yielded {len(records)} rows over {len(CONFIGS)} configs, expected "
            f"{EXPECTED_ROWS}. A different row count is a different corpus."
        )

    records, dropped = filter_reachable(records)
    UNREACHABLE.parent.mkdir(parents=True, exist_ok=True)
    UNREACHABLE.write_text(json.dumps(dropped, indent=2) + "\n")
    log.info(
        "traject: %d of %d rows reachable, %d dropped (listed in %s)",
        len(records),
        EXPECTED_ROWS,
        len(dropped),
        UNREACHABLE,
    )

    if len(records) != EXPECTED_REACHABLE:
        raise ValueError(
            f"the reachability filter kept {len(records)} of {EXPECTED_ROWS} rows, expected "
            f"{EXPECTED_REACHABLE}. Either the mirror or the recoverability rule has changed; "
            "repin EXPECTED_REACHABLE and the seed deliberately."
        )

    if search:
        seed, deviation = choose_seed(records)
        log.info("traject: best seed %d, deviation %.4f", seed, deviation)

    chosen = SPLIT_SEED if seed is None else seed
    splits = assign_splits(records, chosen)
    flags = unseen_value_flags(splits)
    write_split(splits, flags, chosen, len(dropped))

    for name in SPLITS:
        domains = Counter(record["domain"] for record in splits[name])
        unseen = sum(flags[record["id"]] for record in splits[name])
        log.info(
            "traject %s: %d items (%.1f%%), %d with unseen values, domains %s",
            name,
            len(splits[name]),
            100 * len(splits[name]) / len(records),
            unseen,
            dict(sorted(domains.items())),
        )


def main() -> None:
    """Command line entry point for building the shipped artifacts.

    Args:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", choices=["index", "split"])
    parser.add_argument("--repo-id", default=HF_REPO)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--search-seed", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.artifact == "index":
        build_index(args.repo_id)
    else:
        build_split(args.repo_id, args.seed, args.search_seed)


if __name__ == "__main__":
    main()
