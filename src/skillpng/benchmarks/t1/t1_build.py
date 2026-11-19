"""Build the artifacts the T1 adapter ships: the zero-shot prompt and the id split.

Both are generated once, here, and committed under data/. Nothing recomputes
them at runtime: the adapter verifies the prompt's digest and materializes the
items the manifest names from the pinned dataset revision.

Run it as a module:

    python -m skillpng.benchmarks.t1.t1_build prompt
    python -m skillpng.benchmarks.t1.t1_build split
    python -m skillpng.benchmarks.t1.t1_build audit

prompt and split import the vendored tools into this process, and with them
the 1.35 GB ontology; outside the executor's worker nothing else does.

split replays every gold plan of the corpus twice: once per dialogue
("session", the semantics every item is built and scored under) and once per
file in upstream's order (Python <= 3.12, i.e. the paper). A user turn ships
only if it has a gold plan, no gold plan of its dialogue raises under session
replay up to and including it, and the two replays agree on every row up to
and including it. On every shipped item our gold is therefore the paper's.

Every count and digest the task statement gives is asserted, and a mismatch
stops the build: those numbers are the specification of this split, and a
different one is a different corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import pickle
import resource
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..base import _unit_hash
from .t1_exec import ONTOLOGY, SEMANTICS, VENDORED, Row, Tools, replay_file, split_turn

log = logging.getLogger(__name__)

HF_REPO = "capitalone/T1"
HF_REVISION = "7a3e58355959426e3daeecc410c2b87139295140"
UPSTREAM_URL = "https://github.com/CapitalOne-Research/T1"
UPSTREAM_COMMIT = "22a16b7570a5c20b1dc05f22f78a82ab94488a00"
DOMAINS = (
    "flight",
    "hotel",
    "restaurant",
    "attraction",
    "flighthotel",
    "hotelrestaurant",
    "hotelattraction",
    "flighthotelrestaurant",
    "flighthotelattraction",
)
SPLITS = ("train", "val", "test")
# This repo's split name to T1's.
HF_SPLITS = {"train": "train", "val": "validation", "test": "test"}
N_FILES = 25
COLUMNS = ["Unnamed: 0", "ID", "Template", "Plan", "Filled_Template", "Filled_Plan"]
SEED = 42

PACKAGE_ROOT = Path(__file__).parent
UPSTREAM_ROOT = PACKAGE_ROOT / "upstream"
DATA_ROOT = PACKAGE_ROOT.parent / "data"
SPLIT_ROOT = DATA_ROOT / "t1_id_split"
PROMPT_PATH = DATA_ROOT / "t1_prompt.txt"
CONFIG_PATH = PACKAGE_ROOT.parents[3] / "configs" / "t1.yaml"

# The system turn: upstream's system text without its surrounding newlines
# and trailing space. configs/t1.yaml sends exactly this.
SYSTEM_PROMPT = (
    "You are an expert AI travel planner and your responsibility is to generate Python code using APIs or Tools."
)
CONVERSATION_SLOT = "<<T1_CONVERSATION>>"
CACHE_SLOT = "<<T1_CACHE>>"
# The zero-shot OUTPUT FORMAT body, replacing upstream's reasoning request.
OUTPUT_FORMAT = (
    "- You need to generate only the python code. The python code should be within the <CODE> </CODE> tags. Note while generating the python code, never have any markdown tags.\n"
    "- Every tool in TOOL CONFIG is already defined where the code runs. Call the tools directly: never import them, and never redefine them or their input models.\n"
    "- Except for seek_information, call every tool with keyword arguments only, for example "
    'save_to_cache(key="hotels", value=hotels).\n'
    "- Do not reference variables from earlier turns; read earlier results "
    "with get_results_from_cache."
)
OUTPUT_FORMAT_START = "- You need to generate the reasoning"
OUTPUT_FORMAT_END = "it should only have executable code."
# Upstream's save rule passes the key positionally, which the keyword-only
# save_to_cache rejects, and saves only what "could be reused". Gold saves
# every search, filter, sort and nearest result.
SAVE_RULE_UPSTREAM = (
    "- If you generate a tool call and its result could be reused later, save it with "
    'save_to_cache("<key>",value).'
)
SAVE_RULE = (
    "- Save the result of every search, filter, sort_results and search_nearest call with "
    'save_to_cache(key="<key>", value=<result>).'
)
# Upstream documents the nearest tool twice, once as find_nearest, which T1
# never defines. Both entries are named search_nearest here.
NEAREST_UPSTREAM = "Tool Name:\n\n    find_nearest\n"
NEAREST = "Tool Name:\n\n    search_nearest\n"

# Tools whose gold call makes an item a tools item, as opposed to one that
# only asks (seek_information) or only reads the cache.
T1_TOOLS = frozenset(
    {
        "search_flights",
        "filter_flights",
        "search_hotels",
        "filter_hotels",
        "search_restaurants",
        "filter_restaurants",
        "search_attractions",
        "filter_attractions",
        "search_nearest",
        "find_nearest",
        "sort_results",
        "adjust_date",
    }
)
GOLD_KINDS = ("tools", "seek_only", "seek_plus_tools", "cache_read_only")

# ---------------------------------------------------------------- the facts
# Verified on 2026-09-26/27 against the pinned code and data. Asserted, never
# worked around.

EXPECTED_FILES = 686
TOOL_CONFIG_CHARS = 46568
TOOL_CONFIG_DIGEST = "fab4c9a133aca07b"
IDS_PER_FILE = {"train": 15, "validation": 5, "test": 40}
USER_TURNS = {"train": 15000, "validation": 5175, "test": 40975}
NO_PLAN_USER_TURNS = {"train": 725, "validation": 175, "test": 1375}
EMPTY_ROWS = 4250
ENDS_WITH_ASSISTANT = 25
SAME_ROLE_PAIRS = 225
ASSISTANT_PLANS = {"attraction:test": 25, "hotelattraction:test": 25}
# Share of user turns per gold kind over the whole corpus, in percent.
GOLD_SHARES = {
    "tools": 69.0,
    "seek_only": 20.3,
    "seek_plus_tools": 4.8,
    "cache_read_only": 2.2,
    "none": 3.7,
}
WHITESPACE_PLAN = ("flight", "train", "47", 2)
GOLD_FAILURES = {
    "fresh": {"train": 164, "validation": 0, "test": 800},
    "session": {"train": 64, "validation": 0, "test": 375},
    "file": {"train": 64, "validation": 0, "test": 0},
}
FAILURES_BY_DOMAIN = {
    "session:test": {
        "flighthotel": 225,
        "flight": 75,
        "hotel": 25,
        "hotelrestaurant": 25,
        "flighthotelrestaurant": 25,
    },
    "session:train": {"flight": 46, "flighthotelrestaurant": 18},
    "file:train": {"flight": 46, "flighthotelrestaurant": 18},
}
DISAGREEMENTS = {"error": 375, "after": 500, "before": 325}
ELIGIBLE = {
    "flight": (1354, 475, 3775),
    "hotel": (1475, 550, 3950),
    "restaurant": (1200, 425, 3475),
    "attraction": (925, 375, 2700),
    "flighthotel": (1850, 800, 5600),
    "hotelrestaurant": (2050, 675, 5400),
    "hotelattraction": (2100, 675, 5375),
    "flighthotelrestaurant": (1553, 475, 4200),
    "flighthotelattraction": (1650, 550, 4625),
}
EXCLUDED = {"rule1": (725, 175, 1375), "rule2": (118, 0, 500), "rule3": (0, 0, 0)}
DIALOGUES_WITH_ELIGIBLE = (3375, 1125, 8975)
DROPPED_TEMPLATES = ["flighthotelrestaurant:test:26"]
SHIPPED = {
    "flight": (44, 22, 156),
    "hotel": (45, 23, 156),
    "restaurant": (45, 22, 156),
    "attraction": (45, 22, 156),
    "flighthotel": (44, 22, 156),
    "hotelrestaurant": (44, 22, 155),
    "hotelattraction": (45, 22, 155),
    "flighthotelrestaurant": (44, 22, 155),
    "flighthotelattraction": (44, 23, 155),
}
SHIPPED_DIGESTS = {
    "train": "36801cec4695bd2b",
    "val": "95837288aa261ea2",
    "test": "5a35dbd6e99156bf",
}
SHIPPED_KINDS = {
    "tools": (283, 152, 989),
    "seek_only": (85, 38, 289),
    "seek_plus_tools": (18, 4, 66),
    "cache_read_only": (14, 6, 56),
}
WHITESPACE_ITEMS = ["flight:train:1:47:2", "flight:train:15:47:2", "flight:train:18:47:2"]
ITEM_FIELDS = [
    "id",
    "domain",
    "hf_split",
    "file_index",
    "template_id",
    "turn",
    "gold_kind",
    "plan_digest",
    "cache_digest",
]


# ---------------------------------------------------------------- layout


@dataclass
class Dialogue:
    """One dialogue of a CSV: every row of one template ID, in file order."""

    id: str
    # [Filled_Template, Filled_Plan] per row, None for an empty cell.
    rows: list[Row]
    # The delexicalized Template column, never shown to a model.
    templates: list[str | None]


def dialogue_file(domain: str, hf_split: str, n: int) -> str:
    """Path of one dialogue CSV inside the dataset repo.

    Args:
        domain: One of DOMAINS.
        hf_split: train, validation or test.
        n: File index, 1 to 25.

    Returns:
        The repo-relative path.
    """
    return f"{domain}/{hf_split}/output_file_{domain}_{hf_split}_{n}.csv"


def item_id(domain: str, hf_split: str, n: int, template_id: str, turn: int) -> str:
    """The id of one user turn: domain, T1 split, file, template ID and turn.

    Args:
        domain: One of DOMAINS.
        hf_split: train, validation or test.
        n: File index, 1 to 25.
        template_id: The CSV's ID column, a template id.
        turn: 0-based ordinal of the user turn among the dialogue's non-empty
            user rows.

    Returns:
        The item id, e.g. flighthotel:test:17:23:2.
    """
    return f"{domain}:{hf_split}:{n}:{template_id}:{turn}"


def parse_item_id(value: str) -> dict[str, object]:
    """Split an item id back into its keys.

    Args:
        value: An id made by item_id.

    Returns:
        domain, hf_split, file_index, template_id and turn.
    """
    domain, hf_split, n, template_id, turn = value.split(":")

    return {
        "domain": domain,
        "hf_split": hf_split,
        "file_index": int(n),
        "template_id": template_id,
        "turn": int(turn),
    }


def digest(text: str) -> str:
    """First 16 hex characters of the sha256 of text.

    Args:
        text: Text to fingerprint.

    Returns:
        The short digest the manifest records.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ids_digest(ids: list[str]) -> str:
    """The digest of one split: its ids sorted and joined by newlines.

    Args:
        ids: Item ids of the split.

    Returns:
        The short digest.
    """
    return digest("\n".join(sorted(ids)))


def upstream_digest(root: Path = UPSTREAM_ROOT) -> str:
    """Fingerprint the vendored code, so the item cache follows any edit to it.

    Args:
        root: The vendored upstream directory.

    Returns:
        The short digest over every file's relative path and bytes.
    """
    hasher = hashlib.sha256()

    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            hasher.update(path.relative_to(root).as_posix().encode() + b"\0")
            hasher.update(path.read_bytes() + b"\0")

    return hasher.hexdigest()[:16]


def gold_kind(plan: str) -> str:
    """Classify a gold plan by the names upstream's extract_tool_calls finds.

    Args:
        plan: A gold Filled_Plan.

    Returns:
        tools, seek_only, seek_plus_tools or cache_read_only.
    """
    from .upstream.evaluation.eval_metrics import extract_tool_calls

    names = {name for call in extract_tool_calls(plan) for name in call}
    seek = "seek_information" in names
    tools = bool(names & T1_TOOLS)

    if seek:
        return "seek_plus_tools" if tools else "seek_only"

    return "tools" if tools else "cache_read_only"


# ---------------------------------------------------------------- data


def snapshot(files: list[str], repo_id: str = HF_REPO, attempts: int = 8) -> Path:
    """The local snapshot of the pinned revision, holding at least files.

    Files already in the Hub cache are not fetched again. Anything missing is
    downloaded two at a time: the Hub's xet token endpoint answers a default
    concurrency full download with HTTP 429. HF_TOKEN is honoured.

    Args:
        files: Repo-relative paths that must be present.
        repo_id: Hub dataset repo.
        attempts: Download attempts before giving up.

    Returns:
        The snapshot directory.
    """
    from huggingface_hub import snapshot_download

    try:
        # allow_patterns here too: without it huggingface_hub refuses a
        # snapshot missing any repo file, README.md included.
        local = Path(
            snapshot_download(
                repo_id,
                repo_type="dataset",
                revision=HF_REVISION,
                allow_patterns=files,
                local_files_only=True,
            )
        )

        if all((local / name).is_file() for name in files):
            return local
    except Exception:
        pass

    for attempt in range(1, attempts + 1):
        try:
            return Path(
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    allow_patterns=files,
                    max_workers=2,
                )
            )
        except Exception as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"could not download {len(files)} file(s) of {repo_id} at {HF_REVISION} "
                    f"after {attempts} attempts ({error}). Set HF_TOKEN for a higher rate "
                    "limit, or retry later."
                ) from error

            wait = min(120.0, 5.0 * 2**attempt)
            log.warning("t1 download failed (%s); retrying in %.0fs", error, wait)
            time.sleep(wait)

    raise AssertionError("unreachable")


def repo_files(repo_id: str = HF_REPO) -> list[str]:
    """Every file of the pinned revision, as the Hub lists it.

    Args:
        repo_id: Hub dataset repo.

    Returns:
        Repo-relative paths.
    """
    from huggingface_hub import HfApi

    return list(HfApi().list_repo_files(repo_id, repo_type="dataset", revision=HF_REVISION))


def ontology_root(repo_id: str = HF_REPO) -> Path:
    """The snapshot directory with the whole ontology present.

    Args:
        repo_id: Hub dataset repo.

    Returns:
        The directory the ontology variables are resolved against.
    """
    return snapshot(sorted(set(ONTOLOGY.values())), repo_id)


def read_dialogues(path: Path) -> list[Dialogue]:
    """Every dialogue of one CSV, read exactly as upstream reads it.

    Args:
        path: Local path of one dialogue CSV.

    Returns:
        One dialogue per ID, in upstream's order: IDs sorted as strings.
    """
    import pandas as pd

    frame = pd.read_csv(path, sep=",", quoting=csv.QUOTE_MINIMAL, dtype=str)

    if list(frame.columns) != COLUMNS:
        raise ValueError(f"{path}: columns {list(frame.columns)}, expected {COLUMNS}")

    def cell(value: object) -> str | None:
        """An empty CSV cell as None, anything else as the string it is.

        Args:
            value: One cell as pandas read it.

        Returns:
            None for a missing value, else the string.
        """
        return None if pd.isna(value) else str(value)

    dialogues: dict[str, Dialogue] = {}
    last = None

    for template_id, template, filled, plan in zip(
        frame["ID"], frame["Template"], frame["Filled_Template"], frame["Filled_Plan"], strict=True
    ):
        key = cell(template_id)

        if key is None:
            raise ValueError(f"{path}: a row has no ID")

        if key != last and key in dialogues:
            raise ValueError(f"{path}: the rows of ID {key} are not contiguous")

        last = key
        dialogue = dialogues.setdefault(key, Dialogue(key, [], []))
        dialogue.rows.append([cell(filled), cell(plan)])
        dialogue.templates.append(cell(template))

    return [dialogues[key] for key in sorted(dialogues)]


def user_turns(rows: list[Row]) -> list[int]:
    """Row index of every non-empty user row, in turn order.

    Args:
        rows: A dialogue's rows.

    Returns:
        Row indices; position in the list is the turn ordinal.
    """
    return [
        index
        for index, (template, _) in enumerate(rows)
        if template is not None and split_turn(template)[0] == "user"
    ]


# ---------------------------------------------------------------- prompt


def build_prompt() -> str:
    """Render upstream's prompt with sentinels and cut the zero-shot template out.

    Args:
        None.

    Returns:
        The user-turn template, with CONVERSATION_SLOT and CACHE_SLOT once each.
    """
    # planner_code calls logging.basicConfig at import; configured first, that
    # call is a no-op.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    Tools(str(ontology_root()))
    import importlib

    planner = importlib.import_module(VENDORED + ".planner.planner_code")
    examples = importlib.import_module(VENDORED + ".planner.planner_utils").few_shot_examples_2
    config = planner.all_tools_config
    _check(len(config) == TOOL_CONFIG_CHARS, f"TOOL CONFIG is {len(config)} chars")
    _check(digest(config) == TOOL_CONFIG_DIGEST, f"TOOL CONFIG digest is {digest(config)}")
    rendered = planner.prompt_reasoning_final(CONVERSATION_SLOT, CACHE_SLOT)
    head = "\n" + SYSTEM_PROMPT + " \n" + " " * 8 + "\n" + "\n"
    _check(
        rendered.startswith(head) and rendered.endswith("\n"),
        "the rendered prompt is not system text, then the user text, as expected",
    )
    user = rendered[len(head) : -1]
    block = "# EXAMPLES\n" + examples + "\n\n"
    _check(user.count(block) == 1, f"the few-shot block occurs {user.count(block)} times")
    user = user.replace(block, "")
    _check(
        "executable code.\n\n# CONVERSATION" in user,
        "after removing the examples the OUTPUT FORMAT does not run into # CONVERSATION",
    )

    for anchor in (OUTPUT_FORMAT_START, OUTPUT_FORMAT_END):
        _check(user.count(anchor) == 1, f"{anchor!r} occurs {user.count(anchor)} times")

    start = user.index(OUTPUT_FORMAT_START)
    end = user.index(OUTPUT_FORMAT_END) + len(OUTPUT_FORMAT_END)
    user = user[:start] + OUTPUT_FORMAT + user[end:]
    _check(
        user.count(SAVE_RULE_UPSTREAM) == 1,
        f"upstream's save rule occurs {user.count(SAVE_RULE_UPSTREAM)} times",
    )
    user = user.replace(SAVE_RULE_UPSTREAM, SAVE_RULE)
    _check(
        user.count(NEAREST_UPSTREAM) == 1,
        f"upstream's find_nearest entry occurs {user.count(NEAREST_UPSTREAM)} times",
    )
    user = user.replace(NEAREST_UPSTREAM, NEAREST)

    for slot in (CONVERSATION_SLOT, CACHE_SLOT):
        _check(user.count(slot) == 1, f"{slot} occurs {user.count(slot)} times")

    return str(user)


def write_prompt() -> None:
    """Write data/t1_prompt.txt and record its digest in the manifest, if one exists.

    Args:
        None.
    """
    text = build_prompt()
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Bytes, not text mode: the template's trailing whitespace is meaningful.
    PROMPT_PATH.write_bytes(text.encode())
    sha = hashlib.sha256(text.encode()).hexdigest()
    manifest_path = SPLIT_ROOT / "split_manifest.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest["prompt_sha256"] = sha
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    log.info("t1 prompt: %d chars, sha256 %s, written to %s", len(text), sha, PROMPT_PATH)


# ---------------------------------------------------------------- corpus


def load_corpus(
    repo_id: str = HF_REPO,
) -> tuple[Path, dict[str, list[Dialogue]]]:
    """Every dialogue file of the pinned revision.

    Args:
        repo_id: Hub dataset repo.

    Returns:
        The snapshot directory, and domain:hf_split:n to that file's dialogues.
    """
    files = [
        dialogue_file(domain, hf_split, n)
        for domain in DOMAINS
        for hf_split in HF_SPLITS.values()
        for n in range(1, N_FILES + 1)
    ]
    root = snapshot(files + sorted(set(ONTOLOGY.values())), repo_id)
    corpus = {}

    for domain in DOMAINS:
        for hf_split in HF_SPLITS.values():
            for n in range(1, N_FILES + 1):
                corpus[f"{domain}:{hf_split}:{n}"] = read_dialogues(
                    root / dialogue_file(domain, hf_split, n)
                )

    return root, corpus


def replay_corpus(
    root: Path,
    corpus: dict[str, list[Dialogue]],
    semantics: tuple[str, ...],
    cache: Path | None = None,
) -> dict[str, dict[str, list[list[object] | None]]]:
    """Replay every gold plan of the corpus under each of semantics.

    Records keep digests of the summaries rather than the strings, so three
    full replays fit in memory.

    Args:
        root: Snapshot directory holding the ontology.
        corpus: File key to its dialogues.
        semantics: Which of SEMANTICS to run.
        cache: Pickle to reuse between runs, keyed on the code and the data.

    Returns:
        Semantics to dialogue key (file key:ID) to one record per row: None
        for an empty row, else [role, before, after, error, error type,
        seconds], summaries as digests.
    """
    key = [HF_REVISION, UPSTREAM_COMMIT, upstream_digest()]
    replays: dict[str, dict[str, list[list[object] | None]]] = {}

    if cache is not None and cache.is_file():
        stored = pickle.loads(cache.read_bytes())

        if stored["key"] == key:
            replays = stored["replays"]
            log.info("t1: reusing %s corpus replays from %s", sorted(replays), cache)

    missing = [name for name in semantics if name not in replays]
    tools = Tools(str(root)) if missing else None

    for name in missing:
        assert tools is not None
        replays[name] = {}
        started = time.perf_counter()

        for file_key, dialogues in corpus.items():
            records = replay_file(tools, [dialogue.rows for dialogue in dialogues], name)

            for dialogue, dialogue_records in zip(dialogues, records, strict=True):
                replays[name][f"{file_key}:{dialogue.id}"] = [
                    None
                    if record is None
                    else [
                        record["role"],
                        digest(str(record["before"])),
                        digest(str(record["after"])),
                        record["error"],
                        record["error_type"],
                        record["seconds"],
                    ]
                    for record in dialogue_records
                ]

        log.info("t1: %s replay took %.0fs", name, time.perf_counter() - started)

    if cache is not None and missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps({"key": key, "replays": replays}))

    return {name: replays[name] for name in semantics}


def corpus_stats(corpus: dict[str, list[Dialogue]]) -> dict[str, object]:
    """Count what §2.1 of the task statement states about the data, and check it.

    Args:
        corpus: File key to its dialogues.

    Returns:
        The per domain and split table plus the corpus-wide counts.
    """
    table: dict[str, dict[str, object]] = {}
    empty_rows = empty_with_plan = ends_assistant = same_role = 0
    starts_other = 0
    shares: Counter[str] = Counter()
    whitespace: set[tuple[str, str, str, int]] = set()
    templates: dict[str, list[object]] = {}
    ids_by_split: dict[str, dict[str, set[str]]] = {}

    for file_key, dialogues in corpus.items():
        domain, hf_split, _ = file_key.split(":")
        cell = table.setdefault(
            f"{domain}:{hf_split}",
            {
                "files": 0,
                "dialogues": 0,
                "ids_per_file": set(),
                "user_turns": 0,
                "no_plan": 0,
                "assistant_plans": 0,
            },
        )
        cell["files"] += 1  # type: ignore[operator]
        cell["dialogues"] += len(dialogues)  # type: ignore[operator]
        cell["ids_per_file"].add(len(dialogues))  # type: ignore[attr-defined]
        ids_by_split.setdefault(domain, {}).setdefault(hf_split, set()).update(
            dialogue.id for dialogue in dialogues
        )

        for dialogue in dialogues:
            rows = dialogue.rows
            templates.setdefault(f"{domain}:{hf_split}:{dialogue.id}", []).append(
                dialogue.templates
            )
            roles = []

            for template, plan in rows:
                if template is None:
                    empty_rows += 1
                    empty_with_plan += plan is not None
                    continue

                role = split_turn(template)[0]

                if role not in ("user", "assistant"):
                    raise ValueError(f"{file_key}:{dialogue.id}: role {role!r}")

                roles.append(role)

                if role == "assistant":
                    cell["assistant_plans"] += plan is not None  # type: ignore[operator]
                    continue

                cell["user_turns"] += 1  # type: ignore[operator]

                if plan is None:
                    cell["no_plan"] += 1  # type: ignore[operator]
                    shares["none"] += 1
                else:
                    shares[gold_kind(plan)] += 1

            for turn, index in enumerate(user_turns(rows)):
                plan = rows[index][1]

                if plan is not None and not plan.strip():
                    whitespace.add((domain, hf_split, dialogue.id, turn))

            starts_other += roles[0] != "assistant"
            ends_assistant += roles[-1] == "assistant"
            same_role += sum(a == b for a, b in itertools.pairwise(roles))

    listing = repo_files()
    csvs = [name for name in listing if name.endswith(".csv")]
    ontology = [name for name in csvs if name.startswith("ontology/")]
    _check(len(listing) == EXPECTED_FILES, f"the revision has {len(listing)} files")
    _check(len(csvs) - len(ontology) == 675, f"{len(csvs) - len(ontology)} dialogue CSVs")
    _check(len(ontology) == 9, f"{len(ontology)} ontology CSVs")
    _check(
        sorted(set(listing) - set(csvs)) == [".gitattributes", "README.md"],
        f"non-CSV files {sorted(set(listing) - set(csvs))}",
    )
    _check(len(corpus) == 675, f"{len(corpus)} dialogue files, expected 675")
    _check(empty_rows == EMPTY_ROWS, f"{empty_rows} empty rows, expected {EMPTY_ROWS}")
    _check(empty_with_plan == 0, f"{empty_with_plan} empty rows carry a plan")
    _check(starts_other == 0, f"{starts_other} dialogues do not start with the assistant")
    _check(ends_assistant == ENDS_WITH_ASSISTANT, f"{ends_assistant} end with the assistant")
    _check(same_role == SAME_ROLE_PAIRS, f"{same_role} adjacent same-role pairs")
    _check(
        all(len(set(map(tuple, versions))) == 1 for versions in templates.values()),  # type: ignore[arg-type]
        "an ID's Template column differs between files",
    )

    for domain, splits in ids_by_split.items():
        names = list(splits)

        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                _check(not splits[left] & splits[right], f"{domain}: IDs shared by splits")

    for hf_split in HF_SPLITS.values():
        users = sum(table[f"{d}:{hf_split}"]["user_turns"] for d in DOMAINS)  # type: ignore[misc]
        no_plan = sum(table[f"{d}:{hf_split}"]["no_plan"] for d in DOMAINS)  # type: ignore[misc]
        _check(users == USER_TURNS[hf_split], f"{hf_split}: {users} user turns")
        _check(no_plan == NO_PLAN_USER_TURNS[hf_split], f"{hf_split}: {no_plan} without plan")

        for domain in DOMAINS:
            cell = table[f"{domain}:{hf_split}"]
            _check(cell["files"] == N_FILES, f"{domain}:{hf_split}: {cell['files']} files")
            _check(
                cell["ids_per_file"] == {IDS_PER_FILE[hf_split]},
                f"{domain}:{hf_split}: IDs per file {cell['ids_per_file']}",
            )
            expected = ASSISTANT_PLANS.get(f"{domain}:{hf_split}", 0)
            _check(
                cell["assistant_plans"] == expected,
                f"{domain}:{hf_split}: {cell['assistant_plans']} assistant plans",
            )

    total = sum(shares.values())
    percent = {kind: round(100 * count / total, 1) for kind, count in shares.items()}
    _check(percent == GOLD_SHARES, f"gold shares {percent}")
    domain, hf_split, template_id, turn = WHITESPACE_PLAN
    expected_whitespace = {(domain, hf_split, template_id, turn)}
    _check(
        {(d, s, i, t) for d, s, i, t in whitespace} == expected_whitespace,
        f"whitespace-only plans at {sorted(whitespace)}",
    )

    return {
        "table": table,
        "empty_rows": empty_rows,
        "ends_with_assistant": ends_assistant,
        "same_role_pairs": same_role,
        "gold_shares": percent,
    }


def replay_stats(
    replays: dict[str, dict[str, list[list[object] | None]]],
) -> dict[str, object]:
    """Gold failures per semantics, and where session and file-wide replay disagree.

    Counted over user turns and assistant plans, as the task statement counts.

    Args:
        replays: replay_corpus's records.

    Returns:
        Failures per semantics and split, per domain, their error types, and
        the disagreement counts.
    """
    failures: dict[str, Counter[str]] = {}
    by_domain: dict[str, Counter[str]] = {}
    types: dict[str, Counter[str]] = {}
    disagree: Counter[str] = Counter()
    slowest = (0.0, "")

    for name, records in replays.items():
        failures[name] = Counter()
        types[name] = Counter()

        for key, rows in records.items():
            domain, hf_split = key.split(":")[:2]

            for index, record in enumerate(rows):
                if record is None:
                    continue

                seconds = float(record[5])  # type: ignore[arg-type]

                if name == "session" and seconds > slowest[0]:
                    slowest = (seconds, f"{key}:row{index}")

                if record[3] in ("", "success"):
                    continue

                failures[name][hf_split] += 1
                by_domain.setdefault(f"{name}:{hf_split}", Counter())[domain] += 1
                types[name][str(record[4])] += 1

    if "session" in replays and "file" in replays:
        for key, rows in replays["session"].items():
            for session, file in zip(rows, replays["file"][key], strict=True):
                if session is None or file is None:
                    continue

                if session[0] == "assistant" and session[3] == "":
                    continue

                disagree["before"] += session[1] != file[1]
                disagree["after"] += session[2] != file[2]
                disagree["error"] += session[3] != file[3]

    return {
        "failures": {name: dict(counts) for name, counts in failures.items()},
        "by_domain": {name: dict(counts) for name, counts in by_domain.items()},
        "error_types": {name: dict(counts) for name, counts in types.items()},
        "disagreements": dict(disagree),
        "slowest_gold": slowest,
    }


def check_replay_stats(stats: dict[str, object]) -> None:
    """Assert the replay facts of §2.3.

    Args:
        stats: replay_stats's result.
    """
    failures: dict[str, dict[str, int]] = stats["failures"]  # type: ignore[assignment]
    by_domain: dict[str, dict[str, int]] = stats["by_domain"]  # type: ignore[assignment]

    for name, counts in failures.items():
        expected = {k: v for k, v in GOLD_FAILURES[name].items() if v}
        _check(counts == expected, f"{name} gold failures {counts}, expected {expected}")

    for name, expected in FAILURES_BY_DOMAIN.items():
        if name.split(":")[0] in failures:
            _check(by_domain.get(name) == expected, f"{name} failures {by_domain.get(name)}")

    if "file" in failures:
        _check(
            stats["disagreements"] == DISAGREEMENTS,
            f"session and file-wide replay disagree on {stats['disagreements']}",
        )


# ---------------------------------------------------------------- split


def eligible_turns(
    corpus: dict[str, list[Dialogue]],
    replays: dict[str, dict[str, list[list[object] | None]]],
) -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    """Every user turn that may ship, with the rule that excluded the others.

    Args:
        corpus: File key to its dialogues.
        replays: Session and file-wide records from replay_corpus.

    Returns:
        The eligible turns with their keys, gold kind and digests, and per
        rule the (train, validation, test) count it excluded.
    """
    order = list(HF_SPLITS.values())
    excluded = {rule: [0, 0, 0] for rule in EXCLUDED}
    out: list[dict[str, object]] = []

    for file_key, dialogues in corpus.items():
        domain, hf_split, n = file_key.split(":")

        for dialogue in dialogues:
            rows = dialogue.rows
            key = f"{file_key}:{dialogue.id}"
            session = replays["session"][key]
            file = replays["file"][key]

            for turn, index in enumerate(user_turns(rows)):
                plan = rows[index][1]
                prefix = range(index + 1)

                if plan is None:
                    rule = "rule1"
                elif any(_failed(session[i]) for i in prefix):
                    rule = "rule2"
                elif any(_outcome(session[i]) != _outcome(file[i]) for i in prefix):
                    rule = "rule3"
                else:
                    rule = ""

                if rule:
                    excluded[rule][order.index(hf_split)] += 1
                    continue

                assert plan is not None
                out.append(
                    {
                        "id": item_id(domain, hf_split, int(n), dialogue.id, turn),
                        "domain": domain,
                        "hf_split": hf_split,
                        "file_index": int(n),
                        "template_id": dialogue.id,
                        "turn": turn,
                        "gold_kind": gold_kind(plan),
                        "plan_digest": digest(plan),
                        "cache_digest": session[index][1],  # type: ignore[index]
                        "dialogue": key,
                        "blank": not plan.strip(),
                    }
                )

    return out, excluded


def _failed(record: list[object] | None) -> bool:
    """Whether a replayed row ran a plan that raised.

    Args:
        record: One replay record, None for an empty row.

    Returns:
        True when the row's error is neither empty nor success.
    """
    return record is not None and record[3] not in ("", "success")


def _outcome(record: list[object] | None) -> list[object] | None:
    """What rule 3 compares of a replayed row: both summaries and the error.

    Args:
        record: One replay record, None for an empty row.

    Returns:
        The pre-turn and post-turn summary digests and the error, or None.
    """
    return None if record is None else record[1:4]


def select(
    pool: list[dict[str, object]], split: str, total: int, templates: dict[str, set[str]]
) -> tuple[list[dict[str, object]], list[str]]:
    """Draw one split from its eligible pool, following the stated rules.

    Every draw ranks by _unit_hash of a formatted key, so the result follows
    from the rules alone and not from any RNG call order.

    Args:
        pool: Eligible turns of the T1 split this split draws from.
        split: This repo's split name, which the hash keys carry.
        total: Items to draw.
        templates: Domain to every template ID of its T1 split.

    Returns:
        The drawn entries, and the templates without an eligible turn.
    """
    quotas = {domain: total // len(DOMAINS) for domain in DOMAINS}

    for domain in sorted(DOMAINS, key=lambda d: _unit_hash(f"{SEED}:quota:{split}:{d}"))[
        : total % len(DOMAINS)
    ]:
        quotas[domain] += 1

    chosen: list[dict[str, object]] = []
    dropped: list[str] = []

    for domain in DOMAINS:
        by_template: dict[str, list[dict[str, object]]] = {}

        for entry in pool:
            if entry["domain"] == domain:
                by_template.setdefault(str(entry["template_id"]), []).append(entry)

        dropped += sorted(
            f"{domain}:{HF_SPLITS[split]}:{t}" for t in templates[domain] - set(by_template)
        )
        names = sorted(by_template)
        allocation = {template: quotas[domain] // len(names) for template in names}

        for template in sorted(
            names, key=lambda t: _unit_hash(f"{SEED}:alloc:{split}:{domain}:{t}")
        )[: quotas[domain] % len(names)]:
            allocation[template] += 1

        for template in names:
            ranked = sorted(
                by_template[template], key=lambda e: _unit_hash(f"{SEED}:item:{e['id']}")
            )
            used: set[object] = set()
            taken = 0

            for entry in ranked:
                if taken == allocation[template]:
                    break

                if entry["dialogue"] in used:
                    continue

                used.add(entry["dialogue"])
                chosen.append(entry)
                taken += 1

            if taken < allocation[template]:
                raise ValueError(
                    f"{domain}:{template} ({split}) offers {taken} dialogues for an allocation of "
                    f"{allocation[template]}; the stated rules never need to redistribute"
                )

    return chosen, dropped


def build_split(
    repo_id: str = HF_REPO, cache: Path | None = None, write: bool = True
) -> dict[str, object]:
    """Replay the corpus, draw the split, check every stated number, write it out.

    Args:
        repo_id: Hub dataset repo.
        cache: Pickle of the corpus replays to reuse between runs.
        write: Whether the manifest and items files are written.

    Returns:
        The manifest, with the drawn entries under splits.
    """
    root, corpus = load_corpus(repo_id)
    replays = replay_corpus(root, corpus, ("session", "file"), cache)
    stats = replay_stats(replays)
    check_replay_stats(stats)

    return draw(corpus, replays, write)


def draw(
    corpus: dict[str, list[Dialogue]],
    replays: dict[str, dict[str, list[list[object] | None]]],
    write: bool = True,
) -> dict[str, object]:
    """Eligibility, selection, the checks, and the shipped files.

    Args:
        corpus: File key to its dialogues.
        replays: Session and file-wide records.
        write: Whether the manifest and items files are written.

    Returns:
        The manifest, with the drawn entries under splits.
    """
    templates: dict[str, dict[str, set[str]]] = {}

    for file_key, dialogues in corpus.items():
        domain, hf_split, _ = file_key.split(":")
        templates.setdefault(hf_split, {}).setdefault(domain, set()).update(
            dialogue.id for dialogue in dialogues
        )

    eligible, excluded = eligible_turns(corpus, replays)
    order = list(HF_SPLITS.values())

    for rule, removed in excluded.items():
        _check(tuple(removed) == EXCLUDED[rule], f"{rule} excludes {removed}")

    for domain in DOMAINS:
        pools = tuple(
            sum(1 for e in eligible if e["domain"] == domain and e["hf_split"] == s) for s in order
        )
        _check(pools == ELIGIBLE[domain], f"{domain}: eligible pools {pools}")

    reached = tuple(len({e["dialogue"] for e in eligible if e["hf_split"] == s}) for s in order)
    _check(reached == DIALOGUES_WITH_ELIGIBLE, f"dialogues with an eligible turn {reached}")
    totals = {"train": 400, "val": 200, "test": 1400}
    splits: dict[str, list[dict[str, object]]] = {}
    dropped: list[str] = []

    for split in SPLITS:
        pool = [e for e in eligible if e["hf_split"] == HF_SPLITS[split]]
        splits[split], lost = select(pool, split, totals[split], templates[HF_SPLITS[split]])
        dropped += lost

    _check(dropped == DROPPED_TEMPLATES, f"the allocation drops {dropped}")
    everything = [e for split in splits.values() for e in split]
    _check(
        len({e["dialogue"] for e in everything}) == len(everything),
        "a dialogue appears twice in the split",
    )

    for split, entries in splits.items():
        column = SPLITS.index(split)

        for domain in DOMAINS:
            count = sum(1 for e in entries if e["domain"] == domain)
            _check(count == SHIPPED[domain][column], f"{split}:{domain}: {count} items")

        ids = [str(e["id"]) for e in entries]
        _check(ids_digest(ids) == SHIPPED_DIGESTS[split], f"{split} digest {ids_digest(ids)}")
        kinds = Counter(str(e["gold_kind"]) for e in entries)

        for kind, expected in SHIPPED_KINDS.items():
            _check(kinds[kind] == expected[column], f"{split}: {kinds[kind]} {kind} items")

    blank = sorted(str(e["id"]) for e in everything if e["blank"])
    _check(blank == sorted(WHITESPACE_ITEMS), f"whitespace-only gold items {blank}")

    for split in SPLITS:
        splits[split].sort(key=lambda e: _unit_hash(f"{SEED}:order:{e['id']}"))

    manifest = _manifest(splits)

    if write:
        _write(manifest, splits)

    return {**manifest, "splits": splits}


def _manifest(splits: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """The manifest describing how the split was drawn.

    Args:
        splits: Split name to its entries.

    Returns:
        The manifest mapping.
    """
    prompt_sha = (
        hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest() if PROMPT_PATH.is_file() else ""
    )

    return {
        "benchmark": "T1",
        "manifest_type": "id_split",
        "source_repo": HF_REPO,
        "source_repo_type": "dataset",
        "source_url": f"https://huggingface.co/datasets/{HF_REPO}",
        "source_revision": HF_REVISION,
        "source_license": "CC-BY-4.0",
        "vendored_code": UPSTREAM_URL,
        "vendored_commit": UPSTREAM_COMMIT,
        "seed": SEED,
        "counts": {split: len(entries) for split, entries in splits.items()},
        "split_sources": HF_SPLITS,
        "id_format": "{domain}:{hf_split}:{file_index}:{ID}:{turn}",
        "replay_semantics": "session: one locals dict per dialogue, a fresh copy of "
        "process_model_output.py's globals per exec, the cache reset per dialogue",
        "eligibility": [
            "1. The user turn has a gold Filled_Plan (a whitespace-only plan counts).",
            "2. Under session replay no gold plan of its dialogue raises, up to and "
            "including this turn, on user and assistant rows alike.",
            "3. Session replay and upstream's file-wide replay (one locals dict per file, "
            "dialogues in groupby('ID') order, i.e. Python <= 3.12) give identical "
            "pre-turn summaries, post-turn summaries and errors on every row of the "
            "dialogue up to and including this turn. Summaries compare as str() of "
            "the dict, key order included.",
        ],
        "selection": [
            "1. Domain quotas: total // 9 each; the total % 9 domains with the lowest "
            "_unit_hash('42:quota:{split}:{domain}') get one more.",
            "2. Template allocation: the templates of the domain's T1 split with at least "
            "one eligible turn get quota // n_templates each; the quota % n_templates "
            "with the lowest _unit_hash('42:alloc:{split}:{domain}:{ID}') get one more.",
            "3. Items: a template's eligible turns ranked by _unit_hash('42:item:{id}'), "
            "taken in order, skipping any turn whose dialogue already has an item, "
            "until the allocation is met.",
            "Order: items.json is sorted by _unit_hash('42:order:{id}').",
            "Hash keys use this repo's split name (train, val, test) and unpadded "
            "integers; _unit_hash is skillpng.benchmarks.base._unit_hash.",
        ],
        "dropped_templates": DROPPED_TEMPLATES,
        "per_domain": {
            split: {domain: sum(1 for e in entries if e["domain"] == domain) for domain in DOMAINS}
            for split, entries in splits.items()
        },
        "digests": {
            split: ids_digest([str(e["id"]) for e in entries]) for split, entries in splits.items()
        },
        "digest_rule": "sha256 of the sorted ids joined by newlines, first 16 hex characters",
        "gold_kinds": {
            split: dict(sorted(Counter(str(e["gold_kind"]) for e in entries).items()))
            for split, entries in splits.items()
        },
        "prompt_file": "t1_prompt.txt",
        "prompt_sha256": prompt_sha,
        "item_fields": ITEM_FIELDS,
        "notes": [
            "This is a split manifest, not the T1 payload. Materialize the items from ",
            f"the dialogue CSVs of {HF_REPO} at source_revision; an item is one user turn ",
            "of one dialogue, keyed (domain, T1 split, file, ID, turn), where ID is a ",
            "template id repeated once per file and turn counts the dialogue's non-empty ",
            "user rows from 0. Train, val and test are drawn from T1's train, validation ",
            "and test splits respectively, which are template- and city-disjoint, so ",
            "nothing crosses a T1 boundary. Every item comes from a different dialogue. ",
            "plan_digest is the sha256 of the gold Filled_Plan and cache_digest the sha256 ",
            "of the pre-turn cache summary string, both cut to 16 hex characters; the ",
            "adapter refuses an item whose data or replay no longer gives them. Test ",
            "template 26 of flighthotelrestaurant has no eligible turn: its second user ",
            "turn calls search_nearest(hotels=hotels) before hotels exists, which only ",
            "runs upstream because a same-named variable leaks in from another dialogue ",
            "of the file. Three train items (flight template 47, turn 2) have a ",
            "whitespace-only gold plan: their target is an empty <CODE> block.",
        ],
    }


def _write(manifest: dict[str, object], splits: dict[str, list[dict[str, object]]]) -> None:
    """Write the manifest and the per-split items files.

    Args:
        manifest: The manifest mapping.
        splits: Split name to its entries, in canonical order.
    """
    for split, entries in splits.items():
        path = SPLIT_ROOT / split / "items.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{name: entry[name] for name in ITEM_FIELDS} for entry in entries]
        path.write_text(json.dumps(rows, indent=1) + "\n")

    (SPLIT_ROOT / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("t1 split written to %s", SPLIT_ROOT)


# ---------------------------------------------------------------- audit


def audit(
    repo_id: str = HF_REPO, cache: Path | None = None, tokenizer: str = "Qwen/Qwen3.5-9B"
) -> None:
    """Print everything §13.1 asks for, asserting what the statement gives.

    The cold item build runs first, so its peak RSS is measured before the
    corpus replay imports the ontology into this process.

    Args:
        repo_id: Hub dataset repo.
        cache: Pickle of the corpus replays to reuse between runs.
        tokenizer: Tokenizer the prompt and target budgets are counted with.
    """
    build = _cold_build()
    tokens = _token_stats(build.pop("items"), tokenizer)  # type: ignore[arg-type]
    root, corpus = load_corpus(repo_id)
    stats = corpus_stats(corpus)
    replays = replay_corpus(root, corpus, SEMANTICS, cache)
    replay = replay_stats(replays)
    check_replay_stats(replay)
    drawn = draw(corpus, {k: replays[k] for k in ("session", "file")}, write=False)
    shipped = json.loads((SPLIT_ROOT / "split_manifest.json").read_text())
    _check(drawn["digests"] == shipped["digests"], "the drawn split is not the shipped one")
    print(_render_audit(stats, replay, drawn, tokens, build))


def _cold_build() -> dict[str, object]:
    """Build all 2,000 items through the adapter with an empty item cache.

    Args:
        None.

    Returns:
        Wall time, peak RSS of this process and of the worker, and the items.
    """
    from ...config import load_config
    from . import t1

    cfg = load_config(CONFIG_PATH)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        t1.ITEM_CACHE_ROOT = Path(tmp)
        bench = t1.T1(cfg.benchmark)
        items = {split: bench.items(split) for split in SPLITS}
        seconds = time.perf_counter() - started
        bench.close()
        t1.ITEM_CACHE_ROOT = None

    scale = 1 if sys.platform == "darwin" else 1024

    return {
        "seconds": seconds,
        "rss_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
        "rss_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale,
        "items": (bench, items),
    }


def _token_stats(built: tuple[object, dict[str, list[object]]], name: str) -> dict[str, object]:
    """Prompt and target token counts over the shipped items.

    Prompts are the system and user turns under the chat template with the
    generation prompt and thinking off; targets carry no eos or suffix.

    Args:
        built: The adapter and its items per split.
        name: Tokenizer to load, from the local Hub cache.

    Returns:
        min, p50, p95 and max of prompt, target and their sum, plus the TOOL
        CONFIG's own count.
    """
    from transformers import AutoTokenizer

    bench, items = built
    tokenizer = AutoTokenizer.from_pretrained(name)
    prompt: list[int] = []
    target: list[int] = []

    for split_items in items.values():
        for item in split_items:
            messages = bench.build_messages(item, with_image=False)  # type: ignore[attr-defined]
            encoded: object = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
            # A BatchEncoding under transformers 5, a list of ids before.
            ids: list[int] = encoded["input_ids"] if hasattr(encoded, "keys") else encoded  # type: ignore[index, assignment]
            prompt.append(len(ids))
            target.append(len(tokenizer(item.target, add_special_tokens=False)["input_ids"]))  # type: ignore[attr-defined]

    text = PROMPT_PATH.read_text()
    start = text.index("# TOOL CONFIG\n") + len("# TOOL CONFIG\n")
    config = text[start : text.index("\n\n# INSTRUCTIONS")]

    return {
        "tokenizer": name,
        "prompt": _percentiles(prompt),
        "target": _percentiles(target),
        "total": _percentiles([a + b for a, b in zip(prompt, target, strict=True)]),
        "tool_config": len(tokenizer(config, add_special_tokens=False)["input_ids"]),
    }


def _percentiles(values: list[int]) -> dict[str, int]:
    """Nearest-rank min, p50, p95 and max.

    Args:
        values: Token counts.

    Returns:
        The four statistics.
    """
    ordered = sorted(values)

    def rank(p: float) -> int:
        """Nearest-rank percentile.

        Args:
            p: Percentile in (0, 100].

        Returns:
            The value at that rank.
        """
        return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]

    return {"min": ordered[0], "p50": rank(50), "p95": rank(95), "max": ordered[-1]}


def _render_audit(
    stats: dict[str, object],
    replay: dict[str, object],
    drawn: dict[str, object],
    tokens: dict[str, object],
    build: dict[str, object],
) -> str:
    """Format the audit as plain text tables.

    Args:
        stats: corpus_stats's result.
        replay: replay_stats's result.
        drawn: draw's result.
        tokens: _token_stats's result.
        build: _cold_build's timings.

    Returns:
        The rendered report.
    """
    lines = ["== corpus (per domain x T1 split)"]
    lines.append(
        f"   {'domain:split':<34s} {'files':>5s} {'dialog':>6s} {'IDs/f':>5s} "
        f"{'users':>6s} {'noplan':>6s} {'asst':>4s}"
    )
    table: dict[str, dict[str, object]] = stats["table"]  # type: ignore[assignment]

    for key, cell in table.items():
        per_file = ",".join(str(v) for v in sorted(cell["ids_per_file"]))  # type: ignore[call-overload]
        lines.append(
            f"   {key:<34s} {cell['files']:>5} {cell['dialogues']:>6} {per_file:>5s} "
            f"{cell['user_turns']:>6} {cell['no_plan']:>6} {cell['assistant_plans']:>4}"
        )

    lines.append(
        f"   empty rows {stats['empty_rows']}, dialogues ending with the assistant "
        f"{stats['ends_with_assistant']}, adjacent same-role pairs {stats['same_role_pairs']}"
    )
    lines.append(f"   gold kind shares of user turns (%): {stats['gold_shares']}")
    lines.append("")
    lines.append("== gold replay")

    for name, counts in replay["failures"].items():  # type: ignore[attr-defined]
        lines.append(f"   {name:<8s} failures {counts}  types {replay['error_types'][name]}")  # type: ignore[index]

    for name, counts in replay["by_domain"].items():  # type: ignore[attr-defined]
        lines.append(f"   {name:<14s} {counts}")

    lines.append(f"   session vs file-wide disagreements: {replay['disagreements']}")
    slowest: tuple[float, str] = replay["slowest_gold"]  # type: ignore[assignment]
    seconds, where = slowest
    lines.append(f"   slowest gold exec (session): {seconds:.2f}s at {where}")
    lines.append("")
    lines.append("== split")
    lines.append(f"   dropped templates {drawn['dropped_templates']}")

    for split in SPLITS:
        lines.append(
            f"   {split:<5s} {drawn['counts'][split]:>5}  digest {drawn['digests'][split]}  "  # type: ignore[index]
            f"kinds {drawn['gold_kinds'][split]}"  # type: ignore[index]
        )
        lines.append(f"         {drawn['per_domain'][split]}")  # type: ignore[index]

    lines.append("")
    lines.append(f"== tokens ({tokens['tokenizer']}, thinking off, nearest rank)")

    for name in ("prompt", "target", "total"):
        lines.append(f"   {name:<7s} {tokens[name]}")

    lines.append(f"   TOOL CONFIG alone: {tokens['tool_config']}")
    lines.append("")
    lines.append("== cold item build (all 2,000 items, empty item cache, data already local)")
    lines.append(
        f"   {build['seconds']:.0f}s, peak RSS {build['rss_self'] / 2**30:.2f} GiB (this "  # type: ignore[operator]
        f"process), {build['rss_worker'] / 2**30:.2f} GiB (worker)"  # type: ignore[operator]
    )

    return "\n".join(lines)


def _check(condition: bool, message: str) -> None:
    """Stop the build on a fact that does not reproduce.

    Args:
        condition: Whether the fact holds.
        message: What was found instead.
    """
    if not condition:
        raise ValueError(
            f"t1 build: {message}. This contradicts the verified facts of the task "
            "statement; stop and report it rather than working around it."
        )


def main() -> None:
    """Command line entry point for building and auditing the shipped artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", choices=["prompt", "split", "audit"])
    parser.add_argument("--repo-id", default=HF_REPO)
    parser.add_argument(
        "--replay-cache",
        type=Path,
        default=None,
        help="pickle the full-corpus replays here and reuse them on the next run",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.artifact == "prompt":
        write_prompt()
    elif args.artifact == "split":
        build_split(args.repo_id, args.replay_cache)
    else:
        audit(args.repo_id, args.replay_cache, args.tokenizer)


if __name__ == "__main__":
    main()
