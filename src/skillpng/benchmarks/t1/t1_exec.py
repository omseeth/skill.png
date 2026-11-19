"""T1 executor: gold replay and generated code, run against the vendored tools.

Seven vendored tool modules read the 1.35 GB ontology into module globals at
import, so they are imported in exactly two places: the worker process this
module starts, and t1_build. The main process and the pytest process never
import them.

Execution follows upstream's evaluation/process_model_output.py:

- Globals are a fresh copy of the namespace that script has at exec time:
  every tool, every cache function, configure_tools_definitions, pd, os, re,
  csv, Path and the typing names. Locals are a separate dict, as upstream's
  bare exec inside a function has them, so a name bound by the executed code
  and read inside a lambda or a generator expression raises NameError there
  as it does upstream.
- Gold replay runs one locals dict per dialogue ("session"). Upstream on
  Python <= 3.12 keeps one per file, which lets a variable leak from one
  dialogue into the next; the shipped split excludes every turn where the two
  differ, so on shipped items our gold is the paper's gold.
- Generated code runs against a deep copy of the gold session locals and the
  gold cache as they stand before the turn: teacher-forced state. Both are
  copied in one call, so an object bound to a variable and saved to the cache
  stays one object, and generated code can never change what gold computes.

The worker is one spawned process, never forked: the parent may hold CUDA.
Every request replays its dialogue from an empty cache, so no state is carried
from one item to the next and a restart changes nothing; any rebinding the
code made in builtins or the tool modules is undone after it runs. The worker
reports when the replay is done, so the per-item timeout covers the generated
code only. On expiry the parent kills the worker and records "timeout"; the
next request starts a new one. The pipe is the only channel: stdin is closed,
so input() raises, and whatever executed code prints is discarded as it is
written.

Top-level imports are the standard library only, so this file also runs by
path under upstream's own pins, against upstream's own t1 package; see
run_prefixes and the environment drift test.
"""

from __future__ import annotations

import builtins
import contextlib
import copy
import importlib
import io
import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, fields
from multiprocessing.connection import Connection
from pathlib import Path

# The tool package: our vendored copy, or upstream's t1 for the drift check.
VENDORED = "skillpng.benchmarks.t1.upstream"

# Environment variables the tool modules read at import, relative to the
# dataset snapshot. ALL_AIRPORTS is read by no module; upstream's README lists
# it, so it is set anyway.
ONTOLOGY = {
    "ALL_FLIGHTS": "ontology/single-domain/t1_flight_data.csv",
    "ALL_HOTELS": "ontology/single-domain/t1_hotel_data.csv",
    "ALL_RESTAURANTS": "ontology/single-domain/t1_restaurant_data.csv",
    "ALL_ATTRACTIONS": "ontology/single-domain/t1_attraction_data.csv",
    "HOTEL_ATTRACTIONS": "ontology/multi-domain/t1_hotel_attraction_data.csv",
    "HOTEL_RESTAURANTS": "ontology/multi-domain/t1_hotel_restaurant_data.csv",
    "RESTAURANT_ATTRACTIONS": "ontology/multi-domain/t1_restaurant_attraction_data.csv",
    "ALL_AIRPORTS": "ontology/miscellaneous/t1_all_airports.csv",
}

# The module-level names of evaluation/process_model_output.py at exec time,
# as (module, attribute). A module starting with a dot is inside the tool
# package; attribute None binds the module itself. Its own functions are
# left out. A needs_clone test pins this against upstream's file.
GLOBALS: dict[str, tuple[str, str | None]] = {
    "csv": ("csv", None),
    "os": ("os", None),
    "re": ("re", None),
    "Path": ("pathlib", "Path"),
    "Any": ("typing", "Any"),
    "Dict": ("typing", "Dict"),
    "List": ("typing", "List"),
    "Optional": ("typing", "Optional"),
    "Tuple": ("typing", "Tuple"),
    "pd": ("pandas", None),
    "adjust_date": (".tools.adjust_date", "adjust_date"),
    "dump_cache_query": (".tools.cache", "dump_cache_query"),
    "dump_entire_cache": (".tools.cache", "dump_entire_cache"),
    "get_cache": (".tools.cache", "get_cache"),
    "get_cache_for_current_turn": (".tools.cache", "get_cache_for_current_turn"),
    "reset_cache": (".tools.cache", "reset_cache"),
    "retrieve_ground_truth_cache": (".tools.cache", "retrieve_ground_truth_cache"),
    "get_results_from_cache": (".tools.cache", "get_results_from_cache"),
    "save_to_cache": (".tools.cache", "save_to_cache"),
    "filter_attractions": (".tools.filter_attractions", "filter_attractions"),
    "filter_flights": (".tools.filter_flights", "filter_flights"),
    "filter_hotels": (".tools.filter_hotels", "filter_hotels"),
    "filter_restaurants": (".tools.filter_restaurants", "filter_restaurants"),
    "search_nearest": (".tools.find_nearest", "search_nearest"),
    "search_attractions": (".tools.search_attractions", "search_attractions"),
    "search_flights": (".tools.search_flights", "search_flights"),
    "search_hotels": (".tools.search_hotels", "search_hotels"),
    "search_restaurants": (".tools.search_restaurants", "search_restaurants"),
    "seek_information": (".tools.seek_information", "seek_information"),
    "sort_results": (".tools.sort_results", "sort_results"),
    "configure_tools_definitions": (
        ".tools.utils.get_tool_configurations",
        "configure_tools_definitions",
    ),
}

# Replay semantics for t1_build and the audit: a fresh locals dict per exec
# (upstream on Python >= 3.13), one per dialogue (ours), one per file in
# upstream's order (upstream on Python <= 3.12, i.e. the paper).
SEMANTICS = ("fresh", "session", "file")

SUCCESS = "success"
TIMEOUT = "timeout"
# The worker imports the whole ontology before it answers; a slow shared
# filesystem can take minutes over it.
STARTUP_TIMEOUT_S = 900.0
# Gold replay is trusted code that takes milliseconds per turn; this only
# catches a wedged worker.
REPLAY_TIMEOUT_S = 600.0

Row = list[str | None]
"""One CSV row as the executor sees it: [Filled_Template, Filled_Plan], None
where the CSV cell is empty."""


@dataclass
class Built:
    """One item as build_turn makes it from its dialogue."""

    role: str
    question: str
    # The conversation so far, as inference.py formats it: f"{history}".
    conversation: str
    # The pre-turn cache summary dict, as a string.
    cache: str
    # [row index, error, error type] for every gold plan before the turn.
    errors: list[list[object]]
    gold_error: str
    gold_error_type: str
    gold_after: dict[str, object]


@dataclass
class Outcome:
    """One execute request: gold's outcome at the turn and the generated code's."""

    # The pre-turn cache summary string the replay reached.
    before: str
    gold_error: str
    gold_error_type: str
    gold_after: dict[str, object]
    # "success", str(exception), "timeout" or "worker died (...)".
    error: str
    error_type: str
    # The post-turn summary, None when the code did not run to an outcome.
    after: dict[str, object] | None
    timeout: bool
    # False when there is no post-turn state to score.
    executed: bool
    # Builtins and tool-module names the code rebound, undone since.
    restored: list[str]


def split_turn(template: str) -> tuple[str, str]:
    """Split a Filled_Template into role and utterance, as inference.py does.

    The role is normalized by substring and the utterance keeps its leading
    space: the prompt's conversation string is built from exactly this.

    Args:
        template: A non-empty Filled_Template cell.

    Returns:
        The role and the unstripped utterance.
    """
    role, utterance = template.split(":", 1)

    if "user" in role:
        role = "user"
    elif "assistant" in role:
        role = "assistant"

    return role, utterance


class Tools:
    """The vendored tools, imported once, plus the exec namespace built from them."""

    def __init__(self, root: str, package: str = VENDORED) -> None:
        """Point the ontology variables at root, then import every tool.

        Args:
            root: Dataset snapshot directory holding ontology/.
            package: Tool package to import, the vendored one by default.
        """
        for variable, relative in ONTOLOGY.items():
            os.environ[variable] = str(Path(root) / relative)

        self.namespace: dict[str, object] = {"__name__": "__main__"}

        for name, (module, attribute) in GLOBALS.items():
            loaded = importlib.import_module(package + module if module[0] == "." else module)
            self.namespace[name] = loaded if attribute is None else getattr(loaded, attribute)

        self.cache_module = importlib.import_module(package + ".tools.cache")
        # What every module the executed code can reach binds at startup, so
        # a rebinding (import builtins; builtins.len = None) is undone before
        # the next item runs. The cache module's cache is the state managed
        # here, so it is left out.
        self._pristine = [
            (module, dict(vars(module)))
            for name, module in sorted(sys.modules.items())
            if module is builtins or name == package or name.startswith(package + ".")
        ]

    def restore(self) -> list[str]:
        """Undo every rebinding executed code made in builtins or the tool modules.

        Args:
            None.

        Returns:
            The qualified names that had to be restored.
        """
        restored: list[str] = []

        for module, saved in self._pristine:
            current = vars(module)

            # The warnings machinery files what it has shown per module; that
            # is bookkeeping, not state the code could steer a tool with.
            for name in [n for n in current if n not in saved and n != "__warningregistry__"]:
                del current[name]
                restored.append(f"{module.__name__}.{name}")

            for name, value in saved.items():
                if module is self.cache_module and name == "cache":
                    continue

                if name not in current or current[name] is not value:
                    current[name] = value
                    restored.append(f"{module.__name__}.{name}")

        return restored

    def fresh_globals(self) -> dict[str, object]:
        """A new copy of the exec globals, with builtins of its own.

        Args:
            None.

        Returns:
            The namespace executed code sees as its globals.
        """
        namespace = dict(self.namespace)
        # Copied, so code that rebinds a builtin cannot reach the next item.
        namespace["__builtins__"] = dict(vars(builtins))

        return namespace

    def reset(self) -> None:
        """Empty the tool cache with upstream's own reset_cache.

        Args:
            None.
        """
        reset = self.namespace["reset_cache"]
        assert callable(reset)
        reset()

    def cache(self) -> dict[str, object]:
        """The live tool cache, key to SearchResultsOutputModel.

        Args:
            None.

        Returns:
            The module global dict itself, not a copy.
        """
        cache: dict[str, object] = vars(self.cache_module)["cache"]
        return cache

    def set_cache(self, cache: dict[str, object]) -> None:
        """Replace the tool cache, as retrieve_ground_truth_cache does.

        Args:
            cache: The dict the tools read and write from now on.
        """
        vars(self.cache_module)["cache"] = cache

    def summary(self) -> dict[str, object]:
        """The cache summary upstream shows and scores: key to query_summary.

        Args:
            None.

        Returns:
            One entry per cache key, in insertion order.
        """
        return {key: value.query_summary for key, value in self.cache().items()}  # type: ignore[attr-defined]

    def run(
        self,
        code: str,
        scope: dict[str, object],
        catch: tuple[type[BaseException], ...] = (Exception,),
    ) -> list[str]:
        """Exec code once, recording the outcome the way upstream does.

        Args:
            code: Python source to execute.
            scope: Locals dict the code binds its names in.
            catch: Exception types recorded rather than raised.

        Returns:
            "success" or str(exception), and the exception type name.
        """
        try:
            exec(code, self.fresh_globals(), scope)
        except catch as error:
            return [str(error), type(error).__name__]

        return [SUCCESS, ""]


def replay_dialogue(
    tools: Tools, rows: list[Row], scope: dict[str, object] | None
) -> list[dict[str, object] | None]:
    """Replay one dialogue's gold from an empty cache, as upstream's first pass does.

    The pre-turn summary is taken before an empty row is skipped, and every
    row with a plan runs it, whatever its role.

    Args:
        tools: The imported tools.
        rows: The dialogue's rows in file order.
        scope: Locals dict every plan shares, or None for a fresh one each.

    Returns:
        Per row, None when its template is empty, else its role, the pre- and
        post-turn summary strings, the error, its type and the exec seconds.
    """
    tools.reset()
    records: list[dict[str, object] | None] = []

    for template, plan in rows:
        before = str(tools.summary())

        if template is None:
            records.append(None)
            continue

        role, _ = split_turn(template)
        error, kind, seconds = "", "", 0.0

        if plan is not None:
            started = time.perf_counter()
            error, kind = tools.run(plan, {} if scope is None else scope)
            seconds = time.perf_counter() - started

        records.append(
            {
                "role": role,
                "before": before,
                "after": str(tools.summary()),
                "error": error,
                "error_type": kind,
                "seconds": seconds,
            }
        )

    return records


def replay_file(
    tools: Tools, dialogues: list[list[Row]], semantics: str
) -> list[list[dict[str, object] | None]]:
    """Replay every dialogue of one file under one of SEMANTICS.

    Args:
        tools: The imported tools.
        dialogues: The file's dialogues in upstream order, IDs sorted as strings.
        semantics: fresh, session or file.

    Returns:
        replay_dialogue's records, one list per dialogue.
    """
    if semantics not in SEMANTICS:
        raise ValueError(f"semantics={semantics!r}; expected one of {SEMANTICS}")

    shared: dict[str, object] = {}
    out: list[list[dict[str, object] | None]] = []

    for rows in dialogues:
        scope = None if semantics == "fresh" else {} if semantics == "session" else shared
        out.append(replay_dialogue(tools, rows, scope))
        tools.reset()

    return out


def replay_plans(rows: list[Row]) -> list[str]:
    """The gold plans a session replay of rows executes, in order.

    Args:
        rows: Dialogue rows before the item's turn.

    Returns:
        Every plan on a row whose template is not empty.
    """
    return [plan for template, plan in rows if template is not None and plan is not None]


def build_turn(tools: Tools, rows: list[Row]) -> dict[str, object]:
    """Build one item from its dialogue: conversation, cache, and gold outcome.

    The last row is the item's turn. Gold code never enters the history;
    earlier code shows only through the cache summary, which is taken before
    the turn's own gold runs.

    Args:
        tools: The imported tools.
        rows: The dialogue's rows up to and including the item's turn.

    Returns:
        The conversation and cache strings, the question, every gold error
        before the turn, and the turn's gold error and post-turn summary.
    """
    tools.reset()
    session: dict[str, object] = {}
    history: list[dict[str, str]] = []
    errors: list[list[object]] = []

    for index, (template, plan) in enumerate(rows):
        before = tools.summary()

        if template is None:
            continue

        role, utterance = split_turn(template)
        history.append({role: utterance})

        if index == len(rows) - 1:
            error, kind = tools.run(plan, session) if plan is not None else ["", ""]

            return {
                "role": role,
                "question": utterance.strip(),
                "conversation": f"{history}",
                "cache": str(before),
                "errors": errors,
                "gold_error": error,
                "gold_error_type": kind,
                "gold_after": tools.summary(),
            }

        if plan is not None:
            error, kind = tools.run(plan, session)
            errors.append([index, error, kind])

    raise ValueError("the item's turn is an empty row; it has no template to build from")


class _Sink(io.TextIOBase):
    """Where executed code's stdout and stderr go: nothing is kept."""

    def write(self, text: str) -> int:
        """Discard text, so a flood costs time, which the timeout bounds, not memory.

        Args:
            text: What the code printed.

        Returns:
            The length of text, as a file would report it.
        """
        return len(text)

    def writable(self) -> bool:
        """Accept writes.

        Args:
            None.

        Returns:
            Always True.
        """
        return True


def execute_turn(tools: Tools, request: dict[str, object], send: object) -> None:
    """Replay the item's prefix, run gold, then run the generated code.

    Two replies: one when the teacher-forced state is ready, so the parent
    can start the timeout clock, and one with the generated code's outcome.

    Args:
        tools: The imported tools.
        request: The replay plans, the gold plan and the generated code.
        send: Callable delivering one reply to the parent.
    """
    assert callable(send)
    replay = request["replay"]
    gold = request["gold"]
    code = request["code"]
    assert isinstance(replay, list) and isinstance(code, str)
    tools.reset()
    session: dict[str, object] = {}

    for plan in replay:
        error, kind = tools.run(str(plan), session)

        # Not strict: go on past a raising gold plan, as upstream's first pass
        # does. Only the parity tests score turns whose gold can raise.
        if error != SUCCESS and request.get("strict", True):
            send({"ok": False, "error": f"gold replay raised {kind}: {error}"})
            return

    before = tools.summary()
    # One call, one memo: a variable and the cache entry it was saved to stay
    # the same object in the copy, as they are in the original.
    snapshot = copy.deepcopy((session, tools.cache()))
    gold_error, gold_kind = tools.run(gold, session) if isinstance(gold, str) else ["", ""]
    send(
        {
            "ok": True,
            "before": str(before),
            "gold_error": gold_error,
            "gold_error_type": gold_kind,
            "gold_after": tools.summary(),
        }
    )
    scope, cache = snapshot
    tools.set_cache(cache)
    error, kind = SUCCESS, ""

    if code:
        sink = _Sink()

        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            # BaseException: sys.exit() or a raised KeyboardInterrupt is the
            # code's outcome, not the worker's end.
            error, kind = tools.run(code, scope, (BaseException,))

    # Before the summary is read: it runs on the same builtins.
    restored = tools.restore()

    try:
        after: dict[str, object] | None = tools.summary()
    except Exception:
        # Code that wrote something other than a result into the cache.
        after = None

    send({"ok": True, "error": error, "error_type": kind, "after": after, "restored": restored})


def _isolate() -> None:
    """Cut the worker off from the terminal: no stdin, stdout and stderr to null.

    Args:
        None.
    """
    # Ctrl-C reaches the whole process group; the parent handles it and
    # stops the worker, so it must not land in an item's exec instead.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    null = os.open(os.devnull, os.O_RDWR)

    for fd in (1, 2):
        os.dup2(null, fd)

    with contextlib.suppress(OSError):
        os.close(0)

    # input() raises RuntimeError("input(): lost sys.stdin").
    sys.stdin = None  # type: ignore[assignment]


def serve(conn: Connection, root: str, package: str = VENDORED) -> None:
    """The worker's main loop: import the tools once, then answer requests.

    Args:
        conn: This end of the pipe to the parent.
        root: Dataset snapshot directory holding ontology/.
        package: Tool package to import.
    """
    _isolate()

    try:
        tools = Tools(root, package)
    except BaseException:
        conn.send({"ok": False, "error": traceback.format_exc()})
        return

    conn.send({"ok": True, "pid": os.getpid()})

    while True:
        try:
            request = conn.recv()
        except (EOFError, OSError):
            return

        try:
            if request["op"] == "build":
                conn.send({"ok": True, **build_turn(tools, request["rows"])})
            elif request["op"] == "execute":
                execute_turn(tools, request, conn.send)
            else:
                conn.send({"ok": False, "error": f"unknown op {request['op']!r}"})
        except Exception:
            conn.send({"ok": False, "error": traceback.format_exc()})


# What _receive returns when the worker's end of the pipe has closed.
_GONE: dict[str, object] = {"ok": False, "error": "the worker's pipe closed"}


class Executor:
    """The parent's handle on the worker: started on first use, restarted after a kill."""

    def __init__(self, root: str, package: str = VENDORED) -> None:
        """Remember where the ontology is; nothing starts yet.

        Args:
            root: Dataset snapshot directory holding ontology/.
            package: Tool package the worker imports.
        """
        self.root = root
        self.package = package
        self.starts = 0
        self._process: multiprocessing.process.BaseProcess | None = None
        self._conn: Connection | None = None

    @property
    def pid(self) -> int | None:
        """Process id of the running worker.

        Args:
            None.

        Returns:
            The pid, or None when no worker is running.
        """
        return self._process.pid if self._process is not None else None

    def build(self, rows: list[Row]) -> Built:
        """Build one item in the worker; see build_turn.

        Args:
            rows: The dialogue's rows up to and including the item's turn.

        Returns:
            build_turn's result.
        """
        reply = self._request({"op": "build", "rows": rows}, REPLAY_TIMEOUT_S)

        if reply is None:
            raise RuntimeError(f"the T1 worker died or stalled building an item ({self._exit()})")

        return Built(**{field.name: reply[field.name] for field in fields(Built)})  # type: ignore[arg-type]

    def execute(
        self,
        replay: list[str],
        gold: str | None,
        code: str,
        timeout_s: float,
        strict: bool = True,
    ) -> Outcome:
        """Run generated code against an item's teacher-forced state.

        Args:
            replay: Gold plans before the turn, in order.
            gold: The turn's gold plan, or None when it has none.
            code: The generated code; empty is not executed.
            timeout_s: Seconds the generated code may run.
            strict: Whether a raising gold plan in replay is an error.

        Returns:
            Gold's outcome at the turn and the generated code's.
        """
        request: dict[str, object] = {
            "op": "execute",
            "replay": replay,
            "gold": gold,
            "code": code,
            "strict": strict,
        }
        ready = self._request(request, REPLAY_TIMEOUT_S)

        if ready is None:
            # Nothing generated has run yet, so a dead worker here is not the
            # item's doing: try once more on a fresh one.
            ready = self._request(request, REPLAY_TIMEOUT_S)

        if ready is None:
            raise RuntimeError(f"the T1 worker died or stalled replaying gold ({self._exit()})")

        outcome = self._receive(timeout_s)

        if outcome is None or outcome is _GONE:
            # The pipe says which: a reply that never came is a timeout, a
            # closed pipe is a worker that died, whatever is_alive says in
            # the moment before it is reaped.
            timed_out = outcome is None
            error = TIMEOUT if timed_out else f"worker died ({self._exit()})"
            self.close()
            outcome = {"error": error, "error_type": "", "after": None, "timeout": timed_out}
        elif not outcome.get("ok"):
            raise RuntimeError(f"the T1 worker failed: {outcome.get('error')}")

        error = str(outcome["error"])

        return Outcome(
            before=str(ready["before"]),
            gold_error=str(ready["gold_error"]),
            gold_error_type=str(ready["gold_error_type"]),
            gold_after=ready["gold_after"],  # type: ignore[arg-type]
            error=error,
            error_type=str(outcome["error_type"]),
            after=outcome["after"],  # type: ignore[arg-type]
            timeout=bool(outcome.get("timeout", False)),
            executed=error != TIMEOUT and not error.startswith("worker died"),
            restored=list(outcome.get("restored", [])),  # type: ignore[call-overload]
        )

    def close(self) -> None:
        """Stop the worker, killing it if it does not leave on its own.

        Args:
            None.
        """
        if self._conn is not None:
            self._conn.close()

        if self._process is not None:
            self._process.kill()
            self._process.join(timeout=30)

        self._process = None
        self._conn = None

    def _ensure(self) -> Connection:
        """Start the worker when none is running and wait until it has the tools.

        Args:
            None.

        Returns:
            The parent's end of the pipe.
        """
        if self._conn is not None and self._process is not None and self._process.is_alive():
            return self._conn

        self.close()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=serve, args=(child, self.root, self.package), daemon=True, name="t1-exec"
        )
        process.start()
        child.close()
        self._process, self._conn = process, parent
        self.starts += 1
        reply = self._receive(STARTUP_TIMEOUT_S)

        if reply is None or reply is _GONE or not reply.get("ok"):
            detail = self._exit() if reply is None or reply is _GONE else reply.get("error")
            self.close()
            raise RuntimeError(
                f"the T1 worker could not import the tools from {self.root}: {detail}. "
                "The ontology variables point below that directory; check that its "
                "ontology/ holds the dataset's CSVs."
            )

        return parent

    def _request(self, request: dict[str, object], timeout_s: float) -> dict[str, object] | None:
        """Send one request and wait for its first reply.

        Args:
            request: The message to send.
            timeout_s: Seconds to wait for the reply.

        Returns:
            The reply, or None when the worker died or stalled (it is then stopped).
        """
        conn = self._ensure()

        try:
            conn.send(request)
        except (BrokenPipeError, OSError):
            self.close()
            return None

        reply = self._receive(timeout_s)

        if reply is None or reply is _GONE:
            self.close()
            return None

        if not reply.get("ok"):
            raise RuntimeError(f"the T1 worker failed: {reply.get('error')}")

        return reply

    def _receive(self, timeout_s: float) -> dict[str, object] | None:
        """Wait for one reply.

        Args:
            timeout_s: Seconds to wait.

        Returns:
            The reply, None on timeout, or _GONE when the pipe closed.
        """
        assert self._conn is not None

        try:
            if not self._conn.poll(timeout_s):
                return None
            reply: dict[str, object] = self._conn.recv()
        except (EOFError, OSError):
            return _GONE

        return reply

    def _exit(self) -> str:
        """Describe how the worker ended, for an error message.

        Args:
            None.

        Returns:
            Its exit code, or that it is still running.
        """
        if self._process is None:
            return "no worker"

        self._process.join(timeout=1)
        code = self._process.exitcode

        return "still running" if code is None else f"exit code {code}"


def run_prefixes(path: str) -> None:
    """Replay dialogue prefixes from a JSON request file, for the drift check.

    The request names the ontology root, the tool package, and the prefixes;
    the result, written next to it with .out appended, holds each prefix's
    session replay records.

    Args:
        path: Path of the JSON request file.
    """
    request = json.loads(Path(path).read_text())
    tools = Tools(request["root"], request["package"])
    out = {}

    with contextlib.redirect_stdout(_Sink()):
        for key, rows in request["prefixes"].items():
            records = replay_dialogue(tools, rows, {})
            out[key] = [
                None
                if record is None
                else {name: record[name] for name in ("before", "after", "error")}
                for record in records
            ]

    Path(path + ".out").write_text(json.dumps(out))


if __name__ == "__main__":
    # Run by path, this file's directory leads sys.path, and the adapter
    # module t1.py in it would shadow upstream's t1 package.
    here = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != here]
    run_prefixes(sys.argv[1])
