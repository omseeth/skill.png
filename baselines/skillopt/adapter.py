"""SkillOpt's environment adapter over any registered skillpng benchmark.

SkillOpt's trainer reaches the target only through an EnvAdapter. This one
hands every rollout to the benchmark's own rollout, with the skill composed
into the system turn exactly as eval.condition=skillopt composes it. The
messages, decoding and scoring are therefore the ones skillpng eval measures.

Train rollouts leave upstream's trajectory files behind for reflection:
predictions/<id>/conversation.json (the answer, then an evaluation block that
shows the gold target) beside the target's system and user prompts. Val
rollouts leave scores only, so no val trajectory can reach the optimizer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from skillpng.benchmarks.base import Benchmark, Item, Result
from skillpng.benchmarks.livemathematicianbench import livemathematicianbench as livemath
from skillpng.benchmarks.registry import build_benchmark
from skillpng.benchmarks.searchqa import searchqa
from skillpng.benchmarks.t1.t1 import T1
from skillpng.benchmarks.t1.t1_build import ontology_root
from skillpng.benchmarks.t1.t1_exec import Built, Executor, Outcome, Row
from skillpng.benchmarks.trajectbench import traject
from skillpng.config import Config
from skillpng.evaluate import compose_system_prompt
from skillpng.models.base import VLM

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
# Upstream SkillOpt is vendored, not installed; see upstream/VENDORED.md.
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from skillopt.datasets.base import _SPLIT_ALIAS, BatchSpec, SplitDataLoader  # noqa: E402
from skillopt.envs.base import EnvAdapter  # noqa: E402

# Benchmarks with analyst prompts of their own upstream. The rest fall back
# to upstream's generic prompts.
ENV_PROMPTS = {
    "searchqa": "searchqa",
    "livemathematicianbench": "livemathematicianbench",
}
# The task type upstream's SearchQA env reports.
SEARCHQA_TASK_TYPE = "qa"


@dataclass
class Scored:
    """What the benchmark measured for one rollout, in upstream's terms."""

    hard: float
    soft: float
    predicted: str
    fail_reason: str
    evaluation: str


def task_type(benchmark: Benchmark, item: Item) -> str:
    """The task type the analyst sees for an item.

    Args:
        benchmark: Benchmark the item belongs to.
        item: Item to describe.

    Returns:
        The item's domain or theorem type, else a name for the benchmark.
    """
    if isinstance(benchmark, searchqa.SearchQA):
        return SEARCHQA_TASK_TYPE

    for key in ("domain", "theorem_type"):
        value = item.meta.get(key)

        if isinstance(value, list):
            value = value[0] if value else None

        if value:
            return str(value)

    return benchmark.cfg.name


def item_record(benchmark: Benchmark, item: Item) -> dict[str, object]:
    """The JSON record SkillOpt's data loader samples batches of.

    Args:
        benchmark: Benchmark the item belongs to.
        item: Item to describe.

    Returns:
        The item's id, question and task type.
    """
    return {
        "id": item.id,
        "question": str(item.meta.get("question", "")),
        "task_type": task_type(benchmark, item),
    }


def user_text(messages: list[dict[str, object]]) -> str:
    """The text of the last user turn.

    Args:
        messages: Chat messages of one item.

    Returns:
        The concatenated text blocks of the last user message.
    """
    content = messages[-1].get("content")

    if isinstance(content, str):
        return content

    blocks = content if isinstance(content, list) else []

    return "".join(str(block.get("text", "")) for block in blocks if isinstance(block, dict))


def score(benchmark: Benchmark, item: Item, result: Result) -> Scored:
    """Map one result onto hard, soft and the analyst's view of it.

    hard is always the benchmark's own score, so the gate selects on the
    reported metric. soft is a partial-credit metric the benchmark already
    computes, or hard where it has none.

    Args:
        benchmark: Benchmark that produced the result.
        item: Item the result belongs to.
        result: The benchmark's rollout result.

    Returns:
        The scores, the extracted answer, the failure reason and the
        evaluation block shown to the analyst.
    """
    question = str(item.meta.get("question", ""))
    hard = float(result.score)
    meta = result.meta

    if isinstance(benchmark, searchqa.SearchQA):
        predicted = searchqa.extract_answer(result.output)
        golds = searchqa._golds(item.ground_truth)
        soft = max(searchqa.token_f1(predicted, gold) for gold in golds)
        em = max(searchqa.exact_match(predicted, gold) for gold in golds)
        name = "EM" if benchmark.opts.metric == "em" else "F1"
        reason = f"{name}={hard:g}: predicted '{predicted}' but expected {golds}"
        evaluation = (
            f"[EVALUATION RESULT]\n"
            f"Question: {question}\n"
            f"Predicted answer: {predicted!r}\n"
            f"Gold answers: {golds!r}\n"
            f"Exact Match: {em}\n"
            f"F1: {soft:.4f}"
        )

        return Scored(hard, soft, predicted, "" if hard else reason, evaluation)

    if isinstance(benchmark, livemath.LiveMathematicianBench):
        label = livemath.extract_label(result.output)
        gold = str(item.ground_truth)
        options = item.meta.get("options")
        texts = {
            str(option["label"]): str(option["text"])
            for option in (options if isinstance(options, list) else [])
        }
        reason = f"MCQ=0: predicted '{label or ''}' but expected '{gold}'"
        evaluation = (
            f"[EVALUATION RESULT]\n"
            f"Question: {question}\n"
            f"Predicted label: {label!r}\n"
            f"Predicted text: {texts.get(label or '', '')!r}\n"
            f"Correct label: {gold!r}\n"
            f"Correct text: {texts.get(gold, '')!r}\n"
            f"Exact Match: {hard}"
        )

        return Scored(hard, hard, label or "", "" if hard else reason, evaluation)

    if isinstance(benchmark, T1):
        soft = float(meta["tool_f1"])
        code = str(meta["code"])
        measured = (
            f"error={meta['error']!r}, error_type={meta['error_type']!r}, "
            f"timeout={meta['timeout']}, executed={meta['executed']}, "
            f"code_success={meta['code_success']}, tool_f1={soft:.4f}, "
            f"tool tp/fp/fn={meta['tool_tp']}/{meta['tool_fp']}/{meta['tool_fn']}, "
            f"param tp/fp/fn={meta['param_tp']}/{meta['param_fp']}/{meta['param_fn']}, "
            f"gold calls={meta['n_gold_calls']}, generated calls={meta['n_gen_calls']}, "
            f"parse_error={meta['parse_error']!r}, gold_kind={meta['gold_kind']}"
        )
        evaluation = (
            f"[EVALUATION RESULT]\n"
            f"Question: {question}\n"
            f"Generated code:\n{code or '(none)'}\n"
            f"Gold code:\n{item.ground_truth}\n"
            f"{benchmark.opts.metric}: {hard}\n"
            f"{measured}"
        )
        reason = f"{benchmark.opts.metric}=0: {measured}"

        return Scored(hard, soft, code, "" if hard else reason, evaluation)

    if isinstance(benchmark, traject.TrajectBench):
        calls = traject.extract_calls(result.output)
        predicted = traject.serialize_calls(calls) if calls else ""
        gold_calls = item.ground_truth if isinstance(item.ground_truth, list) else []
        measured = ", ".join(
            f"{key}={meta[key]}"
            for key in (
                "parse_success",
                "tool_em",
                "tool_inclusion",
                "required_em",
                "optional_em",
                "usage_em",
                "n_pred",
                "n_gold",
            )
        )
        evaluation = (
            f"[EVALUATION RESULT]\n"
            f"Question: {question}\n"
            f"Predicted calls: {predicted or '(none parsed)'}\n"
            f"Gold calls: {traject.serialize_calls(gold_calls)}\n"
            f"{benchmark.opts.metric}: {hard}\n"
            f"{measured}"
        )
        reason = f"{benchmark.opts.metric}=0: {measured}"
        soft = float(meta["tool_inclusion"])

        return Scored(hard, soft, predicted, "" if hard else reason, evaluation)

    reason = f"{benchmark.metric_name}={hard:g}: expected {item.target!r}"
    evaluation = (
        f"[EVALUATION RESULT]\n"
        f"Question: {question}\n"
        f"Gold: {item.target}\n"
        f"{benchmark.metric_name}: {hard}"
    )

    return Scored(hard, hard, result.output, "" if hard else reason, evaluation)


class BenchmarkLoader(SplitDataLoader):
    """Upstream's split loader, fed with a benchmark's train and val items.

    Epoch planning and batch seeding stay upstream's. The test split is never
    handed over, and asking for it is an error rather than upstream's silent
    fallback to val.
    """

    def __init__(self, splits: dict[str, list[dict[str, object]]], seed: int):
        """Keep the item records until the trainer calls setup.

        Args:
            splits: Split name, train or val, to its item records.
            seed: Seed upstream's loader records.
        """
        super().__init__(split_mode="split_dir", seed=seed)
        self._given = splits

    def setup(self, cfg: dict[str, object]):
        """Install the given splits in place of upstream's split directory.

        Args:
            cfg: The trainer's flat config, unused.
        """
        self._splits = {name: list(items) for name, items in self._given.items()}

    def get_split_items(self, split: str) -> list[dict[str, object]]:
        """Resolve a split name, refusing any split that was not given.

        Args:
            split: Split name, including upstream's aliases.

        Returns:
            The split's item records.
        """
        name = _SPLIT_ALIAS.get(split, split)

        if name not in self._splits:
            raise ValueError(
                f"split {split!r} ({name}) never reaches SkillOpt; only "
                f"{sorted(self._splits)} are loaded"
            )

        return list(self._splits[name])


class ExecutorPool(Executor):
    """T1's code executor, shared safely between rollout threads.

    One Executor is one worker process behind one pipe, so two threads must
    never use it at once. Each call borrows an idle member and returns it.
    """

    def __init__(self, members: list[Executor]):
        """Hold the member executors; none of their workers starts yet.

        Args:
            members: The executors to lend out, at least one.
        """
        if not members:
            raise ValueError("an executor pool needs at least one member")

        super().__init__(members[0].root, members[0].package)
        self.members = members
        self._idle: queue.Queue[Executor] = queue.Queue()

        for member in members:
            self._idle.put(member)

    def build(self, rows: list[Row]) -> Built:
        """Build one item on an idle member.

        Args:
            rows: The dialogue's rows up to and including the item's turn.

        Returns:
            The member's build result.
        """
        member = self._idle.get()

        try:
            return member.build(rows)
        finally:
            self._idle.put(member)

    def execute(
        self,
        replay: list[str],
        gold: str | None,
        code: str,
        timeout_s: float,
        strict: bool = True,
    ) -> Outcome:
        """Run generated code on an idle member.

        Args:
            replay: Gold plans before the turn, in order.
            gold: The turn's gold plan, or None.
            code: The generated code.
            timeout_s: Seconds the generated code may run.
            strict: Whether a raising gold plan in replay is an error.

        Returns:
            The member's outcome.
        """
        member = self._idle.get()

        try:
            return member.execute(replay, gold, code, timeout_s, strict)
        finally:
            self._idle.put(member)

    def close(self):
        """Stop every member's worker."""
        for member in self.members:
            member.close()


class BenchmarkAdapter(EnvAdapter):
    """SkillOpt's EnvAdapter for any registered skillpng benchmark."""

    def __init__(
        self,
        config: Config,
        benchmark: Benchmark,
        train: list[Item],
        val: list[Item],
        *,
        workers: int,
        exec_workers: int,
        analyst_workers: int,
        failure_only: bool,
        minibatch_size: int,
        edit_budget: int,
        seed: int,
        model: VLM | None = None,
    ):
        """Index the items and build upstream's loader over them.

        Args:
            config: The resolved target config.
            benchmark: Benchmark built from config.benchmark.
            train: Train items handed to SkillOpt.
            val: Val items the gate scores on.
            workers: Rollouts run at once.
            exec_workers: T1 code executors run at once.
            analyst_workers: Upstream's parallel optimizer calls.
            failure_only: Upstream's reflect-on-failures-only switch.
            minibatch_size: Trajectories per analyst call.
            edit_budget: Upstream's per-analyst edit budget.
            seed: Seed upstream's loader records.
            model: The target model; may be set later.
        """
        self.config = config
        self.benchmark = benchmark
        self.model = model
        self.workers = workers
        self.exec_workers = exec_workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.items = {item.id: item for item in [*train, *val]}
        self.val_ids = {item.id for item in val}
        self.dataloader = BenchmarkLoader(
            {
                "train": [item_record(benchmark, item) for item in train],
                "val": [item_record(benchmark, item) for item in val],
            },
            seed,
        )
        self._executors: ExecutorPool | None = None

    @property
    def _env_name(self) -> str:
        """The upstream env whose analyst prompts this benchmark uses.

        Returns:
            The env directory name, or empty for the generic prompts.
        """
        return ENV_PROMPTS.get(self.benchmark.cfg.name, "")

    def setup(self, cfg: dict[str, object]):
        """Keep the trainer's config and set up the loader.

        Args:
            cfg: The trainer's flat config.
        """
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self) -> BenchmarkLoader:
        """The loader the trainer plans epochs with.

        Returns:
            The benchmark loader.
        """
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs: object) -> list[dict[str, object]]:
        """A batch's item records are its environment.

        Args:
            batch: The batch upstream planned.
            **kwargs: Upstream's extra arguments, unused.

        Returns:
            The batch's item records.
        """
        payload = batch.payload
        return list(payload) if isinstance(payload, list) else []

    def build_train_env(
        self, batch_size: int, seed: int, **kwargs: object
    ) -> list[dict[str, object]]:
        """Sample a train batch with upstream's loader.

        Args:
            batch_size: Items in the batch.
            seed: Seed of the draw.
            **kwargs: Upstream's extra arguments.

        Returns:
            The batch's item records.
        """
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch)

    def build_eval_env(
        self, env_num: int, split: str, seed: int, **kwargs: object
    ) -> list[dict[str, object]]:
        """Take an eval batch with upstream's loader.

        Args:
            env_num: Items to take, 0 for the whole split.
            split: Split name, including upstream's aliases.
            seed: Seed upstream records.
            **kwargs: Upstream's extra arguments.

        Returns:
            The batch's item records.
        """
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch)

    def get_task_types(self) -> list[str]:
        """Every task type among the loaded items.

        Returns:
            Task types in order of first appearance.
        """
        seen: list[str] = []

        for item in self.items.values():
            kind = task_type(self.benchmark, item)

            if kind not in seen:
                seen.append(kind)

        return seen

    def benchmark_for(self, skill: str) -> Benchmark:
        """The benchmark with the skill in its system turn, as run_eval builds it.

        Args:
            skill: The skill text, already stripped.

        Returns:
            A benchmark built from the composed system prompt.
        """
        prompt = compose_system_prompt(
            self.config.benchmark.system_prompt, self.config.injection.skill_template, skill
        )
        benchmark = build_benchmark(
            self.config.benchmark.model_copy(update={"system_prompt": prompt})
        )

        if isinstance(benchmark, T1):
            # One worker per T1 instance would restart the ontology import
            # for every skill; the pool is shared and safe across threads.
            benchmark._executor = self.executors()

        return benchmark

    def executors(self) -> ExecutorPool:
        """The shared T1 executor pool, created on first use.

        Returns:
            The pool.
        """
        if self._executors is None:
            assert isinstance(self.benchmark, T1)
            root = str(ontology_root(self.benchmark.repo_id))
            self._executors = ExecutorPool([Executor(root) for _ in range(self.exec_workers)])

        return self._executors

    def close(self):
        """Stop the T1 executors, if any were started."""
        if self._executors is not None:
            self._executors.close()

    def rollout(
        self,
        env_manager: list[dict[str, object]],
        skill_content: str,
        out_dir: str,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        """Roll out a batch with the benchmark's own rollout and score it.

        The skill is stripped as condition_skill strips a skill file, so a
        rollout sends the bytes skillpng eval sends for the same skill. Any
        rollout error aborts the run: a failed request scored as a wrong
        answer would mislead the gate.

        Args:
            env_manager: The batch's item records.
            skill_content: The skill to roll out with.
            out_dir: Directory the batch's files go to.
            **kwargs: Upstream's extra arguments, unused.

        Returns:
            One upstream result dict per item, in batch order.
        """
        if self.model is None:
            raise RuntimeError("the adapter has no target model to roll out with")

        model = self.model
        items = [self.items[str(record["id"])] for record in env_manager]
        splits = {"val" if item.id in self.val_ids else "train" for item in items}

        if len(splits) > 1:
            raise ValueError("a rollout batch mixes train and val items")

        split = splits.pop() if splits else "train"
        benchmark = self.benchmark_for(skill_content.strip())

        def one(item: Item) -> Result:
            """Roll out one item in this thread's own event loop.

            Args:
                item: Item to roll out.

            Returns:
                The benchmark's result.
            """
            return asyncio.run(benchmark.rollout(item, model, None))

        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            results = list(pool.map(one, items))

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        lines: list[str] = []
        hards: list[float] = []
        softs: list[float] = []

        for item, result in zip(items, results, strict=True):
            scored = score(benchmark, item, result)
            hards.append(scored.hard)
            softs.append(scored.soft)
            lines.append(
                json.dumps(
                    {
                        "id": item.id,
                        "split": split,
                        "hard": scored.hard,
                        "soft": scored.soft,
                        "output": result.output,
                        "meta": result.meta,
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )

            if split == "val":
                rows.append({"id": item.id, "hard": scored.hard, "soft": scored.soft})
                continue

            rows.append(self.trajectory(benchmark, item, result, scored, out))

        (out / "rollout.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        hard = sum(hards) / max(len(hards), 1)
        soft = sum(softs) / max(len(softs), 1)
        summary = {
            "split": split,
            "n": len(rows),
            "hard": hard,
            "soft": soft,
            "skill_sha256": hashlib.sha256(skill_content.encode()).hexdigest(),
        }
        (out / "rollout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"    [rollout] {len(rows)} {split} items: hard={hard:.4f} soft={soft:.4f}", flush=True
        )

        return rows

    def trajectory(
        self, benchmark: Benchmark, item: Item, result: Result, scored: Scored, out: Path
    ) -> dict[str, object]:
        """Write one train item's trajectory files and build its result dict.

        Args:
            benchmark: Benchmark the item was rolled out on.
            item: The train item.
            result: The benchmark's result.
            scored: The result in upstream's terms.
            out: The batch's directory.

        Returns:
            The upstream result dict reflection reads.
        """
        messages = benchmark.build_messages(item, with_image=False)
        question = str(item.meta.get("question", ""))
        directory = out / "predictions" / item.id
        directory.mkdir(parents=True, exist_ok=True)
        conversation = [
            {"type": "message", "turn": 1, "content": result.output},
            {"role": "system", "content": scored.evaluation},
        ]
        (directory / "target_system_prompt.txt").write_text(
            benchmark.cfg.system_prompt, encoding="utf-8"
        )
        (directory / "target_user_prompt.txt").write_text(user_text(messages), encoding="utf-8")
        (directory / "conversation.json").write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {
            "id": item.id,
            "hard": scored.hard,
            "soft": scored.soft,
            "task_description": question,
            "question": question,
            "task_type": task_type(benchmark, item),
            "predicted_answer": scored.predicted,
            "fail_reason": scored.fail_reason,
            "n_turns": 1,
        }


def refuse_upstream_target(*args: object, **kwargs: object) -> object:
    """Stand in for SkillOpt's own target backend, which must never run.

    Args:
        *args: Whatever the caller passed.
        **kwargs: Whatever the caller passed.

    Returns:
        Never returns.
    """
    raise RuntimeError(
        "SkillOpt's own target backend was called. The target here is skillpng's "
        "model client, reached only through BenchmarkAdapter.rollout."
    )


def guard_upstream_target():
    """Make every path into SkillOpt's own target backend raise."""
    import skillopt.model as api
    from skillopt.model import azure_openai

    for name in ("chat_target", "chat_target_messages"):
        setattr(api, name, refuse_upstream_target)

    for name in ("get_target_client", "chat_target", "chat_target_messages"):
        setattr(azure_openai, name, refuse_upstream_target)
