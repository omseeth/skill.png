"""SkillOpt: one skill document trained for one benchmark and one target model.

Upstream SkillOpt (vendored in upstream/, see upstream/VENDORED.md) runs its
optimizer loop unchanged: rollouts, reflection, merging, ranking under the
textual learning rate, the update, the val gate, and at epoch boundaries the
slow update and the meta skill. Everything on the target side is this
repository's: the benchmark's prompts, splits and scoring, the vLLM client,
the decoding of the non-reasoning test evals, and thinking off.

    uv run --extra skillopt python baselines/skillopt/train.py --benchmark searchqa \\
        --set model.model_id=/path/to/qwen3.5-9b \\
        --set train.run_dir=runs/searchqa/qwen3.5-9B

Artifacts go to <train.run_dir>/<train.run_name>/. The selected skill is
frozen in skills/<benchmark>/<model>/<name>/ with skill.meta.json, and an
existing directory there is never overwritten. A run killed mid-way continues
from its last completed step with --resume.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from skillpng.benchmarks.base import Benchmark, Item
from skillpng.benchmarks.registry import build_benchmark
from skillpng.config import Config, load_config
from skillpng.models.base import VLM
from skillpng.models.registry import build_model
from skillpng.results import git_sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# llmskill's generator holds the config lookup, the key handling and the
# generation stats lookup this script reuses.
sys.path.append(str(ROOT / "baselines" / "llmskill"))

import anthropic_api  # noqa: E402
from adapter import UPSTREAM, BenchmarkAdapter, guard_upstream_target  # noqa: E402
from generate import STATS_KEYS, api_key, generation_stats, resolve_config  # noqa: E402
from openrouter import API, CallLog, install, preflight  # noqa: E402

CONFIGS = ROOT / "configs"
SKILLS = HERE / "skills"
# One blank skill per benchmark, named benchmark.name; searchqa.md is
# upstream's blank SearchQA skill, the others its placeholder retitled.
SEEDS = HERE / "seeds"
UPSTREAM_COMMIT = "32deaeb50f92fc2f3a225bbf67d3eb3a604ce8b6"
UPSTREAM_DEFAULTS = UPSTREAM / "configs" / "_base_" / "default.yaml"

DEFAULT_OPTIMIZER = "openai/gpt-5.5"
DEFAULT_EFFORT = "medium"
# Where each optimizer provider's key is read from.
KEY_VARIABLES = {"openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# The decoding of the non-reasoning test evals, so a skill is trained under
# the sampling it is scored with. Applied before --set, which may override.
EVAL_DECODING = (
    "model.name=vllm",
    "model.generate.do_sample=true",
    "model.generate.temperature=0.7",
    "model.generate.seed=42",
    "seed=42",
)
# Skill.png images are optimized with thinking off; so is every skill here.
THINKING_OFF = "model.chat_template_kwargs.enable_thinking=false"
# Flat SkillOpt keys that name paths or endpoints, not the recipe.
LOCATION_KEYS = ("out_root", "skill_init", "optimizer_azure_openai_endpoint")
# The recipe as recorded in skill.meta.json.
HYPERPARAMETERS = (
    "optimizer_model",
    "reasoning_effort",
    "num_epochs",
    "train_size",
    "steps_per_epoch",
    "batch_size",
    "accumulation",
    "seed",
    "minibatch_size",
    "merge_batch_size",
    "analyst_workers",
    "failure_only",
    "edit_budget",
    "min_edit_budget",
    "lr_scheduler",
    "lr_control_mode",
    "skill_update_mode",
    "use_slow_update",
    "slow_update_samples",
    "slow_update_gate_with_selection",
    "longitudinal_pair_policy",
    "use_meta_skill",
    "use_skill_aware_reflection",
    "use_gate",
    "sel_env_num",
    "eval_test",
)


@dataclass
class Options:
    """Everything about a run that is not the target config."""

    optimizer: str = DEFAULT_OPTIMIZER
    effort: str = DEFAULT_EFFORT
    seed: int = 42
    limit: int | None = None
    workers: int = 32
    exec_workers: int = 4
    init_skill: Path | None = None
    base_url: str = API
    provider: str = "openrouter"
    resume: bool = False
    dry_run: bool = False
    # Flat SkillOpt keys set on top of upstream's defaults. The command line
    # sets none; tests shrink the recipe with it.
    skillopt: dict[str, object] = field(default_factory=dict)

    def name(self) -> str:
        """The frozen skill's directory name.

        Returns:
            E.g. gpt-5.5_medium_nothink_seed42, plus _init<sha8> when the
            run starts from a given skill.
        """
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", self.optimizer.rsplit("/", 1)[-1]).strip("-")
        name = f"{slug}_{self.effort}_nothink_seed{self.seed}"

        if self.init_skill is not None:
            name += f"_init{_sha256(self.init_skill.read_bytes())[:8]}"

        return name

    def run_name(self) -> str:
        """The default train.run_name.

        Returns:
            skillopt_<name>, plus _limit<N> for a capped run.
        """
        suffix = f"_limit{self.limit}" if self.limit else ""
        return f"skillopt_{self.name()}{suffix}"


def _sha256(data: bytes) -> str:
    """Hex sha256 of some bytes.

    Args:
        data: Bytes to hash.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    """The current UTC time.

    Returns:
        An ISO 8601 timestamp.
    """
    return datetime.now(UTC).isoformat()


def benchmark_names() -> list[str]:
    """Every benchmark.name an experiment config under configs/ sets.

    Returns:
        The names, sorted.
    """
    names = {load_config(path).benchmark.name for path in CONFIGS.glob("*.yaml")}
    return sorted(name for name in names if name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments, or None for sys.argv.

    Returns:
        The parsed flags.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark", required=True, choices=benchmark_names())
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override of the benchmark's config, as skillpng eval takes it",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=tuple(KEY_VARIABLES),
        help="where the optimizer is reached; anthropic reads ANTHROPIC_API_KEY",
    )
    parser.add_argument(
        "--optimizer", default=DEFAULT_OPTIMIZER, help="the optimizer's model id at the provider"
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_EFFORT,
        choices=("minimal", "low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--seed", type=int, default=42, help="SkillOpt's train.seed")
    parser.add_argument("--limit", type=int, help="cap the train and val items, for smoke runs")
    parser.add_argument("--workers", type=int, default=32, help="target rollouts run at once")
    parser.add_argument(
        "--exec-workers", type=int, default=4, help="T1 code executor processes run at once"
    )
    parser.add_argument("--init-skill", type=Path, help="start from this skill, not the blank one")
    parser.add_argument(
        "--api-key-file", type=Path, help="read the key here when the provider's variable is unset"
    )
    parser.add_argument("--openrouter-url", default=API, help="OpenRouter's API base URL")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the first train item's messages and the SkillOpt config, call nothing",
    )
    parser.add_argument(
        "--probe-optimizer",
        action="store_true",
        help="send one tiny optimizer request through SkillOpt's route and print its stats",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.workers < 1 or args.exec_workers < 1:
        parser.error("--workers and --exec-workers must be at least 1")

    if args.init_skill is not None and not args.init_skill.is_file():
        parser.error(f"--init-skill {args.init_skill} does not exist")

    if args.provider != "openrouter" and args.openrouter_url != API:
        parser.error("--openrouter-url applies to --provider openrouter only")

    return args


def resolve_target(path: Path, overrides: list[str]) -> Config:
    """The target config: the eval decoding, the overrides, thinking off.

    Args:
        path: The benchmark's experiment config.
        overrides: Dotted key=value overrides, applied after the decoding.

    Returns:
        The validated config with enable_thinking false.
    """
    cfg = load_config(path, [*EVAL_DECODING, *overrides])
    thinking = cfg.model.chat_template_kwargs.get("enable_thinking", False)

    if thinking is not False:
        raise SystemExit(
            f"model.chat_template_kwargs.enable_thinking={thinking!r}: SkillOpt trains with "
            "thinking off on every model, as the Skill.png images are optimized"
        )

    if cfg.benchmark.limit is not None:
        raise SystemExit("benchmark.limit caps the held-out check too; use --limit instead")

    if cfg.model.name != "vllm":
        raise SystemExit(
            f"model.name={cfg.model.name}: rollouts run concurrently against a vLLM server"
        )

    return load_config(path, [*EVAL_DECODING, *overrides, THINKING_OFF])


def split_items(benchmark: Benchmark, limit: int | None) -> tuple[list[Item], list[Item]]:
    """The train and val items, after proving no split shares an id.

    Test ids are read for that check only; no test item reaches SkillOpt.

    Args:
        benchmark: Benchmark built from the uncapped config.
        limit: Prefix of train and val to keep, or None for all of them.

    Returns:
        The train items and the val items.
    """
    train = benchmark.items("train")
    val = benchmark.items("val")
    test = {item.id for item in benchmark.items("test")}
    train_ids = {item.id for item in train}
    val_ids = {item.id for item in val}
    leaked = sorted((train_ids & (val_ids | test)) | (val_ids & test))

    # Not an assert: the check has to survive python -O.
    if leaked:
        raise SystemExit(f"LEAK: id(s) {leaked[:5]} occur in two splits; nothing trained")

    if limit:
        train, val = train[:limit], val[:limit]

    return train, val


def seed_skill(benchmark: str, init: Path | None) -> Path:
    """The skill SkillOpt starts from.

    Args:
        benchmark: The benchmark.name.
        init: A skill given with --init-skill, or None.

    Returns:
        Path of the starting skill.
    """
    if init is not None:
        if not init.is_file():
            raise SystemExit(f"--init-skill {init} does not exist")

        return init.resolve()

    path = SEEDS / f"{benchmark}.md"

    if not path.is_file():
        raise SystemExit(f"no seed skill for {benchmark} in {SEEDS}; pass --init-skill")

    return path


def skillopt_config(
    options: Options, benchmark: str, model_id: str, seed: Path, run_path: Path
) -> dict[str, object]:
    """Upstream's default recipe with only this run's identity filled in.

    Args:
        options: The run's options.
        benchmark: The benchmark.name.
        model_id: The target checkpoint, recorded as upstream's target name.
        seed: The starting skill.
        run_path: Where upstream writes its artifacts.

    Returns:
        The flat config ReflACTTrainer takes.
    """
    from skillopt.config import flatten_config
    from skillopt.config import load_config as load_upstream

    structured = load_upstream(str(UPSTREAM_DEFAULTS))
    structured["model"].update(
        {
            "optimizer": options.optimizer,
            "target": model_id,
            "reasoning_effort": options.effort,
            "optimizer_azure_openai_endpoint": options.base_url,
            "optimizer_azure_openai_auth_mode": "openai_compatible",
        }
    )
    structured["train"]["seed"] = options.seed
    # The test split is scored by skillpng eval only, never here.
    structured["evaluation"]["eval_test"] = False
    structured["env"].update(
        {"name": benchmark, "skill_init": str(seed), "out_root": str(run_path)}
    )
    flat: dict[str, object] = flatten_config(structured)
    flat.update(options.skillopt)

    return flat


def run_record(
    cfg: Config, flat: dict[str, object], options: Options, seed: Path
) -> dict[str, object]:
    """What a resumed run must share with the run it continues.

    The vLLM section is left out: the server's address may change between
    jobs, and the client refuses a server that is not this checkpoint's.

    Args:
        cfg: The target config.
        flat: The SkillOpt config.
        options: The run's options.
        seed: The starting skill.

    Returns:
        The recorded settings and their digest.
    """
    target = cfg.model_dump(
        mode="json",
        include={"seed", "model", "benchmark", "injection"},
        exclude={"model": {"vllm", "num_gpus", "device"}},
    )
    hashed = {
        "target": target,
        "skillopt": {key: value for key, value in flat.items() if key not in LOCATION_KEYS},
        "seed_skill_sha256": _sha256(seed.read_bytes()),
        "limit": options.limit,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    digest = _sha256(json.dumps(hashed, sort_keys=True).encode())[:16]

    return {**hashed, "digest": digest}


def check_run_dir(run_path: Path, record: dict[str, object], resume: bool):
    """Refuse a new run on an old directory, or a resume with other settings.

    Args:
        run_path: The run directory.
        record: This run's record from run_record.
        resume: Whether --resume was given.
    """
    saved_path = run_path / "run.json"

    if not resume:
        if run_path.exists() and any(run_path.iterdir()):
            raise SystemExit(f"{run_path} exists; pass --resume to continue that run")
        return

    if not saved_path.is_file():
        raise SystemExit(f"--resume: {run_path} holds no run to continue")

    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    if saved.get("digest") != record["digest"]:
        changed = sorted(key for key in record if key != "digest" and saved.get(key) != record[key])
        raise SystemExit(f"--resume: {run_path} was started with other settings: {changed}")


def write_run_record(run_path: Path, record: dict[str, object], resume: bool):
    """Create or extend run.json, the run's settings and start times.

    Args:
        run_path: The run directory.
        record: This run's record from run_record.
        resume: Whether the run continues an earlier one.
    """
    path = run_path / "run.json"

    if resume:
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["resumed_utc"].append(_now())
    else:
        run_path.mkdir(parents=True, exist_ok=True)
        saved = {**record, "started_utc": _now(), "resumed_utc": []}

    path.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def warn_if_proxied(url: str):
    """Warn when a proxy variable would carry requests to the target.

    Args:
        url: The vLLM base URL.
    """
    parts = urllib.parse.urlsplit(url)
    proxies = urllib.request.getproxies()
    host = parts.hostname or ""

    if (proxies.get(parts.scheme) or proxies.get("all")) and not (
        urllib.request.proxy_bypass(host)
    ):
        print(
            f"warning: a proxy variable applies to the vLLM server at {url}; "
            f"add {host} to NO_PROXY",
            file=sys.stderr,
        )


def show(
    cfg: Config,
    adapter: BenchmarkAdapter,
    train: list[Item],
    val: list[Item],
    seed: Path,
    flat: dict[str, object],
    dest: Path,
):
    """Print what a run would send and how it is configured.

    Args:
        cfg: The target config.
        adapter: The adapter the run would use.
        train: The train items.
        val: The val items.
        seed: The starting skill.
        flat: The SkillOpt config.
        dest: Where the selected skill would land.
    """
    from skillpng.models.vllm import sampling_body

    item = train[0]
    benchmark = adapter.benchmark_for(seed.read_text(encoding="utf-8").strip())
    messages = benchmark.build_messages(item, with_image=False)
    decoding = sampling_body({"max_new_tokens": cfg.model.max_new_tokens, **cfg.model.generate})

    print("===== target =====")
    print(f"benchmark: {cfg.benchmark.name} ({len(train)} train, {len(val)} val items)")
    print(f"model: {cfg.model.model_id} at {cfg.model.vllm.base_url}")
    print(f"decoding: {json.dumps(decoding)}")
    print(f"chat_template_kwargs: {json.dumps(cfg.model.chat_template_kwargs)}")
    print(f"seed skill: {seed}")
    print(f"run dir: {cfg.run_path}")
    print(f"selected skill goes to: {dest}")
    print(f"\n===== messages of the first train item, {item.id} =====")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print("\n===== SkillOpt config =====")
    print(json.dumps(flat, indent=2, ensure_ascii=False))


def val_scores(run_path: Path) -> dict[str, dict[str, object]]:
    """Every val score the run measured, keyed by the skill's sha256.

    Args:
        run_path: The run directory.

    Returns:
        Skill sha256 to its val hard, soft and item count.
    """
    scores: dict[str, dict[str, object]] = {}

    for path in sorted(run_path.rglob("rollout_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))

        if summary.get("split") == "val":
            scores[str(summary["skill_sha256"])] = {
                key: summary[key] for key in ("hard", "soft", "n")
            }

    return scores


def freeze(
    cfg: Config,
    options: Options,
    run_path: Path,
    dest: Path,
    log: CallLog,
    seed: Path,
    counts: dict[str, int],
) -> Path:
    """Copy the selected skill out of the run and describe how it was made.

    Args:
        cfg: The target config.
        options: The run's options.
        run_path: The finished run's directory.
        dest: Directory the skill goes to; never overwritten when frozen.
        log: The run's optimizer call log.
        seed: The starting skill.
        counts: Train and val item counts.

    Returns:
        The directory holding skill.md and skill.meta.json.
    """
    best = (run_path / "best_skill.md").read_bytes()

    if not best.decode("utf-8").strip():
        raise SystemExit("the selected skill is empty; skillpng eval would refuse it")

    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    saved = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    # The trainer's own record, with the values it derived (steps per epoch).
    trained = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    scores = val_scores(run_path)
    seed_sha = _sha256(seed.read_bytes())
    best_sha = _sha256(best)
    text = best.decode("utf-8")
    meta: dict[str, object] = {
        "benchmark": cfg.benchmark.name,
        "model": Path(cfg.train.run_dir).name,
        "optimizer_model": options.optimizer,
        "reasoning_effort": options.effort,
        "optimizer_provider": options.provider,
        "optimizer_endpoint": options.base_url,
        "seed": options.seed,
        "seed_skill": {"path": _display(seed), "sha256": seed_sha},
        "skill_sha256": best_sha,
        "skill_chars": len(text),
        "skill_words": len(text.split()),
        "val": {"seed_skill": scores.get(seed_sha), "selected_skill": scores.get(best_sha)},
        "selection": {
            key: summary.get(key)
            for key in (
                "best_step",
                "best_origin",
                "baseline_selection_hard",
                "best_selection_hard",
            )
        },
        "steps": {
            "total": summary.get("total_steps"),
            "accepted": summary.get("total_accepts"),
            "rejected": summary.get("total_rejects"),
            "skipped": summary.get("total_skips"),
            "per_epoch": summary.get("epoch_stats"),
        },
        # gate_metric is unset in upstream's defaults, which means hard.
        "hyperparameters": {
            **{key: trained.get(key) for key in HYPERPARAMETERS},
            "gate_metric": trained.get("gate_metric", "hard"),
        },
        "target": {
            "model_id": cfg.model.model_id,
            "served_model_name": cfg.model.vllm.served_model_name or cfg.model.model_id,
            "chat_template_kwargs": cfg.model.chat_template_kwargs,
            "generate": cfg.model.generate,
            "max_new_tokens": cfg.model.max_new_tokens,
            "benchmark_options": cfg.benchmark.options,
            "system_prompt_sha256": _sha256(cfg.benchmark.system_prompt.encode()),
            "skill_template_sha256": _sha256(cfg.injection.skill_template.encode()),
            "injection_site": cfg.injection.site,
        },
        "splits": {**counts, "limit": options.limit},
        "optimizer_usage": log.totals(),
        "run_dir": _display(run_path),
        "git_sha": git_sha(),
        "upstream_commit": UPSTREAM_COMMIT,
        "config_digest": saved.get("digest"),
        "config_hash": cfg.hash(),
        "started_utc": saved.get("started_utc"),
        "resumed_utc": saved.get("resumed_utc"),
        "finished_utc": _now(),
    }
    frozen = not options.limit
    staging = dest.with_name(dest.name + ".partial")

    if frozen and dest.exists():
        raise SystemExit(f"{dest} exists; a frozen skill is never overwritten")

    staging.mkdir(parents=True, exist_ok=True)
    (staging / "skill.md").write_bytes(best)
    (staging / "skill.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not frozen and dest.exists():
        for path in dest.iterdir():
            path.unlink()
        dest.rmdir()

    # One rename, so a kill never leaves a half-written frozen skill.
    staging.rename(dest)
    print(f"wrote {_display(dest)}: {meta['skill_words']} words, val {meta['val']}")

    return dest


def _display(path: Path) -> str:
    """A path relative to the repository when it lies inside it.

    Args:
        path: Path to show.

    Returns:
        The relative path, or the path as given.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run(
    cfg: Config,
    options: Options,
    api_key: str = "",
    model: VLM | None = None,
    transport: object | None = None,
    check_endpoints: bool = True,
    skills_root: Path = SKILLS,
) -> Path | None:
    """Train, select and freeze one skill, or with dry_run only show the setup.

    Args:
        cfg: The target config, thinking off.
        options: The run's options.
        api_key: The provider's key; unused by a dry run.
        model: A prebuilt target model, or None to build the vLLM client.
        transport: A stand-in for the optimizer's network, for tests.
        check_endpoints: Whether to preflight the optimizer endpoint.
        skills_root: Root of the frozen skill tree.

    Returns:
        The directory holding the selected skill, or None for a dry run.
    """
    benchmark = build_benchmark(cfg.benchmark)
    train, val = split_items(benchmark, options.limit)
    seed = seed_skill(cfg.benchmark.name, options.init_skill)
    run_path = cfg.run_path.resolve()
    model_name = Path(cfg.train.run_dir).name
    dest = (
        run_path / "selected"
        if options.limit
        else skills_root / cfg.benchmark.name / model_name / options.name()
    )
    flat = skillopt_config(options, cfg.benchmark.name, cfg.model.model_id, seed, run_path)
    adapter = BenchmarkAdapter(
        cfg,
        benchmark,
        train,
        val,
        workers=options.workers,
        exec_workers=options.exec_workers,
        analyst_workers=int(flat["analyst_workers"]),  # type: ignore[call-overload]
        failure_only=bool(flat["failure_only"]),
        minibatch_size=int(flat["minibatch_size"]),  # type: ignore[call-overload]
        edit_budget=int(flat["edit_budget"]),  # type: ignore[call-overload]
        seed=options.seed,
    )

    if options.dry_run:
        show(cfg, adapter, train, val, seed, flat, dest)
        return None

    if not options.limit and dest.exists():
        raise SystemExit(f"{dest} exists; a frozen skill is never overwritten")

    if importlib.util.find_spec("openai") is None:
        raise SystemExit("SkillOpt needs the skillopt extra: uv sync --extra dev --extra skillopt")

    record = run_record(cfg, flat, options, seed)
    check_run_dir(run_path, record, options.resume)

    if check_endpoints:
        preflight_optimizer(options, api_key)
        warn_if_proxied(cfg.model.vllm.base_url)

    write_run_record(run_path, record, options.resume)
    log = CallLog(run_path / "optimizer_calls.jsonl")
    install_optimizer(options, api_key, log, transport)
    guard_upstream_target()

    if model is None:
        print(
            f"waiting up to {cfg.model.vllm.startup_timeout:.0f}s for the vLLM server at "
            f"{cfg.model.vllm.base_url}",
            flush=True,
        )
        model = build_model(cfg.model)

    adapter.model = model

    try:
        if not (run_path / "summary.json").exists():
            from skillopt.engine.trainer import ReflACTTrainer

            ReflACTTrainer(flat, adapter).train()
    finally:
        adapter.close()

    counts = {"train": len(train), "val": len(val)}

    return freeze(cfg, options, run_path, dest, log, seed, counts)


def preflight_optimizer(options: Options, key: str):
    """Fail fast unless the provider takes the key, the model and the effort.

    Args:
        options: The run's options.
        key: The provider's key.
    """
    if options.provider == "anthropic":
        anthropic_api.preflight(options.base_url, key, options.optimizer, options.effort)
    else:
        preflight(options.base_url, key, options.optimizer, options.effort)


def install_optimizer(options: Options, key: str, log: CallLog, transport: object | None = None):
    """Send SkillOpt's optimizer calls to the provider and log each one.

    Args:
        options: The run's options.
        key: The provider's key.
        log: Where each call is recorded.
        transport: A stand-in for the optimizer's network, for tests.
    """
    if options.provider == "anthropic":
        anthropic_api.install(
            options.base_url, key, options.optimizer, options.effort, log, transport
        )
    else:
        install(options.base_url, key, log, transport)


def probe(options: Options, key: str):
    """Send one tiny request through SkillOpt's optimizer route and report it.

    Args:
        options: The run's options; provider, optimizer, effort and URL are used.
        key: The provider's key.
    """
    from skillopt.model import chat_optimizer, set_optimizer_deployment, set_reasoning_effort
    from skillopt.model.backend_config import set_optimizer_backend

    preflight_optimizer(options, key)

    with tempfile.TemporaryDirectory() as tmp:
        log = CallLog(Path(tmp) / "probe.jsonl")
        install_optimizer(options, key, log)
        set_optimizer_backend("openai_chat")
        set_optimizer_deployment(options.optimizer)
        set_reasoning_effort(options.effort)
        text, _ = chat_optimizer(
            system="Answer with one word.",
            user="What is the capital of France?",
            max_completion_tokens=2048,
            retries=1,
            stage="probe",
        )
        entry = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])

    print(json.dumps({"reply": text, **entry}, indent=2))

    # Only OpenRouter keeps per-generation stats to look up.
    if options.provider != "openrouter":
        return

    generation = str(entry.get("generation_id") or "")
    headers = {"Authorization": f"Bearer {key}"}

    with httpx.Client(timeout=30.0, headers=headers) as client:
        stats = generation_stats(client, generation) if generation else {}

    print(json.dumps({name: stats.get(name) for name in STATS_KEYS}, indent=2))


def read_key(path: Path | None, provider: str) -> str:
    """The provider's key, then every optimizer key removed from the environment.

    Child processes, among them T1's executor running model-written code,
    inherit the environment; the key stays in memory only.

    Args:
        path: --api-key-file, or None.
        provider: The optimizer provider, which names the variable read.

    Returns:
        The key.
    """
    key = api_key(path, KEY_VARIABLES[provider])

    for variable in KEY_VARIABLES.values():
        os.environ.pop(variable, None)

    return key


def main(argv: list[str] | None = None):
    """Resolve the run from the command line and carry it out.

    Args:
        argv: Arguments, or None for sys.argv.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    options = Options(
        optimizer=args.optimizer,
        effort=args.reasoning_effort,
        seed=args.seed,
        limit=args.limit,
        workers=args.workers,
        exec_workers=args.exec_workers,
        init_skill=args.init_skill,
        base_url=args.openrouter_url if args.provider == "openrouter" else anthropic_api.API,
        provider=args.provider,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    if args.probe_optimizer:
        probe(options, read_key(args.api_key_file, options.provider))
        return

    config_path, base = resolve_config(args.benchmark)
    cfg = resolve_target(config_path, [f"train.run_name={options.run_name()}", *args.set])
    run_dir = Path(cfg.train.run_dir).resolve()

    if run_dir.parent != Path(base.train.run_dir).resolve():
        raise SystemExit(
            f"train.run_dir must be {base.train.run_dir}/<model>, as for the eval jobs, "
            f"e.g. --set train.run_dir={base.train.run_dir}/qwen3.5-9B; got {cfg.train.run_dir}"
        )

    key = "" if args.dry_run else read_key(args.api_key_file, options.provider)
    run(cfg, options, api_key=key)


if __name__ == "__main__":
    main()
