"""Cost of SkillOpt and Skill.png per model.

The two methods spend different resources, so each column has its own
measure: SkillOpt the input and output tokens of its optimizer LLM, GPT-6
Luna by default, and Skill.png the GPU hours of training its image. Both are
the mean over the benchmarks, for the skill and image that analyze_runs
reports in non-reasoning mode, each traced back to its run by digest. A run
whose optimization found nothing better than no skill still counts, as its
cost was spent all the same. GPU hours are the wandb runtime of every job of
a training run times the GPUs its layers were split across.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from analyze_runs import (
    BENCHMARKS,
    DISTRACTORS,
    INJECTED_CONDITION,
    MAIN_OPTIMIZER,
    MODELS,
    OPTIMIZERS,
    RESOLUTION,
    RUNS_DIR,
    SEED,
    SKILLOPT_CONDITION,
    at_resolution,
    is_ensemble,
    is_transfer,
    load_image_sizes,
    load_rows,
    select_latest,
    skillopt_condition,
)

# Cost column key to display name, in column order.
COLUMNS = {"skillopt": "SkillOpt", "skillpng": "Skill.png"}
# Configs written before model.num_gpus existed ran on one GPU.
DEFAULT_GPUS = 1

# Benchmark, model and cost column.
Key = tuple[str, str, str]
# GPU hours, or input and output tokens.
Cost = float | tuple[int, int]


def find_reported(runs_dir: Path, seed: int, distractors: int,
                  resolution: int,
                  optimizer: str) -> dict[Key, dict[str, Any]]:
    """Pick the Skill.png and SkillOpt rows the main table reports.

    Args:
        runs_dir: Directory holding benchmark/model/results.jsonl files.
        seed: Seed to keep; rows from other seeds are dropped.
        distractors: TRAJECT-Bench distractor count to keep.
        resolution: Image side length reported for Qwen models.
        optimizer: Optimizer LLM as SkillOpt run names spell it.
    Returns:
        Result rows keyed by benchmark, model and cost column.
    """
    # Ensemble and resolution ablation runs would otherwise win as the
    # newest Skill.png row, as in analyze_runs.
    sizes = load_image_sizes(runs_dir)
    rows = [row for row in load_rows(runs_dir, seed)
            if not is_transfer(row) and not is_ensemble(row)
            and row.get("n_distractors") in (None, distractors)
            and at_resolution(row, sizes, resolution)]
    selected, _ = select_latest(rows)
    columns = {INJECTED_CONDITION: "skillpng",
               skillopt_condition(optimizer): "skillopt"}
    reported = {}
    for (benchmark, model, reasoning, condition), row in selected.items():
        if not reasoning and condition in columns:
            reported[(benchmark, model, columns[condition])] = row
    return reported


def index_images(runs_dir: Path) -> dict[Key, Path]:
    """Find the training run that saved each image.

    Args:
        runs_dir: Directory holding benchmark/model/run training trees.
    Returns:
        Training run directories keyed by benchmark, model and digest.
    """
    runs = {}
    for name in ("best.json", "latest.json"):
        for path in runs_dir.glob(f"*/*/*/{name}"):
            run = path.parent
            digest = json.loads(path.read_text())["digest"]
            runs[(run.parent.parent.name, run.parent.name, digest)] = run
    return runs


def index_skills(runs_dir: Path, optimizer: str) -> dict[Key, Path]:
    """Find the SkillOpt run of one optimizer LLM behind each best skill.

    Args:
        runs_dir: Directory holding benchmark/model/run SkillOpt trees.
        optimizer: Optimizer LLM as SkillOpt run names spell it.
    Returns:
        SkillOpt run directories keyed by benchmark, model and digest.
    """
    runs = {}
    # Another optimizer that kept the seed skill has the same digest, so
    # only this optimizer's runs are read.
    pattern = f"*/*/{SKILLOPT_CONDITION}_{optimizer}_*/best_skill.md"
    for path in runs_dir.glob(pattern):
        run = path.parent
        # Eval rows hash the stripped skill text.
        skill = path.read_text(encoding="utf-8").strip()
        digest = hashlib.sha256(skill.encode()).hexdigest()
        runs[(run.parent.parent.name, run.parent.name, digest)] = run
    return runs


def gpu_hours(run: Path) -> float | None:
    """Sum the GPU hours of every job that trained one image.

    Args:
        run: Training run directory.
    Returns:
        GPU hours, or None when no job recorded its runtime.
    """
    # Resolved, as latest-run links to one of the job directories.
    jobs = {path.resolve().parent
            for path in run.glob("wandb/*/files/wandb-summary.json")}
    seconds = []
    for files in jobs:
        summary = json.loads((files / "wandb-summary.json").read_text())
        if summary.get("_runtime") is None:
            continue
        config = yaml.safe_load((files / "config.yaml").read_text())
        gpus = config["model"]["value"].get("num_gpus", DEFAULT_GPUS)
        seconds.append(summary["_runtime"] * gpus)
    return sum(seconds) / 3600 if seconds else None


def optimizer_tokens(run: Path) -> tuple[int, int] | None:
    """Sum the input and output tokens of every optimizer LLM call of a run.

    Args:
        run: SkillOpt run directory.
    Returns:
        Tuple of input and output tokens, or None without a call log.
    """
    path = run / "optimizer_calls.jsonl"
    if not path.exists():
        return None
    prompt = completion = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        usage = json.loads(line).get("usage") or {}
        prompt += usage.get("prompt_tokens", 0)
        # Reasoning tokens are counted within the completion tokens.
        completion += usage.get("completion_tokens", 0)
    return prompt, completion


def measure(reported: dict[Key, dict[str, Any]], images: dict[Key, Path],
            skills: dict[Key, Path]) -> tuple[dict[Key, Cost],
                                              dict[Key, str]]:
    """Measure the cost of each model's runs on every benchmark.

    Args:
        reported: Result rows keyed by benchmark, model and cost column.
        images: Training run directories keyed by benchmark, model, digest.
        skills: SkillOpt run directories keyed the same way.
    Returns:
        Tuple of the costs and of why a cost is missing, keyed like reported.
    """
    costs: dict[Key, Cost] = {}
    missing: dict[Key, str] = {}
    models = {model for _, model, _ in reported}
    for model in models:
        for benchmark in BENCHMARKS:
            for column in COLUMNS:
                key = (benchmark, model, column)
                row = reported.get(key)
                if row is None:
                    missing[key] = "not reported"
                    continue
                cost: Cost | None
                if column == "skillpng":
                    run = images.get((benchmark, model, row["image_digest"]))
                    cost = gpu_hours(run) if run else None
                else:
                    run = skills.get((benchmark, model,
                                      row.get("skill_digest", "")))
                    cost = optimizer_tokens(run) if run else None
                if run is None:
                    missing[key] = f"no run on disk for {row['run_name']}"
                elif cost is None:
                    missing[key] = f"no cost recorded in {run}"
                else:
                    costs[key] = cost
    return costs, missing


def build_table(costs: dict[Key, Cost],
                missing: dict[Key, str]) -> list[dict[str, Any]]:
    """Collect each model's costs over the benchmarks.

    Args:
        costs: Costs keyed by benchmark, model and cost column.
        missing: Reasons of missing costs, keyed the same way.
    Returns:
        List of entries carrying a model and its costs per column.
    """
    models = {model for _, model, _ in [*costs, *missing]}
    order = [name for name in MODELS if name in models]
    order += sorted(models - set(order))
    table = []
    for model in order:
        entry: dict[str, Any] = {"model": model, "costs": {}}
        for column in COLUMNS:
            entry["costs"][column] = [
                costs[(benchmark, model, column)] for benchmark in BENCHMARKS
                if (benchmark, model, column) in costs]
        table.append(entry)
    return table


def format_tokens(count: float) -> str:
    """Render a token count in millions.

    Args:
        count: Token count.
    Returns:
        The count in millions with a unit suffix.
    """
    return f"{count / 1e6:.2f}M"


def format_cell(column: str, values: list[Any]) -> str:
    """Render the benchmark mean of one model's costs.

    Args:
        column: Cost column key.
        values: Costs of the benchmarks that have one.
    Returns:
        The mean cost with its unit, or a dash when there is none.
    """
    if not values:
        return "--"
    if column == "skillpng":
        return f"{mean(values):.2f} GPU h"
    prompt = mean(value[0] for value in values)
    completion = mean(value[1] for value in values)
    return f"{format_tokens(prompt)} in / {format_tokens(completion)} out"


def render_terminal(table: list[dict[str, Any]]) -> str:
    """Render the cost table as an aligned terminal table.

    Args:
        table: Entries from build_table.
    Returns:
        The rendered table.
    """
    header = ["Model", *COLUMNS.values()]
    rows = [[MODELS.get(entry["model"], entry["model"]),
             *(format_cell(column, entry["costs"][column])
               for column in COLUMNS)]
            for entry in table]
    widths = [max(len(row[i]) for row in [header, *rows])
              for i in range(len(header))]

    def line(cells: list[str]) -> str:
        """Lay out one table row.

        Args:
            cells: Cell texts in column order.
        Returns:
            The padded row, model left-aligned and costs right-aligned.
        """
        text = [cell.ljust(widths[i]) if not i else cell.rjust(widths[i])
                for i, cell in enumerate(cells)]
        return "  ".join(text).rstrip()

    lines = [line(header), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    lines.extend(line(row) for row in rows)
    return "\n".join(lines)


def render_missing(missing: dict[Key, str]) -> str:
    """Describe the costs left out of the means.

    Args:
        missing: Reasons of missing costs keyed by benchmark, model, column.
    Returns:
        A report of the missing costs, empty when nothing is missing.
    """
    if not missing:
        return ""
    lines = [f"Left out of the mean ({len(missing)}):"]
    for (benchmark, model, column), reason in sorted(missing.items()):
        lines.append(f"  {BENCHMARKS[benchmark]}, {MODELS.get(model, model)},"
                     f" {COLUMNS[column]}: {reason}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="directory holding the benchmark run trees")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="only report runs with this seed")
    parser.add_argument("--distractors", type=int, default=DISTRACTORS,
                        help="only report TRAJECT-Bench runs with this many"
                             " distractor tools")
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="only report Qwen Skill.png runs whose image"
                             " is this many pixels square")
    parser.add_argument("--optimizer", default=MAIN_OPTIMIZER,
                        choices=OPTIMIZERS,
                        help="report the SkillOpt skills of this optimizer"
                             " LLM")
    args = parser.parse_args()

    reported = find_reported(args.runs_dir, args.seed, args.distractors,
                             args.resolution, args.optimizer)
    if not reported:
        raise SystemExit(f"no Skill.png or SkillOpt results with "
                         f"seed={args.seed} under {args.runs_dir}")
    costs, missing = measure(reported, index_images(args.runs_dir),
                             index_skills(args.runs_dir, args.optimizer))
    print("Cost per benchmark, mean over "
          f"{', '.join(BENCHMARKS.values())}")
    print(f"SkillOpt: {OPTIMIZERS[args.optimizer]} input / output tokens."
          " Skill.png: GPU hours of training.")
    print(render_terminal(build_table(costs, missing)))
    report = render_missing(missing)
    if report:
        print()
        print(report)


if __name__ == "__main__":
    main()
