"""Significance tests behind the comparisons the paper reports.

Each test is a two-sided exact McNemar test on the per-item test scores of
two runs (i.e., from Skill.png and SkillOpt) of one model and benchmark, in 
non-reasoning mode. The runs are the ones analyze_runs reports, and their 
per-item scores are read from each run's eval/<split>-<condition>.jsonl. A file 
is only tested if it holds as many items as its result row and reproduces its score.

The main results test Skill.png against the GPT-6 Luna SkillOpt skill on
every model and benchmark, and on Nemotron the Skill.png trained and
evaluated with the Claude Sonnet 5.5 skill against Skill.png alone. As in
those tables, a failed run is tested with the run whose score its cell
shows, marked with an asterisk.

The transferability results test the SkillOpt skill and the Skill.png of
Qwen3.5-4B and the ensemble image of Qwen3.5-4B and Gemma-4-E4B-it against
No Skill. They also test the ensemble image against each model's own
Skill.png and the Qwen3.5-4B image, and both images against the random image
on the held-out models. The UltraBreak image of Qwen3.5-4B is tested against
No Skill on every model, its source included, and against the random image
on the held-out models. As in the transfer table, these cells show no
stand-ins.

Each table shows the score difference in points and the p-value per model
and benchmark, green or red where it is significant at ALPHA. It counts the
pairs the first run wins and loses.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
from typing import Any

from analyze_runs import (
    BASELINE_CONDITION,
    BENCHMARKS,
    CONDITIONED_OPTIMIZER,
    DISTRACTORS,
    ENSEMBLE_MEMBERS,
    ENSEMBLE_MODELS,
    GREEN,
    INJECTED_CONDITION,
    MAIN_OPTIMIZER,
    MODELS,
    NOISE_CONDITION,
    OPTIMIZER_MODEL,
    OPTIMIZERS,
    RED,
    RESET,
    RESOLUTION,
    RUNS_DIR,
    SEED,
    STAND_IN_MARK,
    TRANSFER_CONDITION,
    TRANSFER_ENSEMBLE_CONDITION,
    TRANSFER_SKILLOPT_CONDITION,
    TRANSFER_SOURCE,
    TRANSFER_ULTRABREAK_CONDITION,
    at_resolution,
    conditioned_condition,
    find_failed,
    is_ensemble,
    is_transfer,
    load_failed_images,
    load_failed_skills,
    load_image_sizes,
    load_rows,
    select_ensemble,
    select_latest,
    select_transfer,
    skillopt_condition,
    split_beta1,
    split_seed,
    stand_in_row,
)

# Result row, and the row and stand-in flag of each cell keyed by
# benchmark, model and cell key.
Row = dict[str, Any]
Cells = dict[tuple[str, str, str], tuple[Row, bool]]

# Significance level of every test.
ALPHA = 0.05

# Result-row field naming the per-item file of the row's run.
EVAL_FILE = "eval_file"

# Cell key to the selection the cell is read from and its condition there.
# Only main-table cells take stand-ins for failed runs.
CELLS = {
    "baseline": ("selected", BASELINE_CONDITION),
    "noise": ("selected", NOISE_CONDITION),
    "injected": ("selected", INJECTED_CONDITION),
    "skillopt": ("selected", skillopt_condition(MAIN_OPTIMIZER)),
    "conditioned": ("selected", conditioned_condition(CONDITIONED_OPTIMIZER)),
    "transfer": ("transfer", TRANSFER_CONDITION),
    "transfer_skillopt": ("transfer", TRANSFER_SKILLOPT_CONDITION),
    "transfer_ultrabreak": ("transfer", TRANSFER_ULTRABREAK_CONDITION),
    "ensemble": ("ensemble", TRANSFER_ENSEMBLE_CONDITION),
}

# Models the Qwen3.5-4B skills were injected into, and those of them that
# were not in the ensemble either.
TARGETS = tuple(model for model in ENSEMBLE_MODELS if model != TRANSFER_SOURCE)
HELD_OUT = tuple(model for model in ENSEMBLE_MODELS
                 if model not in ENSEMBLE_MEMBERS)
SOURCE = MODELS[TRANSFER_SOURCE]

# Title, the cell the difference is taken for, the cell it is taken against
# and the tested models of each table.
COMPARISONS = (
    (f"Main results: Skill.png against SkillOpt"
     f" ({OPTIMIZERS[MAIN_OPTIMIZER]})", "injected", "skillopt",
     tuple(MODELS)),
    (f"Main results: SkillOpt ({OPTIMIZERS[CONDITIONED_OPTIMIZER]})"
     " + Skill.png against Skill.png", "conditioned", "injected",
     (OPTIMIZER_MODEL,)),
    (f"Transferability: SkillOpt ({SOURCE}) against No Skill",
     "transfer_skillopt", "baseline", TARGETS),
    (f"Transferability: Skill.png ({SOURCE}) against No Skill",
     "transfer", "baseline", TARGETS),
    (f"Transferability: UltraBreak ({SOURCE}) against No Skill",
     "transfer_ultrabreak", "baseline", ENSEMBLE_MODELS),
    ("Transferability: Ensemble against No Skill",
     "ensemble", "baseline", ENSEMBLE_MODELS),
    ("Transferability: Ensemble against the model's own Skill.png",
     "ensemble", "injected", ENSEMBLE_MODELS),
    (f"Transferability: Ensemble against Skill.png ({SOURCE})",
     "ensemble", "transfer", TARGETS),
    (f"Transferability: Skill.png ({SOURCE}) against Random Image",
     "transfer", "noise", HELD_OUT),
    ("Transferability: Ensemble against Random Image",
     "ensemble", "noise", HELD_OUT),
    (f"Transferability: UltraBreak ({SOURCE}) against Random Image",
     "transfer_ultrabreak", "noise", HELD_OUT),
)

# Width of the model column and of each benchmark column.
MODEL_WIDTH = 30
COLUMN_WIDTH = 20


def eval_file(row: Row) -> str:
    """Name the per-item file the run of a result row wrote.

    Args:
        row: Result row from load_rows.
    Returns:
        File name in the run's eval directory.
    """
    # load_rows appends a replicate's training seed to its condition, the
    # beta1 of the beta1 ablation before that, and the optimizer LLM to the
    # SkillOpt conditions before both.
    condition = split_beta1(split_seed(row["condition"])[0])[0]
    for optimizer in OPTIMIZERS:
        condition = condition.removesuffix(f"_{optimizer}")
    return f"{row['split']}-{condition}.jsonl"


@functools.cache
def read_scores(path: Path) -> dict[str, float]:
    """Read the score of every item in a per-item file.

    Args:
        path: Per-item file of one run.
    Returns:
        Score of each item, keyed by item id.
    """
    scores = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            scores[record["item_id"]] = float(record["score"])
    return scores


def load_items(runs_dir: Path, row: Row) -> dict[str, float]:
    """Read the per-item scores behind one result row.

    Args:
        runs_dir: Directory holding the benchmark run trees.
        row: Result row tagged with its per-item file.
    Returns:
        Binary score of each item, keyed by item id.
    """
    path = (runs_dir / row["benchmark"] / row["model"] / row["run_name"]
            / "eval" / row[EVAL_FILE])
    name = path.relative_to(runs_dir)
    if not path.exists():
        raise ValueError(f"no per-item file {name}")
    scores = read_scores(path)
    if set(scores.values()) - {0.0, 1.0}:
        raise ValueError(f"non-binary scores in {name}")
    # A rerun under the same run name may have replaced the file.
    if (len(scores) != row["n_items"]
            or not math.isclose(sum(scores.values()) / len(scores),
                                row["score"], abs_tol=1e-9)):
        raise ValueError(f"{name} does not hold the scores of its result row")
    return scores


def mcnemar(first: dict[str, float], second: dict[str, float]) -> float:
    """Run the two-sided exact McNemar test on paired binary scores.

    Args:
        first: Score of each item under the first run.
        second: Score of each item under the second run.
    Returns:
        P-value of the exact binomial test on the discordant items.
    """
    if first.keys() != second.keys():
        raise ValueError("the two runs scored different items")
    wins = sum(first[item] > second[item] for item in first)
    losses = sum(first[item] < second[item] for item in first)
    discordant = wins + losses
    tail = sum(math.comb(discordant, count)
               for count in range(min(wins, losses) + 1))
    p: float = 2 * tail / 2 ** discordant
    return min(1.0, p)


def collect_cells(runs_dir: Path, seed: int, distractors: int,
                  resolution: int) -> Cells:
    """Pick the row behind every non-reasoning cell in CELLS.

    Args:
        runs_dir: Directory holding the benchmark run trees.
        seed: Seed of the reported runs.
        distractors: Distractor tools of the reported TRAJECT-Bench runs.
        resolution: Side length of the reported Qwen Skill.png images.
    Returns:
        Row and stand-in flag keyed by benchmark, model and cell key.
    """
    rows = [{**row, EVAL_FILE: eval_file(row)}
            for row in load_rows(runs_dir, seed)
            if row.get("n_distractors") in (None, distractors)]
    ensembled = [row for row in rows if is_ensemble(row)]
    loaded = [row for row in rows if not is_ensemble(row)]
    sizes = load_image_sizes(runs_dir)
    selected, _ = select_latest([row for row in loaded
                                 if not is_transfer(row)
                                 and at_resolution(row, sizes, resolution)])
    failed = find_failed(selected, load_failed_images(runs_dir),
                         load_failed_skills(runs_dir))
    transfer, _ = select_transfer(loaded, selected)
    ensemble, _ = select_ensemble(ensembled, selected, transfer)
    sources = {"selected": selected, "transfer": transfer,
               "ensemble": ensemble}

    cells: Cells = {}
    for cell, (source, condition) in CELLS.items():
        for key, row in sources[source].items():
            benchmark, model, reasoning, cell_condition = key
            if reasoning or cell_condition != condition:
                continue
            stand_in = source == "selected" and key in failed
            if stand_in:
                row = stand_in_row(selected, key)
            if row:
                cells[(benchmark, model, cell)] = (row, stand_in)
    return cells


def format_result(delta: float, p: float | None, mark: str) -> str:
    """Render the score difference and p-value of one tested pair.

    Args:
        delta: Score difference in points.
        p: P-value, or None when the pair could not be tested.
        mark: Stand-in mark, empty when both cells show their own run.
    Returns:
        Cell text with the p-value, n/a for an untested pair.
    """
    if p is None:
        return f"{delta:+.2f} (n/a){mark}"
    p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    return f"{delta:+.2f} ({p_text}){mark}"


def paint(text: str, delta: float, p: float | None) -> str:
    """Colour a cell green or red when its difference is significant.

    Args:
        text: Cell text to colour.
        delta: Score difference in points.
        p: P-value, or None when the pair could not be tested.
    Returns:
        The coloured text, unchanged when not significant.
    """
    if p is None or p >= ALPHA or delta == 0:
        return text
    return f"{GREEN if delta > 0 else RED}{text}{RESET}"


def render(title: str, first: str, second: str, models: tuple[str, ...],
           cells: Cells, runs_dir: Path) -> str:
    """Test two cells on every model and benchmark and lay out the results.

    Args:
        title: Heading of the table.
        first: Cell key of the run the difference is taken for.
        second: Cell key of the run it is taken against.
        models: Models whose cells are tested.
        cells: Rows from collect_cells.
        runs_dir: Directory holding the benchmark run trees.
    Returns:
        The table, its win and loss counts and the untested pairs.
    """
    results: dict[tuple[str, str], tuple[float, float | None, str]] = {}
    untested: list[str] = []
    for model in models:
        for benchmark in BENCHMARKS:
            first_cell = cells.get((benchmark, model, first))
            second_cell = cells.get((benchmark, model, second))
            if first_cell is None or second_cell is None:
                continue
            first_row, first_in = first_cell
            second_row, second_in = second_cell
            delta = 100 * (first_row["score"] - second_row["score"])
            mark = STAND_IN_MARK if first_in or second_in else ""
            try:
                p = mcnemar(load_items(runs_dir, first_row),
                            load_items(runs_dir, second_row))
            except ValueError as error:
                p = None
                untested.append(f"  {MODELS.get(model, model)},"
                                f" {BENCHMARKS[benchmark]}: {error}")
            results[(model, benchmark)] = (delta, p, mark)
    if not results:
        return f"{title}\n(no runs)\n"

    shown = [benchmark for benchmark in BENCHMARKS
             if any(key[1] == benchmark for key in results)]
    header = " " * MODEL_WIDTH + "".join(
        f"{BENCHMARKS[benchmark]:<{COLUMN_WIDTH}}" for benchmark in shown)
    lines = [title, header.rstrip(), "-" * len(header)]
    for model in models:
        if not any(key[0] == model for key in results):
            continue
        line = f"{MODELS.get(model, model):<{MODEL_WIDTH}}"
        for benchmark in shown:
            result = results.get((model, benchmark))
            if result is None:
                line += f"{'--':<{COLUMN_WIDTH}}"
                continue
            delta, p, _ = result
            text = f"{format_result(*result):<{COLUMN_WIDTH}}"
            line += paint(text, delta, p)
        lines.append(line)

    wins = [p for delta, p, _ in results.values() if delta > 0]
    losses = [p for delta, p, _ in results.values() if delta < 0]
    summary = (f"Higher in {len(wins)} of {len(results)} pairs,"
               f" {sum(p is not None and p < ALPHA for p in wins)}"
               f" significantly; lower in {len(losses)},"
               f" {sum(p is not None and p < ALPHA for p in losses)}"
               " significantly")
    if untested:
        summary += f"; {len(untested)} untested"
    lines.append(summary + ".")
    if untested:
        lines.append("Untested:")
        lines.extend(untested)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="directory holding the benchmark run trees")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="only test runs with this seed")
    parser.add_argument("--distractors", type=int, default=DISTRACTORS,
                        help="only test TRAJECT-Bench runs with this many"
                             " distractor tools")
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="only test Qwen Skill.png runs whose image"
                             " is this many pixels square")
    args = parser.parse_args()

    cells = collect_cells(args.runs_dir, args.seed, args.distractors,
                          args.resolution)
    if not cells:
        raise SystemExit(f"no results with seed={args.seed} under "
                         f"{args.runs_dir}")
    print(f"Two-sided exact McNemar test, non-reasoning, alpha {ALPHA}."
          " Score difference in points, first run minus second;"
          f" {STAND_IN_MARK} marks a stand-in score.")
    print()
    for title, first, second, models in COMPARISONS:
        print(render(title, first, second, models, cells, args.runs_dir))


if __name__ == "__main__":
    main()
