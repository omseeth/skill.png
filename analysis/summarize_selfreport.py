"""Summarize the judged self-reports of selfreport_skillpng.py.

For every benchmark, one line per viewer and condition: no image, the
viewer's own Skill.png, the other models' Skill.png pooled, noise and gray.
The open questions (instructions and behave, pooled) give the judge's rates:
how often the answer brings up an image, each of the benchmark's TARGETS,
and the mean count of rules beyond the system prompt. The rules question gives
what the response did, measured from the output itself: its length in
characters, whether it closed an answer tag, imports and save_to_cache in
T1's code, words in SearchQA's answer, whether LiveMath's
pick is the longest option, TRAJECT's call count, and the benchmark's own
score where it needs no execution. Then the rules each viewer states only
with its own image, against none.

Reads runs/answers/<model>/selfreport_judged.jsonl, or selfreport.jsonl for
the measures that need no judge. With --ablations it reads the
selfreport_ablations files instead, one line per ablation image and control,
with the main file's no-image and own-image answers as the baseline: images
by variant, controls as noise@<side> and gray@<side>, and the skilled
context's conditions prefixed "+skill:".
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from analyze_runs import RUNS_DIR
from describe_skillpng import CONFIGS_DIR, MAIN_MODELS, SKILLED
from judge_selfreport import TARGETS, row_key
from selfreport_skillpng import SPLIT

from skillpng.benchmarks.registry import build_benchmark
from skillpng.config import load_config

CONDITIONS = ("none", "own", "other", "noise", "gray")
OPEN = ("instructions", "behave")
# Scored here only where score() runs no code; T1's executes it.
SCORED = ("searchqa", "livemathematicianbench", "traject")
SHORT = {"qwen3.5-4B": "Q4B", "qwen3.5-9B": "Q9B", "gemma-4-E4B-it": "G4B",
         "nemotron-3-nano-omni-30B-A3B-reasoning": "Nemo"}


def load(runs_dir: Path, model: str,
         name: str = "selfreport") -> list[dict]:
    """Read one viewer's answers, judged where a verdict exists.

    Args:
        runs_dir: Directory holding runs/answers.
        model: Viewer model.
        name: Stem of the answers file.
    Returns:
        One row per answer, the latest verdict joined on, judge None when
        unjudged.
    """
    directory = runs_dir / "answers" / model
    with open(directory / f"{name}.jsonl") as file:
        rows = {row_key(row): {**row, "judge": None}
                for row in map(json.loads, file)}
    judged = directory / f"{name}_judged.jsonl"
    if judged.exists():
        with open(judged) as file:
            for row in map(json.loads, file):
                if row["judge"] is not None and row_key(row) in rows:
                    rows[row_key(row)]["judge"] = row["judge"]
    return list(rows.values())


def condition(row: dict) -> str:
    """Name the condition an answer was given under, from its viewer's side.

    Args:
        row: An answer row.
    Returns:
        One of CONDITIONS for a main image or control; for an ablation image
        its variant, for an ablation control noise@<side> or gray@<side>,
        "+skill:" first under a skilled context.
    """
    variant = row.get("image_variant", "main")
    if variant == "main":
        if row["image"] != "skillpng":
            return row["image"]
        return "own" if row["image_model"] == row["model"] else "other"
    name = (variant if row["image"] == "skillpng"
            else row["image"] if row["image"] == "none"
            else f"{row['image']}@{row['image_side']}")
    return ("+skill:" + name if row["context"].endswith(SKILLED)
            else name)


def last_answer(output: str) -> str | None:
    """Read the last <answer> tag's text.

    Args:
        output: A response.
    Returns:
        The tag's stripped text, or None without a closed tag.
    """
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", output, re.S)
    return tagged[-1] if tagged else None


def behaviour(row: dict, items: dict) -> dict:
    """Measure what a rules response did.

    Args:
        row: A rules answer.
        items: Val items by benchmark and id.
    Returns:
        The benchmark's measures; None where the output holds no answer.
    """
    benchmark, output = row["benchmark"], row["output"]
    item = items[benchmark][row["item_id"]]
    # Straight to the answer: a tag, code or a call list first, and no word
    # of rules in the opening.
    out = {"chars": len(output),
           "skips_rules": bool(re.match(r"\s*(<answer>|<CODE>|```|\[\{)", output))
           and not re.search(r"(?i)\brules?\b", output[:400])}
    if benchmark == "t1":
        out["imports"] = bool(re.search(r"(?m)^\s*(from\s+\S+\s+import|import\s+\w)",
                                        output))
        out["save_to_cache"] = "save_to_cache(" in output
        return out
    answer = last_answer(output)
    out["answered"] = answer is not None
    if benchmark == "searchqa":
        out["answer_words"] = len(answer.split()) if answer else None
    elif benchmark == "livemathematicianbench":
        options = item.meta["options"]
        longest = max(options, key=lambda option: len(option["text"]))
        picked = re.fullmatch(r"\W*([A-E])\W*", answer or "")
        out["longest"] = (picked.group(1) == longest["label"]) if picked else None
    elif benchmark == "traject":
        try:
            calls = json.loads(answer) if answer else None
        except json.JSONDecodeError:
            calls = None
        out["calls"] = len(calls) if isinstance(calls, list) else None
    out["score"] = items["_bench"][benchmark].score(output, item)
    return out


def rate(values: list) -> str:
    """Format booleans as a percentage over those present.

    Args:
        values: Booleans, None for missing.
    Returns:
        "p%" or "-" when none are present.
    """
    present = [value for value in values if value is not None]
    return f"{100 * mean(present):.0f}%" if present else "-"


def average(values: list) -> str:
    """Format numbers as a mean over those present.

    Args:
        values: Numbers, None for missing.
    Returns:
        The mean to one decimal, or "-".
    """
    present = [value for value in values if value is not None]
    return f"{mean(present):.1f}" if present else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--models", nargs="+", default=list(MAIN_MODELS))
    parser.add_argument("--ablations", action="store_true",
                        help="summarize the ablation images instead")
    args = parser.parse_args()

    rows = [row for model in args.models
            if (args.runs_dir / "answers" / model
                / "selfreport.jsonl").exists()
            for row in load(args.runs_dir, model)]
    for row in rows:
        row["condition"] = condition(row)
    conditions = CONDITIONS
    if args.ablations:
        rows = [row for row in rows if row["condition"] in ("none", "own")]
        ablated = [row for model in args.models
                   if (args.runs_dir / "answers" / model
                       / "selfreport_ablations.jsonl").exists()
                   for row in load(args.runs_dir, model,
                                   "selfreport_ablations")]
        for row in ablated:
            row["condition"] = condition(row)
        rows += ablated
        # Baseline, images, then controls by size, skilled ones last.
        conditions = sorted({row["condition"] for row in rows}, key=lambda c: (
            c.startswith("+skill:"), c not in ("none", "own"),
            "@" in c or c.endswith("none"),
            int(re.sub(r"\D", "", c) or 0), c))
    benchmarks = sorted({row["benchmark"] for row in rows})
    items = {"_bench": {}}
    for benchmark in benchmarks:
        bench = build_benchmark(
            load_config(CONFIGS_DIR / f"{benchmark}.yaml").benchmark)
        items["_bench"][benchmark] = bench
        items[benchmark] = {item.id: item for item in bench.items(SPLIT)}
    for row in rows:
        row["behaviour"] = (behaviour(row, items)
                            if row["question_id"] == "rules" else {})

    judged = sum(row["judge"] is not None for row in rows)
    print(f"{len(rows)} answers, {judged} judged")
    for benchmark in benchmarks:
        targets = list(TARGETS[benchmark])
        measures = sorted({key for row in rows if row["benchmark"] == benchmark
                           for key in row["behaviour"]})
        print(f"\n== {benchmark}: open questions (instructions + behave)"
              f" judged; rules question measured")
        header = (["viewer", "cond", "n", "image", *targets, "rules#", "|",
                   "n", *measures])
        print("  ".join(f"{h:>13}" if i > 1 else f"{h:<6}" if i == 0
                        else f"{h:<20}" for i, h in enumerate(header)))
        for model in args.models:
            for name in conditions:
                cell = [row for row in rows if row["benchmark"] == benchmark
                        and row["model"] == model and row["condition"] == name]
                if not cell:
                    continue
                verdicts = [row["judge"] for row in cell
                            if row["question_id"] in OPEN and row["judge"]]
                done = [row["behaviour"] for row in cell
                        if row["question_id"] == "rules"]
                line = [SHORT[model], name, str(len(verdicts)),
                        rate([v["mentions_image"] for v in verdicts]),
                        *(rate([v[key] for v in verdicts]) for key in targets),
                        average([len(v["new_rules"]) for v in verdicts]),
                        "|", str(len(done))]
                for key in measures:
                    values = [d.get(key) for d in done]
                    line.append(rate(values) if isinstance(
                        next((v for v in values if v is not None), 0), bool)
                        else average(values))
                print("  ".join(f"{c:>13}" if i > 1 else f"{c:<6}" if i == 0
                                else f"{c:<20}" for i, c in enumerate(line)))
            print()

    print("\n== rules stated with the own image and never with none"
          " (open questions)")
    for benchmark in benchmarks:
        for model in args.models:
            said = {name: Counter(rule.lower().strip(" .")
                                  for row in rows
                                  if row["benchmark"] == benchmark
                                  and row["model"] == model
                                  and row["condition"] == name
                                  and row["question_id"] in OPEN
                                  and row["judge"]
                                  for rule in row["judge"]["new_rules"])
                    for name in ("own", "none")}
            only = [rule for rule, _ in said["own"].most_common()
                    if rule not in said["none"]][:6]
            if only:
                print(f"{benchmark} {SHORT[model]}: " + " | ".join(only))


if __name__ == "__main__":
    main()
