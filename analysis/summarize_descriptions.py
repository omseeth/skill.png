"""Summarize what each model answered about the Skill.png images.

Reads runs/answers/<model>/descriptions.jsonl, written by
describe_skillpng.py, and prints, per model, the closed questions as counts
next to the noise and gray controls asked the same way: which task the image
is for, what kind of image it is, and its rated meaningfulness. Then, per
benchmark, a grid of the model each image was optimized on against the model
shown it: how often the bare open answers use the benchmark's vocabulary,
and how often the task question under the benchmark's system prompt names
it, controls alongside. Then, from descriptions_ablations.jsonl, the
ablation images next to what they are compared with: the ensemble image next
to its members' own images, the image trained under a SkillOpt skill next to
the plain one, under that skill too, and the image of every ablation size
next to a control of its size. With --examples it also prints the open
answers, the first --examples samples of each.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from analyze_runs import (
    BENCHMARKS,
    ENSEMBLE_MEMBERS,
    MODELS,
    OPTIMIZER_MODEL,
    RESOLUTION,
    RESOLUTION_MODEL,
    RESOLUTIONS,
    RUNS_DIR,
)
from describe_skillpng import KINDS, QUESTIONS, SKILLED, TASKS, parse

CONTROLS = ("noise", "gray")
# Models an image can come from, with the short name its label carries.
SOURCES = {"qwen3.5-4B": "Q4B", "qwen3.5-9B": "Q9B", "gemma-4-E4B-it": "Gem",
           "nemotron-3-nano-omni-30B-A3B-reasoning": "Nem"}
# Open questions, whose answers the vocabulary counts read.
OPEN = ("describe", "read", "instruct", "prompt", "words")
# Words that name each task, matched case-insensitively. A rough count: words
# that also come up for the gray and noise controls are left out ("blank" for
# a gray image, "question" and "answer" for any refusal, "search").
VOCABULARY = {
    "t1": r"travel|\btrips?\b|trail|flight|hotel|restaurant|vacation|"
          r"\btour\b|journey|python|\bcod(?:e|ing)\b|programm",
    "searchqa": r"quiz|trivia|fill[- ]in|jeopardy|passage|retriev",
    "livemathematicianbench": r"math|equation|formula|algebra|geometr|"
                              r"theorem|calcul|variable",
    "traject": r"\btools?\b|\bapi\b|function call|endpoint|parameter",
}
# Short column heads for the options of the closed questions.
SHORT = {**{key: BENCHMARKS.get(key, key) for key in TASKS}, "none": "None"}


def load(runs_dir: Path,
         name: str = "descriptions.jsonl") -> dict[str, list[dict]]:
    """Read every model's answers.

    Args:
        runs_dir: Directory holding runs/answers/<model>/<name>.
        name: Answer file, descriptions.jsonl or descriptions_ablations.jsonl.
    Returns:
        Answer rows per model directory name, closed answers parsed anew.
    """
    out = {}
    for path in sorted((runs_dir / "answers").glob(f"*/{name}")):
        with open(path) as file:
            rows = [json.loads(line) for line in file]
        # Parsed again, so a parser fix needs no new answers.
        for row in rows:
            row["answer"] = parse(row["output"], row["options"],
                                  row["question_id"])
        out[path.parent.name] = rows
    return out


def groups(rows: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """Group a model's answers by image under each context, controls last.

    Args:
        rows: One model's answer rows.
    Returns:
        Context, image label and the rows of that pair. Skill.png images are
        labelled by benchmark and the model they were optimized on, own
        first; under a benchmark's system prompt only that benchmark's
        images and the controls are listed.
    """
    out = []
    contexts = ["bare", *[b for b in BENCHMARKS
                          if any(r["context"] == b for r in rows)]]
    for context in contexts:
        images = {(r["image_benchmark"], r.get("image_model") or r["model"])
                  for r in rows
                  if r["image"] == "skillpng" and r["context"] == context}
        model = rows[0]["model"]
        ordered = sorted(images, key=lambda key: (
            list(BENCHMARKS).index(key[0]), key[1] != model,
            list(SOURCES).index(key[1])))
        for benchmark, source in ordered:
            picked = [r for r in rows if r["context"] == context
                      and r["image"] == "skillpng"
                      and r["image_benchmark"] == benchmark
                      and (r.get("image_model") or r["model"]) == source]
            name = "own" if source == model else SOURCES[source]
            out.append((context, f"{BENCHMARKS[benchmark]} {name}", picked))
        for label in CONTROLS:
            picked = [r for r in rows
                      if r["context"] == context and r["image"] == label]
            if picked:
                out.append((context, label, picked))
    return out


def counts(rows: list[dict], question_id: str, keys: dict) -> str:
    """Tally a multiple-choice question's answers.

    Args:
        rows: Answer rows of one image under one context.
        question_id: Question id.
        keys: The question's options.
    Returns:
        Counts per option in option order, then unreadable ones.
    """
    tally = Counter(r["answer"] for r in rows
                    if r["question_id"] == question_id)
    cells = [f"{tally.get(key, 0):>3}" for key in keys]
    return " ".join(cells) + f" {tally.get(None, 0):>3}"


def cross(answers: dict[str, list[dict]]) -> str:
    """Grid each benchmark's images by the model optimized on and shown.

    Args:
        answers: Answer rows per model shown the images.
    Returns:
        One block per benchmark: a row per model an image was optimized on,
        noise and gray last, a column per model shown it. Each cell holds
        the percentage of bare open answers using the benchmark's vocabulary,
        then how many task answers under its system prompt named it.
    """
    viewers = [model for model in SOURCES if model in answers]
    lines = []
    for benchmark, words in VOCABULARY.items():
        lines.append(f"\n{BENCHMARKS[benchmark]}: vocabulary % (bare open"
                     f" answers) / task named (under its prompt)")
        lines.append(f"{'optimized on':12} | "
                     + " ".join(f"{SOURCES[v]:>9}" for v in viewers))
        for source in [*SOURCES, *CONTROLS]:
            cells = []
            for viewer in viewers:
                rows = [r for r in answers[viewer]
                        if (r["image"] == source if source in CONTROLS else
                            r["image"] == "skillpng"
                            and r["image_benchmark"] == benchmark
                            and (r.get("image_model") or r["model"])
                            == source)]
                bare = [r for r in rows
                        if r["context"] == "bare" and r["question_id"] in OPEN]
                task = [r for r in rows if r["context"] == benchmark
                        and r["question_id"] == "task"]
                if not bare:
                    cells.append(f"{'--':>9}")
                    continue
                share = 100 * sum(bool(re.search(words, r["output"], re.I))
                                  for r in bare) / len(bare)
                named = sum(r["answer"] == benchmark for r in task)
                mark = "*" if source == viewer else " "
                cells.append(f"{share:3.0f}{mark}{named:>2}/{len(task)}")
            name = SOURCES.get(source, source)
            lines.append(f"{name:12} | " + " ".join(f"{c:>9}" for c in cells))
    lines.append("\n* the model shown its own image")
    return "\n".join(lines)


def measure(rows: list[dict], benchmark: str, context: str) -> str:
    """Summarize one viewer's answers about one image in one cell.

    Args:
        rows: The viewer's answers about the image, every context.
        benchmark: Benchmark whose vocabulary and task are looked for.
        context: Context the task question is read under.
    Returns:
        Vocabulary share of the bare open answers, then task answers naming
        the benchmark under the context, e.g. "80% 4/5".
    """
    bare = [r for r in rows
            if r["context"] == "bare" and r["question_id"] in OPEN]
    task = [r for r in rows
            if r["context"] == context and r["question_id"] == "task"]
    if not bare or not task:
        return "--"
    share = 100 * sum(bool(re.search(VOCABULARY[benchmark], r["output"], re.I))
                      for r in bare) / len(bare)
    named = sum(r["answer"] == benchmark for r in task)
    return f"{share:3.0f}% {named}/{len(task)}"


def looks(rows: list[dict]) -> str:
    """Summarize how one viewer saw one image, bare.

    Args:
        rows: The viewer's answers about the image, every context.
    Returns:
        Kind answers calling it noise, then the mean meaning rating, e.g.
        "noise 1/5, 3.4".
    """
    kind = [r for r in rows
            if r["context"] == "bare" and r["question_id"] == "kind"]
    ratings = [r["answer"] for r in rows if r["context"] == "bare"
               and r["question_id"] == "meaning" and r["answer"]]
    rating = f"{mean(ratings):.1f}" if ratings else "--"
    noise = sum(r["answer"] == "noise" for r in kind)
    return f"n{noise}/{len(kind)} m{rating}"


def pick(rows: list[dict], label: str, benchmark: str | None = None,
         model: str | None = None, variant: str | None = None,
         side: int | None = None) -> list[dict]:
    """Select the answers about one image.

    Args:
        rows: One viewer's answers.
        label: skillpng or a control's label.
        benchmark: Benchmark a Skill.png was optimized for.
        model: Model it was optimized on.
        variant: Ablation variant, for answers of descriptions_ablations.jsonl.
        side: Image side, for the controls of the ablations.
    Returns:
        The matching rows.
    """
    return [r for r in rows if r["image"] == label
            and (benchmark is None or r["image_benchmark"] == benchmark)
            and (model is None or r["image_model"] == model)
            and (variant is None or r.get("image_variant") == variant)
            and (side is None or r.get("image_side") == side)]


def grid(title: str, lines: list[tuple[str, dict[str, list[dict]]]],
         cell: Callable[[list[dict]], str], viewers: list[str]) -> list[str]:
    """Render one block: a row per image, a column per viewer.

    Args:
        title: Heading of the block.
        lines: Row label and the answers about its image per viewer.
        cell: Function of one viewer's answers to the cell text.
        viewers: Models shown the images, in column order.
    Returns:
        The block's lines.
    """
    out = [f"\n{title}", f"{'':22} | " + " ".join(
        f"{SOURCES[v]:>12}" for v in viewers)]
    for label, answers in lines:
        out.append(f"{label:22.22} | " + " ".join(
            f"{cell(answers.get(v, [])):>12}" for v in viewers))
    return out


def ablations(main: dict[str, list[dict]],
              ablated: dict[str, list[dict]]) -> str:
    """Set each ablation image next to what it is compared with.

    Args:
        main: Answer rows of the main images per viewer.
        ablated: Answer rows of the ablation images per viewer.
    Returns:
        The ensemble, skill-conditioned and resolution blocks. Cells read
        "vocabulary% task-named/5" (see measure) or "n<noise>/5 m<rating>"
        (see looks).
    """
    viewers = [v for v in SOURCES if v in ablated and v in main]
    lines = []
    members = "+".join(ENSEMBLE_MEMBERS)
    for benchmark in BENCHMARKS:
        rows = [("ensemble", {v: pick(ablated[v], "skillpng", benchmark,
                                       members, "ensemble")
                               for v in viewers})]
        rows += [(f"{SOURCES[m]} own", {v: pick(main[v], "skillpng",
                                                 benchmark, m)
                                         for v in viewers})
                 for m in ENSEMBLE_MEMBERS]
        rows += [(f"{c} 256", {v: pick(ablated[v], c, side=256)
                               for v in viewers}) for c in CONTROLS]
        if rows[0][1][viewers[0]]:
            lines += grid(f"Ensemble, {BENCHMARKS[benchmark]}: vocabulary%"
                          " (bare) task named/5 (under its prompt)", rows,
                          lambda r, b=benchmark: measure(r, b, b), viewers)

    for benchmark in BENCHMARKS:
        image = {v: pick(ablated[v], "skillpng", benchmark, OPTIMIZER_MODEL,
                         "conditioned") for v in viewers}
        if not image[viewers[0]]:
            continue
        plain = {v: pick(main[v], "skillpng", benchmark, OPTIMIZER_MODEL)
                 for v in viewers}
        controls = [(f"{c} 512", {v: pick(ablated[v], c, side=512)
                                  for v in viewers}) for c in CONTROLS]
        for context, where in ((benchmark, "its prompt"),
                               (benchmark + SKILLED, "its prompt + skill")):
            rows = [("trained under skill", image)]
            if context == benchmark:
                rows.append(("plain Nem image", plain))
            lines += grid(f"SkillOpt + Skill.png, {BENCHMARKS[benchmark]}:"
                          f" vocabulary% (bare) task named/5 (under {where})",
                          rows + controls,
                          lambda r, b=benchmark, c=context: measure(r, b, c),
                          viewers)

    for benchmark in BENCHMARKS:
        rows = []
        for size in RESOLUTIONS:
            if size == RESOLUTION:
                answers = {v: pick(main[v], "skillpng", benchmark,
                                   RESOLUTION_MODEL) for v in viewers}
            else:
                answers = {v: pick(ablated[v], "skillpng", benchmark,
                                   RESOLUTION_MODEL, f"res{size}")
                           for v in viewers}
            rows.append((f"{size}px", answers))
        if not any(rows[0][1].values()):
            continue
        for size in RESOLUTIONS:
            rows.append((f"noise {size}px", {v: pick(ablated[v], "noise",
                                                     side=size)
                                             for v in viewers}))
        lines += grid(f"Resolution, {MODELS[RESOLUTION_MODEL]}"
                      f" {BENCHMARKS[benchmark]}: vocabulary% (bare) task"
                      " named/5 (under its prompt)", rows,
                      lambda r, b=benchmark: measure(r, b, b), viewers)
        lines += grid(f"Resolution, {MODELS[RESOLUTION_MODEL]}"
                      f" {BENCHMARKS[benchmark]}: said noise/5, mean"
                      " meaning (bare)", rows, looks, viewers)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--examples", type=int, default=0,
                        help="open answers printed per image and question")
    args = parser.parse_args()

    answers = load(args.runs_dir)
    heads = {"task": [SHORT[key][:8] for key in TASKS],
             "kind": [key[:8] for key in KINDS]}
    for model, rows in answers.items():
        print(f"\n{MODELS.get(model, model)}"
              f" ({len(rows)} answers)")
        print(f"{'context':10} {'image':12} | task: "
              + " ".join(f"{h:>3.3}" for h in heads["task"]) + " ?? | kind: "
              + " ".join(f"{h:>3.3}" for h in heads["kind"])
              + " ?? | meaning")
        for context, label, picked in groups(rows):
            ratings = [r["answer"] for r in picked
                       if r["question_id"] == "meaning" and r["answer"]]
            rating = f"{mean(ratings):.1f}" if ratings else "--"
            ctx = BENCHMARKS.get(context, context)
            print(f"{ctx:10.10} {label:12.12} | task: "
                  f"{counts(picked, 'task', TASKS)} | kind: "
                  f"{counts(picked, 'kind', KINDS)} | {rating:>7}")
    print(cross(answers))
    ablated = load(args.runs_dir, "descriptions_ablations.jsonl")
    if ablated:
        print(ablations(answers, ablated))
    if not args.examples:
        return
    for model, rows in answers.items():
        for context, label, picked in groups(rows):
            for question_id, (_, options) in QUESTIONS.items():
                if options or question_id == "meaning":
                    continue
                answers = [r["output"] for r in picked
                           if r["question_id"] == question_id][:args.examples]
                print(f"\n## {model} | {context} | {label} | {question_id}")
                for answer in answers:
                    print("- " + " ".join(answer.split())[:400])


if __name__ == "__main__":
    main()
