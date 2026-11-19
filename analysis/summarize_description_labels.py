"""Summarize the judge's labels from label_descriptions.py.

First the grids the paper reports, over the open answers, one row per model
an image was optimized for and one column per viewer:

  noise   how often the judge found the answer calling the image noise or
          corrupted
  task    how often the subject the answer ties the image to is the image's
          own benchmark; for an image with no benchmark of its own, the mean
          of that share over the four benchmarks' subjects

Each grid ends in a Random Image row, the uniform-noise control, and an
Ensemble row, the ensemble images of descriptions_ablations.jsonl. Bare, each
image is asked about without a system prompt. Under --context system, each
Skill.png sits under its own benchmark's system prompt and the noise image
under each of the four.

Then per viewer, one line per image condition: its own Skill.png, the other
models' Skill.png pooled, uniform noise and flat gray. Three tables:

  calls   over the open answers, how often the judge found the answer
          calling the image noise, blank, or corrupted, claiming to read
          text, or naming concrete things in it
  looks   over the describe answers, how the judge found the image
          described, as shares of LOOKS
  topic   as the task grid, by condition

Then the closed questions read by the judge against parse(): how often they
agree, and the kind answers by the judge's reading. Last, the judge against
the second annotator of runs/answers/agreement.jsonl, over both contexts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from analyze_runs import RUNS_DIR
from describe_skillpng import MAIN_MODELS
from label_descriptions import LOOKS, row_key

CONDITIONS = ("own", "other", "noise", "gray")
OPEN = ("describe", "read", "instruct", "prompt", "words")
CLOSED = ("task", "kind", "meaning")
TOPIC_OF = {"t1": "travel_code", "searchqa": "trivia",
            "livemathematicianbench": "math", "traject": "tools"}
SHORT = {"qwen3.5-4B": "Q4B", "qwen3.5-9B": "Q9B", "gemma-4-E4B-it": "G4B",
         "nemotron-3-nano-omni-30B-A3B-reasoning": "Nemo"}
LETTERS = "ABCDEFGH"
AGREEMENT_FILE = "agreement.jsonl"


def load(runs_dir: Path, model: str,
         name: str = "descriptions") -> list[dict]:
    """Read one viewer's labeled answers.

    Args:
        runs_dir: Directory holding runs/answers.
        model: Viewer model.
        name: Stem of the answers file.
    Returns:
        One row per labeled answer, the latest label kept.
    """
    with open(runs_dir / "answers" / model / f"{name}_labeled.jsonl") as file:
        rows = {row_key(row): row for row in map(json.loads, file)
                if row["label"] is not None}
    return list(rows.values())


def condition(row: dict) -> str:
    """Name the image condition from its viewer's side.

    Args:
        row: A labeled answer.
    Returns:
        One of CONDITIONS.
    """
    if row["image"] != "skillpng":
        return row["image"]
    return "own" if row["image_model"] == row["model"] else "other"


def in_context(row: dict, context: str) -> bool:
    """Check that an answer was asked in the grids' context.

    Args:
        row: A labeled answer.
        context: bare or system.
    Returns:
        True when bare and asked bare, or when system and asked under a
        system prompt its image is shown under in the grids.
    """
    if context == "bare":
        return row["context"] == "bare"
    if row["image"] == "skillpng":
        return row["context"] == row["image_benchmark"]
    return row["context"] in TOPIC_OF


def noisy(label: dict) -> bool:
    """Check whether a label calls the image noise or corrupted.

    Args:
        label: The judge's fields of one answer.
    Returns:
        True when either call is made.
    """
    return label["calls_noise"] or label["calls_corrupted"]


def on_task(rows: list[dict]) -> float:
    """Share of answers tying the image to its benchmark's subject.

    Args:
        rows: Labeled open answers.
    Returns:
        The share, for an image with no benchmark of its own the mean over
        the subjects of the benchmarks it is asked under, or of all four.
    """
    def benchmark(row: dict) -> str | None:
        """Name the benchmark an answer's image is judged against.

        Args:
            row: A labeled answer.
        Returns:
            The image's benchmark, else its system prompt's, else None.
        """
        if row["image"] == "skillpng":
            return row["image_benchmark"]
        return row["context"] if row["context"] in TOPIC_OF else None

    tied = [r for r in rows if benchmark(r)]
    if tied:
        return sum(r["label"]["topic"] == TOPIC_OF[benchmark(r)]
                   for r in tied) / len(tied)
    return sum(sum(r["label"]["topic"] == topic for r in rows) / len(rows)
               for topic in TOPIC_OF.values()) / len(TOPIC_OF)


def judged_answer(row: dict) -> str | int | None:
    """Turn the judge's choice into the option key or rating it means.

    Args:
        row: A labeled closed answer.
    Returns:
        The option key, the 1-10 rating, or None when the judge found none.
    """
    choice = row["label"]["choice"].strip().upper().rstrip(").")
    if row["options"]:
        index = LETTERS.find(choice[:1]) if len(choice) == 1 else -1
        return row["options"][index] if 0 <= index < len(row["options"]) else None
    return int(choice) if choice.isdigit() and 1 <= int(choice) <= 10 else None


def share(rows: list, test: Callable[[object], bool]) -> str:
    """Format the share of rows passing a test.

    Args:
        rows: Labeled answers or their labels.
        test: Function of a row to bool.
    Returns:
        "p%" or "-" for no rows.
    """
    return f"{100 * sum(map(test, rows)) / len(rows):.0f}%" if rows else "-"


def kappa(pairs: list[tuple[bool, bool]]) -> tuple[int, float]:
    """Count agreements and Cohen's kappa of two binary annotators.

    Args:
        pairs: One pair of labels per item.
    Returns:
        How many pairs agree, and kappa.
    """
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    first = sum(a for a, _ in pairs) / n
    second = sum(b for _, b in pairs) / n
    chance = first * second + (1 - first) * (1 - second)
    return agree, (agree / n - chance) / (1 - chance)


def print_table(title: str, heads: list[str], lines: list[list]) -> None:
    """Print one aligned table.

    Args:
        title: Line above it.
        heads: Column heads.
        lines: Rows of cells.
    """
    print(f"\n== {title}")
    widths = [max(len(str(c)) for c in col) for col in zip(heads, *lines, strict=True)]
    for line in [heads, *lines]:
        print("  ".join(str(c).rjust(w) for c, w in zip(line, widths, strict=True)))


def print_grids(runs_dir: Path, models: list[str], context: str) -> None:
    """Print the noise and task grids of the images the paper reports.

    Args:
        runs_dir: Directory holding runs/answers.
        models: Viewers, also the models whose Skill.png make the rows.
        context: bare or system.
    """
    rows = [r for model in models for r in load(runs_dir, model)
            + load(runs_dir, model, "descriptions_ablations")
            if r["question_id"] in OPEN and in_context(r, context)
            and r.get("image_variant") in (None, "ensemble")]
    sources = [(SHORT[m], lambda r, m=m: r["image"] == "skillpng"
                and r.get("image_variant") is None and r["image_model"] == m)
               for m in models]
    sources += [("Random Image", lambda r: r["image"] == "noise"),
                ("Ensemble", lambda r: r.get("image_variant") == "ensemble")]
    heads = ["optimized for", *(SHORT[m] for m in models)]

    def cells(viewer: str, pick: Callable[[dict], bool]) -> list[dict]:
        """Select one cell's answers.

        Args:
            viewer: Model shown the image.
            pick: Test of a row for the grid row's image.
        Returns:
            The matching answers.
        """
        return [r for r in rows if r["model"] == viewer and pick(r)]

    print_table(f"noise or corrupted, open answers ({context})", heads,
                [[name] + [share(cells(v, pick), lambda r: noisy(r["label"]))
                           for v in models] for name, pick in sources])
    print_table(f"tied to the image's task, open answers ({context})", heads,
                [[name] + [f"{100 * on_task(here):.0f}%" if here else "-"
                           for v in models for here in [cells(v, pick)]]
                 for name, pick in sources])
    print("answers per cell: " + ", ".join(
        f"{name} {len(cells(models[0], pick))}" for name, pick in sources))


def print_agreement(runs_dir: Path) -> None:
    """Print how the judge agrees with the second annotator.

    Args:
        runs_dir: Directory holding runs/answers.
    """
    path = runs_dir / "answers" / AGREEMENT_FILE
    if not path.exists():
        return
    with open(path) as file:
        second = [json.loads(line) for line in file]
    judged = {(model, row_key(r)): r["label"]
              for model in {row["model"] for row in second}
              for r in load(runs_dir, model)}
    labels = [(judged[(row["model"], row_key(row))], row) for row in second]
    lines = []
    for name, ours, theirs in (
            ("noise or corrupted", noisy, "noise_or_corrupted"),
            ("text or objects",
             lambda label: label["reads_text"] or bool(label["objects"]),
             "text_or_objects")):
        agree, value = kappa([(ours(label), row[theirs])
                              for label, row in labels])
        lines.append([name, f"{agree}/{len(labels)}", f"{value:.2f}"])
    print_table(f"judge against {second[0]['annotator']}, open answers",
                ["label", "agree", "kappa"], lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--models", nargs="+", default=list(MAIN_MODELS))
    parser.add_argument("--context", default="bare",
                        choices=("bare", "system"))
    args = parser.parse_args()

    print_grids(args.runs_dir, args.models, args.context)

    rows = []
    for model in args.models:
        for row in load(args.runs_dir, model):
            if (row["context"] == "bare") == (args.context == "bare"):
                row["condition"] = condition(row)
                rows.append(row)

    def cell(model: str, name: str,
             questions: tuple[str, ...]) -> list[dict]:
        """Select one viewer's answers about one image condition.

        Args:
            model: Viewer model.
            name: One of CONDITIONS.
            questions: Question ids to keep.
        Returns:
            The matching answers.
        """
        return [r for r in rows if r["model"] == model
                and r["condition"] == name and r["question_id"] in questions]

    lines = []
    for model in args.models:
        for name in CONDITIONS:
            here = cell(model, name, OPEN)
            labels = [r["label"] for r in here]
            lines.append([SHORT[model], name, len(here)] + [
                share(labels, lambda label, k=k: label[k]) for k in
                ("calls_noise", "calls_blank", "calls_corrupted",
                 "reads_text")] + [
                share(labels, lambda label: bool(label["objects"])),
                share(labels, lambda label: label["cannot_see"])])
    print_table(f"calls, open answers ({args.context})",
                ["viewer", "image", "n", "noise", "blank", "corrupted",
                 "reads_text", "objects", "cannot_see"], lines)

    lines = []
    for model in args.models:
        for name in CONDITIONS:
            here = [r["label"]["looks"] for r in cell(model, name, ("describe",))]
            lines.append([SHORT[model], name, len(here)]
                         + [share(here, lambda look, k=k: look == k) for k in LOOKS])
    print_table(f"looks, describe answers ({args.context})",
                ["viewer", "image", "n", *LOOKS], lines)

    lines = []
    for model in args.models:
        for name in CONDITIONS:
            here = cell(model, name, OPEN)
            lines.append([SHORT[model], name, len(here),
                          f"{100 * on_task(here):.0f}%" if here else "-"])
    print_table(f"topic is the image's benchmark, open answers ({args.context})",
                ["viewer", "image", "n", "own topic"], lines)

    lines = []
    for model in args.models:
        for question_id in CLOSED:
            here = [r for r in rows if r["model"] == model
                    and r["question_id"] == question_id]
            judged = [judged_answer(r) for r in here]
            agree = sum(r["answer"] == j for r, j in zip(here, judged, strict=True))
            lines.append([SHORT[model], question_id, len(here),
                          sum(r["answer"] is None for r in here),
                          sum(j is None for j in judged),
                          f"{100 * agree / len(here):.0f}%" if here else "-"])
    print_table(f"closed questions, parser against judge ({args.context})",
                ["viewer", "question", "n", "parser none", "judge none",
                 "agree"], lines)

    lines = []
    kinds = ("noise", "pattern", "text", "drawing", "photo")
    for model in args.models:
        for name in CONDITIONS:
            here = Counter(judged_answer(r) for r in cell(model, name, ("kind",)))
            total = sum(here.values())
            lines.append([SHORT[model], name, total] + [
                f"{100 * here[k] / total:.0f}%" if total else "-"
                for k in (*kinds, None)])
    print_table(f"kind, as the judge read the choice ({args.context})",
                ["viewer", "image", "n", *kinds, "unread"], lines)

    print_agreement(args.runs_dir)


if __name__ == "__main__":
    main()
