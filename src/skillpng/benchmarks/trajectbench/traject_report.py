"""Read a TRAJECT-Bench eval the way it has to be read, offline.

evaluate.py writes one JSONL per (split, condition) and averages one number.
Everything else the adapter measures rides in each row's meta, and this is what
turns that into the table a run is actually read from:

    python -m skillpng.benchmarks.trajectbench.traject_report runs/.../eval/test-none.jsonl \\
        --against runs/.../eval/test-skillpng.jsonl

Three things it always prints, because leaving any of them out makes a number
easy to misread:

- Parse success, separately, and the headline conditioned on it. An optimized
  image has a cheap route to moving the score by degrading JSON compliance, and
  without this column that is indistinguishable from better tool selection.
- The breakdown by tool_count. Distractors are fixed at ten, so the gold to
  distractor ratio runs from 2:10 to 11:10 and difficulty varies by item.
- The paired difference, joined on item_id, with a bootstrap interval. The
  claim is a paired delta, not two independent intervals.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

COMPONENTS = (
    "parse_success",
    "tool_em",
    "tool_inclusion",
    "required_em",
    "optional_em",
    "usage_em",
    "full_em",
)
HEADLINE = "full_em"
BOOTSTRAP = 10000
CONFIDENCE = 0.95


def read_eval(path: str | Path) -> dict:
    """Read one eval's per-item rows, keyed by item id.

    Args:
        path: Path to a per-item JSONL written by evaluate.py.

    Returns:
        Item id to the row's meta, with the headline score folded in.
    """
    rows: dict = {}

    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        meta = dict(row.get("meta") or {})

        if not meta:
            raise ValueError(
                f"{path}: row {row.get('item_id')!r} carries no meta. The component metrics "
                "ride in PredictionRow.meta; this file predates that, so rerun the eval."
            )
        meta.setdefault("score", row.get("score", 0.0))
        rows[str(row["item_id"])] = meta

    if not rows:
        raise ValueError(f"{path}: no rows")

    return rows


def summarize(rows: dict, metric: str = HEADLINE) -> dict:
    """Aggregate one condition: components, strata, and the parsed-only headline.

    Args:
        rows: Item id to its meta, from read_eval.
        metric: Component treated as the headline.

    Returns:
        Overall means, the parsed-only headline, and the stratified tables.
    """
    values = list(rows.values())
    parsed = [row for row in values if row.get("parse_success", 0.0) > 0]

    return {
        "n": len(values),
        "overall": {name: _mean([row.get(name, 0.0) for row in values]) for name in COMPONENTS},
        "n_parsed": len(parsed),
        # The same headline over the rows that produced usable JSON, so a
        # shift in compliance cannot be mistaken for a shift in tool choice.
        "parsed_only": _mean([row.get(metric, 0.0) for row in parsed]),
        "by_tool_count": _strata(values, "tool_count", metric),
        "by_domain": _strata(values, "domain", metric),
        "by_unseen_values": _strata(values, "has_unseen_values", metric),
    }


def paired_delta(
    base: dict, treatment: dict, metric: str = HEADLINE, draws: int = BOOTSTRAP, seed: int = 0
) -> dict:
    """The per-item difference between two conditions, with a bootstrap interval.

    Args:
        base: Item id to meta for the baseline condition.
        treatment: Item id to meta for the condition being compared.
        metric: Component the difference is taken on.
        draws: Number of bootstrap resamples.
        seed: Seed making the interval reproducible.

    Returns:
        The paired mean difference, its interval, and the win and loss counts.
    """
    shared = sorted(set(base) & set(treatment))

    if not shared:
        raise ValueError("the two evals share no item ids; they are not the same split")

    deltas = [
        float(treatment[item_id].get(metric, 0.0)) - float(base[item_id].get(metric, 0.0))
        for item_id in shared
    ]
    rng = random.Random(seed)
    means = sorted(
        _mean([deltas[rng.randrange(len(deltas))] for _ in deltas]) for _ in range(draws)
    )
    low = means[int((1 - CONFIDENCE) / 2 * draws)]
    high = means[min(draws - 1, int((1 + CONFIDENCE) / 2 * draws))]

    return {
        "metric": metric,
        "n_paired": len(shared),
        "n_base_only": len(set(base) - set(treatment)),
        "n_treatment_only": len(set(treatment) - set(base)),
        "base": _mean([float(base[item_id].get(metric, 0.0)) for item_id in shared]),
        "treatment": _mean([float(treatment[item_id].get(metric, 0.0)) for item_id in shared]),
        "delta": _mean(deltas),
        "ci_low": low,
        "ci_high": high,
        "wins": sum(1 for delta in deltas if delta > 0),
        "losses": sum(1 for delta in deltas if delta < 0),
        "ties": sum(1 for delta in deltas if delta == 0),
    }


def render(report: dict) -> str:
    """Format a report as the plain text table the run is read from.

    Args:
        report: The report mapping built by build_report.

    Returns:
        The rendered table.
    """
    lines: list[str] = []

    for label, summary in report["conditions"].items():
        lines.append(f"== {label}  (n={summary['n']})")
        lines += [f"   {name:<16s} {summary['overall'][name]:.4f}" for name in COMPONENTS]
        lines.append(
            f"   {'(parsed only)':<16s} {summary['parsed_only']:.4f}  "
            f"over {summary['n_parsed']} of {summary['n']} rows"
        )

        for key, title in (
            ("by_tool_count", "tool_count"),
            ("by_domain", "domain"),
            ("by_unseen_values", "has_unseen_values"),
        ):
            lines.append(f"   by {title}:")
            lines += [
                f"     {name!s:<24s} n={stratum['n']:<5d} {stratum['mean']:.4f}"
                for name, stratum in summary[key].items()
            ]
        lines.append("")

    delta = report.get("paired")

    if delta:
        lines.append(
            f"== paired delta on {delta['metric']} "
            f"(n={delta['n_paired']}, joined on item_id)"
        )
        lines.append(f"   base       {delta['base']:.4f}")
        lines.append(f"   treatment  {delta['treatment']:.4f}")
        lines.append(
            f"   delta      {delta['delta']:+.4f}  "
            f"95% CI [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
        )
        lines.append(
            f"   items      {delta['wins']} better, {delta['losses']} worse, "
            f"{delta['ties']} unchanged"
        )

        if delta["n_base_only"] or delta["n_treatment_only"]:
            lines.append(
                f"   unpaired   {delta['n_base_only']} baseline-only, "
                f"{delta['n_treatment_only']} treatment-only, excluded"
            )

    return "\n".join(lines)


def build_report(
    baseline: str | Path,
    against: str | Path | None = None,
    metric: str = HEADLINE,
    draws: int = BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """Summarize one eval, and the paired difference when a second is given.

    Args:
        baseline: Per-item JSONL of the baseline condition.
        against: Per-item JSONL of the condition compared to it, or None.
        metric: Component treated as the headline.
        draws: Number of bootstrap resamples.
        seed: Seed making the interval reproducible.

    Returns:
        The report mapping render and the JSON dump are built from.
    """
    base = read_eval(baseline)
    report: dict = {
        "metric": metric,
        "conditions": {str(baseline): summarize(base, metric)},
    }

    if against is not None:
        treatment = read_eval(against)
        report["conditions"][str(against)] = summarize(treatment, metric)
        report["paired"] = paired_delta(base, treatment, metric, draws, seed)

    return report


def _strata(rows: list[dict], key: str, metric: str) -> dict:
    """Group rows by one meta field and average the metric inside each group.

    Args:
        rows: The condition's per-item metas.
        key: Meta field to group on.
        metric: Component to average.

    Returns:
        Group value to its count and mean, ordered by the group value.
    """
    groups: dict = {}

    for row in rows:
        groups.setdefault(row.get(key), []).append(float(row.get(metric, 0.0)))

    return {
        name: {"n": len(values), "mean": _mean(values)}
        for name, values in sorted(groups.items(), key=lambda pair: _order(pair[0]))
    }


def _order(value: object) -> tuple:
    """Sort numeric strata numerically and everything else as text.

    tool_count is the stratum that matters most here, and sorting it as text
    puts 11 before 2, which reads as a trend that is not there.

    Args:
        value: One group value from a meta field.

    Returns:
        A sort key placing numbers first and in numeric order.
    """
    if isinstance(value, bool):
        return (1, "", str(value))

    if isinstance(value, (int, float)):
        return (0, "", float(value))

    return (2, str(value), 0.0)


def _mean(values: list[float]) -> float:
    """Average a list, returning zero for an empty one.

    Args:
        values: Values to average.

    Returns:
        The mean, or 0.0 when there is nothing to average.
    """
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    """Command line entry point for the offline report.

    Args:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="per-item JSONL of the baseline condition")
    parser.add_argument("--against", default=None, help="per-item JSONL to compare against it")
    parser.add_argument("--metric", default=HEADLINE, choices=list(COMPONENTS))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="also write the report as JSON")
    args = parser.parse_args()
    report = build_report(args.baseline, args.against, args.metric, args.bootstrap, args.seed)
    print(render(report))

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
