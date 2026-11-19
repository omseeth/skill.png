"""Read a T1 eval the way upstream reads it, offline.

evaluate.py writes one JSONL per (split, condition) and averages one number.
Every per-turn component rides in each row's meta, and this is what turns
them into upstream's table:

    python -m skillpng.benchmarks.t1.t1_report runs/t1/.../eval/test-none.jsonl \\
        --against runs/t1/.../eval/test-skillpng.jsonl

Upstream's metrics are computed per domain with upstream's names and formulas
(evaluation/compute_aggregate_metrics.py), then macro-averaged over domains,
which is the paper's "Avg.":

- tool_calling_micro_* and tool_param_micro_*: precision, recall, F1 and
  accuracy from tp/fp/fn summed over the domain's turns where gold and
  generated calls both exist, accuracy being tp / (tp + fp + fn);
- code_success_rate: 1 - the mean non_executable per source file, then the
  mean over files;
- cache_summary_exact_match_avg: the mean over turns;
- exec_only_*: the same but code success, over turns with non_executable != 1.

SacreBLEU is not computed. Beside upstream's numbers it always prints what
they hide:

- the headline, and code success as a plain per-item mean: on a sampled split
  upstream's per-file grouping weights files unevenly;
- the share of turns where the generation made no calls, which upstream's
  P/R/F1 silently leave out (a turn without calls is not a false negative);
- extraction failures (no <CODE> block), parse failures, timeouts and worker
  deaths;
- the breakdown by gold_kind, whose do-nothing floor differs by kind;
- with --against, the paired difference on item_id with a bootstrap 95% CI.

None of this is comparable to the paper's tables: a sampled subset, the
zero-shot prompt, and no leak-dependent turns. See t1.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HEADLINE = "cache_em"
PAIRED_METRICS = ("cache_em", "tool_f1", "code_success")
BOOTSTRAP = 10000
CONFIDENCE = 0.95
KINDS = ("tools", "seek_only", "seek_plus_tools", "cache_read_only")
UPSTREAM = (
    "code_success_rate",
    "cache_summary_exact_match_avg",
    "tool_calling_micro_precision",
    "tool_calling_micro_recall",
    "tool_calling_micro_accuracy",
    "tool_calling_micro_f1",
    "tool_param_micro_precision",
    "tool_param_micro_recall",
    "tool_param_micro_accuracy",
    "tool_param_micro_f1",
)
EXEC_ONLY = tuple(f"exec_only_{name}" for name in UPSTREAM[1:])
# Upstream's order within each block, as its CSV writes it.
FIELDS = UPSTREAM + EXEC_ONLY


def read_eval(path: str | Path) -> dict[str, dict[str, object]]:
    """Read one eval's per-item rows, keyed by item id.

    Args:
        path: Path to a per-item JSONL written by evaluate.py.

    Returns:
        Item id to the row's meta, with the headline score folded in.
    """
    rows: dict[str, dict[str, object]] = {}

    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        meta = dict(row.get("meta") or {})

        if "non_executable" not in meta:
            raise ValueError(
                f"{path}: row {row.get('item_id')!r} carries no T1 components. They ride in "
                "PredictionRow.meta; is this a T1 eval?"
            )

        meta["score"] = row.get("score", 0.0)
        rows[str(row["item_id"])] = meta

    if not rows:
        raise ValueError(f"{path}: no rows")

    return rows


def upstream_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    """Upstream's aggregate metrics over one domain's turns.

    Args:
        rows: The domain's per-turn metas.

    Returns:
        FIELDS to their values.
    """
    files: dict[tuple[object, ...], list[float]] = {}

    for row in rows:
        key = (row["domain"], row["hf_split"], row["file_index"])
        files.setdefault(key, []).append(float(row["non_executable"]))  # type: ignore[arg-type]

    executable = [row for row in rows if row["non_executable"] != 1]
    out = {"code_success_rate": _mean([1.0 - _mean(values) for values in files.values()])}
    out.update(_accumulate(rows, ""))
    out.update(_accumulate(executable, "exec_only_"))

    return out


def summarize(rows: dict[str, dict[str, object]]) -> dict[str, object]:
    """Aggregate one condition: upstream's table plus everything beside it.

    Args:
        rows: Item id to its meta, from read_eval.

    Returns:
        Per-domain and macro upstream metrics, and the side columns.
    """
    values = list(rows.values())
    domains: dict[str, list[dict[str, object]]] = {}

    for row in values:
        domains.setdefault(str(row["domain"]), []).append(row)

    per_domain = {
        domain: {"n": len(group), **upstream_metrics(group)}
        for domain, group in sorted(domains.items())
    }
    macro = {name: _mean([float(m[name]) for m in per_domain.values()]) for name in FIELDS}

    return {
        "n": len(values),
        "headline": _mean([float(row["score"]) for row in values]),  # type: ignore[arg-type]
        "per_domain": per_domain,
        "macro": macro,
        "code_success_per_item": _mean([float(row["code_success"]) for row in values]),  # type: ignore[arg-type]
        "no_calls": _share(values, lambda row: bool(row["no_calls"])),
        "extraction_failures": _share(values, lambda row: row["n_blocks"] == 0),
        "parse_failures": _share(values, lambda row: bool(row["parse_error"])),
        "timeouts": _share(values, lambda row: bool(row["timeout"])),
        "not_executed": _share(values, lambda row: not row["executed"]),
        "by_gold_kind": {
            kind: _side(group)
            for kind in KINDS
            if (group := [row for row in values if row["gold_kind"] == kind])
        },
    }


def paired_delta(
    base: dict[str, dict[str, object]],
    treatment: dict[str, dict[str, object]],
    metric: str = HEADLINE,
    draws: int = BOOTSTRAP,
    seed: int = 0,
) -> dict[str, object]:
    """The per-item difference between two conditions, with a bootstrap interval.

    Args:
        base: Item id to meta for the baseline condition.
        treatment: Item id to meta for the condition being compared.
        metric: Per-item component the difference is taken on.
        draws: Number of bootstrap resamples.
        seed: Seed making the interval reproducible.

    Returns:
        The paired mean difference, its interval, and the win and loss counts.
    """
    shared = sorted(set(base) & set(treatment))

    if not shared:
        raise ValueError("the two evals share no item ids; they are not the same split")

    deltas = [
        float(treatment[i][metric]) - float(base[i][metric])  # type: ignore[arg-type]
        for i in shared
    ]
    rng = random.Random(seed)
    means = sorted(
        _mean([deltas[rng.randrange(len(deltas))] for _ in deltas]) for _ in range(draws)
    )

    return {
        "metric": metric,
        "n_paired": len(shared),
        "n_base_only": len(set(base) - set(treatment)),
        "n_treatment_only": len(set(treatment) - set(base)),
        "base": _mean([float(base[i][metric]) for i in shared]),  # type: ignore[arg-type]
        "treatment": _mean([float(treatment[i][metric]) for i in shared]),  # type: ignore[arg-type]
        "delta": _mean(deltas),
        "ci_low": means[int((1 - CONFIDENCE) / 2 * draws)],
        "ci_high": means[min(draws - 1, int((1 + CONFIDENCE) / 2 * draws))],
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def build_report(
    baseline: str | Path,
    against: str | Path | None = None,
    metric: str = HEADLINE,
    draws: int = BOOTSTRAP,
    seed: int = 0,
) -> dict[str, object]:
    """Summarize one eval, and the paired difference when a second is given.

    Args:
        baseline: Per-item JSONL of the baseline condition.
        against: Per-item JSONL of the condition compared to it, or None.
        metric: Component the paired difference is taken on.
        draws: Number of bootstrap resamples.
        seed: Seed making the interval reproducible.

    Returns:
        The report mapping render and the JSON dump are built from.
    """
    base = read_eval(baseline)
    conditions = {str(baseline): summarize(base)}
    report: dict[str, object] = {"conditions": conditions}

    if against is not None:
        treatment = read_eval(against)
        conditions[str(against)] = summarize(treatment)
        report["paired"] = paired_delta(base, treatment, metric, draws, seed)

    return report


def render(report: dict[str, object]) -> str:
    """Format a report as the plain text tables a run is read from.

    Args:
        report: The report mapping built by build_report.

    Returns:
        The rendered tables.
    """
    short = (
        "cache_em",
        "tool_P",
        "tool_R",
        "tool_F1",
        "tool_Acc",
        "param_P",
        "param_R",
        "param_F1",
        "param_Acc",
    )
    lines: list[str] = []

    for label, summary in report["conditions"].items():  # type: ignore[attr-defined]
        lines.append(f"== {label}  (n={summary['n']})")

        for title, prefix, head in (
            ("upstream metrics", "", "code_succ"),
            ("exec_only_*", "exec_only_", ""),
        ):
            names = ([] if prefix else ["code_success_rate"]) + [
                f"{prefix}{name}" for name in UPSTREAM[1:]
            ]
            header = ([head] if head else []) + list(short)
            lines.append(f"   {title}:")
            lines.append(f"     {'domain':<22s} {'n':>5s} " + " ".join(f"{h:>9s}" for h in header))
            table = {**summary["per_domain"], "macro avg": {"n": summary["n"], **summary["macro"]}}

            for domain, metrics in table.items():
                cells = " ".join(f"{float(metrics[name]):>9.4f}" for name in names)
                lines.append(f"     {domain:<22s} {metrics['n']:>5d} {cells}")

        lines.append(f"   headline (mean score)        {summary['headline']:.4f}")
        lines.append(f"   code success, per item       {summary['code_success_per_item']:.4f}")

        for name, title in (
            ("no_calls", "no tool calls"),
            ("extraction_failures", "no <CODE> block"),
            ("parse_failures", "code did not parse"),
            ("timeouts", "timed out"),
            ("not_executed", "not executed to an outcome"),
        ):
            share = summary[name]
            lines.append(
                f"   {title:<28s} {share['share']:.4f}  ({share['count']} of {summary['n']})"
            )

        lines.append("   by gold_kind:")
        lines.append(
            f"     {'kind':<18s} {'n':>5s} {'headline':>9s} {'cache_em':>9s} "
            f"{'code_succ':>9s} {'no_calls':>9s}"
        )

        for kind, side in summary["by_gold_kind"].items():
            lines.append(
                f"     {kind:<18s} {side['n']:>5d} {side['headline']:>9.4f} "
                f"{side['cache_em']:>9.4f} {side['code_success']:>9.4f} {side['no_calls']:>9.4f}"
            )

        lines.append("")

    delta = report.get("paired")

    if isinstance(delta, dict):
        lines.append(
            f"== paired delta on {delta['metric']} (n={delta['n_paired']}, joined on item_id)"
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


def _accumulate(rows: list[dict[str, object]], prefix: str) -> dict[str, float]:
    """Upstream's accumulate_metrics without SacreBLEU.

    Args:
        rows: Per-turn metas.
        prefix: "" or "exec_only_".

    Returns:
        The cache match mean and the micro tool and parameter metrics.
    """
    out = {f"{prefix}cache_summary_exact_match_avg": _mean([float(r["cache_em"]) for r in rows])}  # type: ignore[arg-type]

    for kind, name in (("tool", "tool_calling"), ("param", "tool_param")):
        tp, fp, fn = (
            sum(int(r[f"{kind}_{c}"]) for r in rows if r[f"{kind}_{c}"] is not None)  # type: ignore[call-overload]
            for c in ("tp", "fp", "fn")
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[f"{prefix}{name}_micro_precision"] = precision
        out[f"{prefix}{name}_micro_recall"] = recall
        out[f"{prefix}{name}_micro_accuracy"] = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        out[f"{prefix}{name}_micro_f1"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    return out


def _side(rows: list[dict[str, object]]) -> dict[str, object]:
    """The side columns for one group of turns.

    Args:
        rows: Per-turn metas of the group.

    Returns:
        Its size and the means of the headline, cache match, code success
        and no-call share.
    """
    return {
        "n": len(rows),
        "headline": _mean([float(r["score"]) for r in rows]),  # type: ignore[arg-type]
        "cache_em": _mean([float(r["cache_em"]) for r in rows]),  # type: ignore[arg-type]
        "code_success": _mean([float(r["code_success"]) for r in rows]),  # type: ignore[arg-type]
        "no_calls": _mean([float(bool(r["no_calls"])) for r in rows]),
    }


def _share(rows: list[dict[str, object]], test: object) -> dict[str, object]:
    """How many rows pass a test, and what share of all rows that is.

    Args:
        rows: Per-turn metas.
        test: Predicate over one meta.

    Returns:
        The count and the share.
    """
    assert callable(test)
    count = sum(1 for row in rows if test(row))

    return {"count": count, "share": count / len(rows) if rows else 0.0}


def _mean(values: list[float]) -> float:
    """Average a list, returning zero for an empty one, as upstream's compute_avg.

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
    parser.add_argument("--metric", default=HEADLINE, choices=list(PAIRED_METRICS))
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
