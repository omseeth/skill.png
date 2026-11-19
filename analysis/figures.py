"""Figures over the skills the methods put in front of a model.

Draws one grouped bar chart of how many tokens each skill costs, per model:
the Skill.png image next to the LLM-Skill and the GPT-6 Luna SkillOpt skill.
Text skills are counted as in token_comparison and averaged over the four
benchmarks, leaving out benchmarks whose skill is the unchanged seed skill.
Error bars span one sample standard deviation over those benchmarks.
Then draws the non-reasoning Qwen3.5-4B Skill.png score at each image size
of the resolution ablation over the no-skill score, as analyze_runs tables
them: the mean over training seeds, in a band of one sample standard
deviation where there are several.

Last, draws a heatmap of the non-reasoning score change over No Skill of
the Skill.png optimized on Qwen3.5-4B and Gemma-4-E4B-it at once, with
unit-norm gradients, on every model and benchmark. Next to it are the GPT-6
Luna SkillOpt skill, the UltraBreak image and the Skill.png of Qwen3.5-4B,
as the analyze_runs transfer table reports them. A light gray band marks the
models each was optimized on, and arrows run from them to the models it was
injected into. Cells without a result yet are drawn empty.
Gemma-4-12B-it is left out.

Finally, draws the training loss of the non-reasoning Qwen3.5-9B Skill.png
on LiveMath for each Adam beta1 of the ablation, from the training runs of
the images the analyze_runs beta1 table reports.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path
from statistics import stdev

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from analyze_runs import (
    BASELINE_CONDITION,
    BENCHMARKS,
    BETA1_BENCHMARK,
    BETA1_MODEL,
    DISTRACTORS,
    ENSEMBLE_MEMBERS,
    ENSEMBLE_MODELS,
    INJECTED_CONDITION,
    REPORTED_BETA1,
    RESOLUTION,
    RESOLUTIONS,
    RUNS_DIR,
    SEED,
    TRAIN_SEED,
    TRANSFER_ENSEMBLE_CONDITION,
    TRANSFER_SOURCE,
    add_spread,
    at_resolution,
    build_table,
    find_failed,
    is_ensemble,
    is_transfer,
    load_failed_images,
    load_failed_skills,
    load_image_sizes,
    load_rows,
    resolution_seeds,
    select_beta1,
    select_ensemble,
    select_latest,
    select_resolutions,
    select_transfer,
    split_beta1,
)
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    PathPatch,
)
from matplotlib.transforms import Bbox, offset_copy
from token_comparison import (
    LLMSKILL_DIR,
    MODELS,
    OPTIMIZER,
    SKILLOPT_DIR,
    count_tokens,
    find_skills,
)

OUT = Path(__file__).resolve().parent / "figures" / "skill_tokens.png"
RESOLUTIONS_OUT = OUT.with_name("resolution_ablation.png")
ENSEMBLE_OUT = OUT.with_name("ensemble_heatmap.png")
BETA1_OUT = OUT.with_name("beta1_losses.png")

# Never drawn. Gemma-4-12B-it needs its own training recipe, so its
# Skill.png is not comparable to the other models.
EXCLUDED_MODELS = {"gemma-4-12B-it"}

# Image tokens the language model reads at each model's fed resolution, as
# the HF backend measures them (see the README). vLLM result rows record
# Nemotron's 1024 unmerged patches instead, so they are not read here.
IMAGE_TOKENS = {
    "qwen3.5-4B": 64,
    "qwen3.5-9B": 64,
    "gemma-4-E4B-it": 256,
    "nemotron-3-nano-omni-30B-A3B-reasoning": 256,
}

# Shorter skills are the seed skill, so no skill was learned there.
MIN_SKILL_TOKENS = 30

# Bar key to legend name and colour, in bar order within a model's group.
BARS = {
    "skillpng": ("Skill.png", "#9F1FBF"),
    "llmskill": ("LLM-Skill", "#80304A"),
    "skillopt": ("SkillOpt", "#C77272"),
}
BAR_WIDTH = 0.16
# Share of the bar width the rounded top corners take.
ROUNDING = 0.3
# Distance between the centres of neighbouring model groups.
GROUP_SPACING = 0.75
# Tick labels of the models whose names are too long for one line.
LABELS = {
    "gemma-4-E4B-it": "Gemma-4-\nE4B-it",
    "nemotron-3-nano-omni-30B-A3B-reasoning": "Nemotron-3-Nano-\nOmni-30B-A3B",
}
# Colour, stroke and cap size of the standard deviation error bars, in
# points.
ERROR_COLOR = "#3A3A3A"
ERROR_WIDTH = 0.8
ERROR_CAP = 2
FIGSIZE = (4.4, 2.6)
# Every figure is set in Times Roman.
plt.rcParams["font.family"] = "Times"
# Math, as log ticks and Greek letters, is set in STIX, which matches Times.
plt.rcParams["mathtext.fontset"] = "stix"
FONT_SIZE = 8
DPI = 300
# Distance of a legend above the axes to them, in font sizes. It adds the
# default border pad to the default axes pad, as the legend has none, so the
# crop meets its text.
LEGEND_PAD = 0.9
# Blank margin above every figure, in saved pixels. The other sides have
# none.
TOP_MARGIN = 15

# Line style of each benchmark in the resolution figure, in legend order.
RESOLUTION_STYLES = {"searchqa": "-", "t1": ":"}
BASELINE_COLOR = "#8C8C8C"
# Fill of the seed spread band, the paper's skillbg.
SPREAD_COLOR = "#E6CDF6"
LINE_WIDTH = 1.5
MARKER_SIZE = 4

# Colour scale of the ensemble heatmap, from loss to gain, in the paper's
# BrickRed and ForestGreen as macOS renders them in sRGB. Both arms step
# through the same OKLab lightness around a neutral gray at no change.
# BrickRed ends the loss arm, ForestGreen sits at 0.8 of the gain arm, where
# the lightness meets its own, and a darker shade of it ends the arm.
DIVERGING = ("#a13127", "#b55a4e", "#c68074", "#d6a49b", "#e4c9c3",
             "#f0efec", "#c0d7c4", "#8fc09e", "#5ba878", "#009053",
             "#007040")
# The scale's limits are the largest change rounded up to a multiple of this.
DELTA_STEP = 5
# Cell key to column name of the methods each benchmark of the heatmap
# shows: Qwen3.5-4B's SkillOpt skill, UltraBreak image and Skill.png, then
# the ensemble image, in the transfer table's order.
HEATMAP_METHODS = {
    "transfer_skillopt": "SkillOpt",
    "transfer_ultrabreak": "UltraBreak",
    "transfer": "Skill.png",
    TRANSFER_ENSEMBLE_CONDITION: "Ensemble",
}
# Models the skill or image of each method was optimized on. Lines run from
# them to the models it was injected into.
HEATMAP_SOURCES = {
    "transfer_skillopt": (TRANSFER_SOURCE,),
    "transfer_ultrabreak": (TRANSFER_SOURCE,),
    "transfer": (TRANSFER_SOURCE,),
    TRANSFER_ENSEMBLE_CONDITION: ENSEMBLE_MEMBERS,
}
# Row labels of the heatmap, where Nemotron's name takes three lines to keep
# the label column narrow.
HEATMAP_LABELS = {
    **LABELS,
    "nemotron-3-nano-omni-30B-A3B-reasoning":
        "Nemotron-3-\nNano-Omni-\n30B-A3B",
}
# Gap between neighbouring cells of a row and extra gap between benchmarks,
# in cell widths.
CELL_GAP = 0.06
BENCHMARK_GAP = 0.3
# Gap between neighbouring rows, wide enough for a plus sign, in row heights.
ROW_GAP = 0.22
# Share of the cell width the rounded corners take.
CELL_ROUNDING = 0.08
# Relative luminance at which white and dark text contrast equally.
DARK_CELL = 0.18
INK = "#0b0b0b"
# Fill and outline of a cell whose result is still missing.
EMPTY_FILL = "white"
EMPTY_EDGE = "#c4c4c4"
EMPTY_EDGE_WIDTH = 0.6
# Light gray band behind the cells of the models a skill or image was
# optimized on.
BAND = "#d6d6d6"
# Arrows from a model to the next one a skill or image is injected into:
# colour, stroke, head size and gap to the cells, in points. Then size and
# stroke of the plus joining an ensemble's models, in points.
LINK = "#9a9a9a"
LINK_WIDTH = 1
ARROW_HEAD = 6
ARROW_PAD = 1
PLUS_SIZE = 5
PLUS_WIDTH = 1
# Height of the benchmark names' rule above the axes, in points.
HEADER_OFFSET = 15
# Gives columns of 37 pt and rows of 31 pt, just wide enough for the
# UltraBreak header.
HEATMAP_FIGSIZE = (11.2, 2.45)

# Line colour, line style and legend name of each beta1 in the loss figure.
# The reported value wears the Skill.png purple, the others recede in gray.
BETA1_STYLES = {
    0.1: ("#7A7A7A", ":", r"$\beta_1 = 0.1$"),
    0.5: (BARS["skillpng"][1], "-", r"$\beta_1 = 0.5$ (ours)"),
    0.9: ("#3A3A3A", "--", r"$\beta_1 = 0.9$"),
}


def average_tokens(table: list[dict]) -> tuple[dict, dict]:
    """Average each skill's token count over the benchmarks it was learned on.

    Args:
        table: Entries from count_tokens.
    Returns:
        Mean and sample standard deviation keyed by model, then by bar key.
    """
    means: dict = {model: {"skillpng": IMAGE_TOKENS[model]}
                   for model in MODELS if model not in EXCLUDED_MODELS}
    # The image token count is fixed per model, so it has no spread.
    spreads: dict = {model: {} for model in means}
    for entry in table:
        model, source = entry["model"], entry["source"]
        if source in means[model]:
            raise SystemExit(f"several {source} runs for {model}; keep one")
        counts = [count for count in entry["counts"].values()
                  if count >= MIN_SKILL_TOKENS]
        means[model][source] = (sum(counts) / len(counts) if counts
                                else float("nan"))
        if len(counts) > 1:
            spreads[model][source] = stdev(counts)
    return means, spreads


def data_aspect(ax: Axes) -> float:
    """Measure how much taller a y data unit is drawn than an x data unit.

    Args:
        ax: Axes whose limits and size are final.
    Returns:
        Ratio of y data units to x data units covering the same pixels.
    """
    box = ax.get_window_extent()
    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    return ((y_high - y_low) / box.height) / ((x_high - x_low) / box.width)


def save_cropped(fig: Figure, out: Path) -> None:
    """Save a figure cropped to what it draws, with a margin above it.

    Args:
        fig: Figure to save and close.
        out: PNG file the figure is written to.
    """
    # Measured at the saved dpi once everything has its place, as savefig
    # measures a tight crop.
    fig.set_dpi(DPI)
    fig.draw_without_rendering()
    box = fig.get_tightbbox()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", dpi=DPI,
                bbox_inches=Bbox.from_extents(box.x0, box.y0, box.x1,
                                              box.y1 + TOP_MARGIN / DPI))
    plt.close(fig)


def plot_tokens(means: dict, spreads: dict, out: Path) -> None:
    """Draw the token counts as grouped bars with rounded tops and save them.

    Args:
        means: Mean token count keyed by model, then by bar key.
        spreads: Standard deviation keyed by model, then by bar key.
        out: PNG file the figure is written to.
    """
    models = list(means)
    centres = [index * GROUP_SPACING for index in range(len(models))]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    top = max(value + spreads[model].get(key, 0)
              for model, bars in means.items()
              for key, value in bars.items() if not math.isnan(value))
    ax.set_xlim(centres[0] - GROUP_SPACING / 2,
                centres[-1] + GROUP_SPACING / 2)
    ax.set_ylim(0, top * 1.05)
    aspect = data_aspect(ax)
    radius = ROUNDING * BAR_WIDTH
    # Each bar starts below the axis, which clips its lower corners, so only
    # the top is rounded. Starting two radii down fits even the lowest bars.
    bottom = -2 * radius * aspect
    for index, (key, (_, color)) in enumerate(BARS.items()):
        offset = (index - (len(BARS) - 1) / 2) * BAR_WIDTH
        for x, model in zip(centres, models, strict=True):
            height = means[model].get(key, float("nan"))
            if math.isnan(height):
                continue
            ax.add_patch(FancyBboxPatch(
                (x + offset - BAR_WIDTH / 2, bottom), BAR_WIDTH,
                height - bottom,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                mutation_aspect=aspect, facecolor=color, edgecolor="white",
                linewidth=1))
            if key in spreads[model]:
                ax.errorbar(x + offset, height, yerr=spreads[model][key],
                            fmt="none", ecolor=ERROR_COLOR,
                            elinewidth=ERROR_WIDTH, capsize=ERROR_CAP,
                            capthick=ERROR_WIDTH, zorder=3)
    ax.set_xticks(centres, [LABELS.get(model, MODELS[model][0])
                            for model in models])
    ax.set_ylabel("Avg. Tokens per Skill", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=color, label=label)
                       for label, color in BARS.values()],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(BARS),
              frameon=False, fontsize=FONT_SIZE, borderpad=0,
              borderaxespad=LEGEND_PAD)
    save_cropped(fig, out)


def resolution_scores(runs_dir: Path) -> tuple[dict, dict]:
    """Collect the resolution ablation scores as analyze_runs tables them.

    Cells with several training seeds hold their mean.

    Args:
        runs_dir: Directory holding the benchmark run trees.
    Returns:
        Score and seed std percentages keyed by benchmark, then cell key.
    """
    rows = [row for row in load_rows(runs_dir, SEED)
            if not is_transfer(row) and not is_ensemble(row)
            and row.get("n_distractors") in (None, DISTRACTORS)]
    ablation, benchmarks = select_resolutions(rows,
                                              load_image_sizes(runs_dir))
    if not benchmarks:
        return {}, {}
    failed = find_failed(ablation, load_failed_images(runs_dir),
                         load_failed_skills(runs_dir))
    [entry] = build_table(ablation, False, failed)
    add_spread([entry], resolution_seeds(ablation, failed))
    scores = {benchmark: {key: score * 100
                          for key, score in entry["cells"][benchmark].items()
                          if score is not None}
              for benchmark in benchmarks}
    spreads = {benchmark: {key: std * 100 for key, std in cells.items()}
               for benchmark, cells in entry["spread"].items()}
    return scores, spreads


def plot_resolutions(scores: dict, spreads: dict, out: Path) -> None:
    """Draw the Skill.png score per image size over the no-skill lines.

    Sizes with a seed std get a band of one std around their mean.

    Args:
        scores: Score percentages keyed by benchmark, then by table cell key.
        spreads: Seed std percentages keyed like scores.
        out: PNG file the figure is written to.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    color = BARS["skillpng"][1]
    handles: list[Artist | tuple[Artist, ...]] = []
    labels: list[str] = []
    for benchmark, style in RESOLUTION_STYLES.items():
        cells = scores.get(benchmark)
        if cells is None:
            continue
        name = BENCHMARKS[benchmark]
        spread = spreads.get(benchmark, {})
        # Sizes sit evenly spaced, so 256 and 320 do not crowd together.
        points = [(x, cells[f"res{size}"], spread.get(f"res{size}"))
                  for x, size in enumerate(RESOLUTIONS)
                  if f"res{size}" in cells]
        (line,) = ax.plot([x for x, _, _ in points],
                          [score for _, score, _ in points],
                          color=color, linestyle=style,
                          linewidth=LINE_WIDTH, marker="o",
                          markersize=MARKER_SIZE)
        band = np.array([point for point in points if point[2] is not None])
        if len(band):
            x, mean, std = band.T
            fill = ax.fill_between(x, mean - std, mean + std,
                                   color=SPREAD_COLOR, linewidth=0)
            # The legend entry shows the band behind the line.
            handles.append((fill, line))
        else:
            handles.append(line)
        labels.append(f"Skill.png ({name})")
        if "baseline" in cells:
            handles.append(ax.axhline(
                cells["baseline"], color=BASELINE_COLOR, linestyle=style,
                linewidth=LINE_WIDTH))
            labels.append(f"No Skill ({name})")
    ax.set_xlim(-0.25, len(RESOLUTIONS) - 0.75)
    ax.set_ylim(20, 80)
    ax.set_xticks(range(len(RESOLUTIONS)),
                  [f"{size}\N{MULTIPLICATION SIGN}{size}"
                   for size in RESOLUTIONS])
    ax.tick_params(labelsize=FONT_SIZE)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    # Filled column by column, so each column holds one benchmark.
    ax.legend(handles=handles, labels=labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=len(RESOLUTION_STYLES),
              frameon=False, fontsize=FONT_SIZE, handlelength=3,
              borderpad=0, borderaxespad=LEGEND_PAD)
    save_cropped(fig, out)


def ensemble_scores(runs_dir: Path) -> dict:
    """Collect the transfer table scores on the benchmarks with ensembles.

    Args:
        runs_dir: Directory holding the benchmark run trees.
    Returns:
        Score percentages by model, members first, then benchmark and key.
    """
    rows = [row for row in load_rows(runs_dir, SEED)
            if row.get("n_distractors") in (None, DISTRACTORS)]
    loaded = [row for row in rows if not is_ensemble(row)]
    # Qwen3.5-4B's own Skill.png is read from selected, at the reported size.
    sizes = load_image_sizes(runs_dir)
    selected, _ = select_latest([row for row in loaded
                                 if not is_transfer(row)
                                 and at_resolution(row, sizes, RESOLUTION)])
    transfer, _ = select_transfer(loaded, selected)
    ensemble, benchmarks = select_ensemble(
        [row for row in rows if is_ensemble(row)], selected, transfer)
    if not benchmarks:
        return {}
    picked = {**transfer,
              **{key: row for key, row in ensemble.items()
                 if key[3] in (BASELINE_CONDITION,
                               TRANSFER_ENSEMBLE_CONDITION)}}
    keys = ("baseline", *HEATMAP_METHODS)
    cells = {entry["model"]: entry["cells"]
             for entry in build_table(picked, False, {})}
    return {model: {benchmark: {key: cells[model][benchmark][key] * 100
                                for key in keys
                                if cells[model][benchmark][key] is not None}
                    for benchmark in benchmarks}
            for model in ENSEMBLE_MODELS if model in cells}


def text_color(color: tuple) -> str:
    """Pick the text colour that contrasts more with a cell colour.

    Args:
        color: RGBA colour of the cell.
    Returns:
        White on dark cells, the dark ink otherwise.
    """
    linear = [channel / 12.92 if channel <= 0.04045
              else ((channel + 0.055) / 1.055) ** 2.4
              for channel in color[:3]]
    luminance = (0.2126 * linear[0] + 0.7152 * linear[1]
                 + 0.0722 * linear[2])
    return "white" if luminance < DARK_CELL else INK


def rounded_outline(corners: list[tuple[float, float]],
                    radius: tuple[float, float]) -> mpath.Path:
    """Trace a polygon of horizontal and vertical edges with round corners.

    Args:
        corners: Vertices of the polygon in order, in data units.
        radius: Corner radius along x and along y, in data units.
    Returns:
        Closed path with a quadratic curve at each corner, as in round boxes.
    """
    vertices: list[np.ndarray] = []
    codes: list[np.uint8] = []
    for before, corner, after in zip(corners[-1:] + corners[:-1], corners,
                                     corners[1:] + corners[:1], strict=True):
        point = np.array(corner, dtype=float)
        vertices += [point + np.sign(np.subtract(before, point)) * radius,
                     point,
                     point + np.sign(np.subtract(after, point)) * radius]
        codes += [mpath.Path.LINETO if codes else mpath.Path.MOVETO,
                  mpath.Path.CURVE3, mpath.Path.CURVE3]
    vertices.append(vertices[0])
    codes.append(mpath.Path.CLOSEPOLY)
    return mpath.Path(vertices, codes)


def draw_span(ax: Axes, left: float, right: float, offset: float,
              label: str) -> None:
    """Draw a label on a rule spanning columns above the axes.

    Args:
        ax: Axes the columns are drawn on.
        left: Left end of the rule, in data units.
        right: Right end of the rule, in data units.
        offset: Height of the rule above the axes, in points.
        label: Text set on the rule.
    """
    rule = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=offset,
                       units="points")
    text = offset_copy(ax.get_xaxis_transform(), fig=ax.figure,
                       y=offset + 2, units="points")
    ax.plot([left, right], [1, 1], transform=rule, color=INK, linewidth=0.8,
            clip_on=False)
    ax.text((left + right) / 2, 1, label, transform=text, ha="center",
            va="bottom", color=INK, fontsize=FONT_SIZE)


def plot_ensemble(scores: dict, out: Path) -> None:
    """Draw each method's change over No Skill as a heatmap and save it.

    Args:
        scores: Score percentages keyed by model, then by benchmark and key.
        out: PNG file the figure is written to.
    """
    models = list(scores)
    benchmarks = list(scores[models[0]])
    rows = {model: index for index, model in enumerate(models)}
    width = len(HEATMAP_METHODS) + BENCHMARK_GAP
    columns = {(benchmark, key): index * width + offset
               for index, benchmark in enumerate(benchmarks)
               for offset, key in enumerate(HEATMAP_METHODS)}
    deltas = {(model, benchmark, key): cells[key] - cells["baseline"]
              for model in models
              for benchmark, cells in scores[model].items()
              for key in HEATMAP_METHODS
              if key in cells and "baseline" in cells}
    limit = DELTA_STEP * math.ceil(max(abs(delta)
                                       for delta in deltas.values())
                                   / DELTA_STEP)
    norm = Normalize(-limit, limit)
    cmap = LinearSegmentedColormap.from_list("change", DIVERGING)

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    # The colour bar shrinks the axes, so it comes before any measuring.
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                       orientation="horizontal", fraction=0.05, pad=0.04,
                       aspect=30, shrink=0.5)
    bar.set_label("\N{GREEK CAPITAL LETTER DELTA} vs. No Skill (p.p.)",
                  fontsize=FONT_SIZE)
    bar.ax.tick_params(labelsize=FONT_SIZE, length=2)
    bar.outline.set_visible(False)
    ax.set_xlim(-0.5, max(columns.values()) + 0.5)
    ax.set_ylim(max(rows.values()) + 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.set_xticks(list(columns.values()),
                  [HEATMAP_METHODS[key] for _, key in columns])
    ax.set_yticks(list(rows.values()),
                  [HEATMAP_LABELS.get(model, MODELS[model][0])
                   for model in rows])
    ax.tick_params(labelsize=FONT_SIZE, length=0)
    ax.spines[:].set_visible(False)
    # The y axis runs downwards, which flips the sign of the aspect.
    aspect = abs(data_aspect(ax))
    gap_y = CELL_GAP * aspect
    half = (1 - ROW_GAP) / 2
    for (benchmark, key), x in columns.items():
        for model, y in rows.items():
            delta = deltas.get((model, benchmark, key))
            cell = FancyBboxPatch(
                (x - 0.5 + CELL_GAP / 2, y - half), 1 - CELL_GAP, 2 * half,
                boxstyle=f"round,pad=0,rounding_size={CELL_ROUNDING}",
                mutation_aspect=aspect)
            ax.add_patch(cell)
            # A missing result leaves its cell empty until it comes in.
            if delta is None:
                cell.set(facecolor=EMPTY_FILL, edgecolor=EMPTY_EDGE,
                         linewidth=EMPTY_EDGE_WIDTH)
                continue
            # The best method of each model and benchmark is set in bold.
            best = max(deltas.get((model, benchmark, method), -math.inf)
                       for method in HEATMAP_METHODS)
            color = cmap(norm(delta))
            cell.set(facecolor=color, edgecolor="none")
            # A true minus sign, as on the colour bar.
            change = f"{delta:+.2f}".replace("-", "\N{MINUS SIGN}")
            ax.text(x, y, change, ha="center", va="center",
                    color=text_color(color), fontsize=FONT_SIZE,
                    fontweight="bold" if delta == best else "normal")

    # Arrows run from the last model a skill or image was optimized on down
    # through the models it was injected into, one in each gap, empty cells
    # included. A plus joins the models of an ensemble.
    for (_, key), x in columns.items():
        trained = [model for model in HEATMAP_SOURCES[key] if model in rows]
        injected = [model for model in rows if model not in trained]
        if not trained or not injected:
            continue
        # A marker, unlike a glyph, sits centred in the gap.
        for upper, lower in pairwise(trained):
            ax.plot(x, (rows[upper] + rows[lower]) / 2, marker="+",
                    markersize=PLUS_SIZE, markeredgewidth=PLUS_WIDTH,
                    color=INK)
        for upper, lower in pairwise([trained[-1], *injected]):
            ax.add_patch(FancyArrowPatch(
                (x, rows[upper] + half), (x, rows[lower] - half),
                arrowstyle="-|>", mutation_scale=ARROW_HEAD,
                shrinkA=ARROW_PAD, shrinkB=ARROW_PAD, color=LINK,
                linewidth=LINK_WIDTH, zorder=0.5))

    # Benchmark names sit on a rule spanning their methods.
    first, *_, last = HEATMAP_METHODS
    for benchmark in benchmarks:
        draw_span(ax, columns[(benchmark, first)] - 0.5 + CELL_GAP / 2,
                  columns[(benchmark, last)] + 0.5 - CELL_GAP / 2,
                  HEADER_OFFSET, BENCHMARKS[benchmark])

    # A band behind the cells of the models each skill or image was
    # optimized on, one per benchmark. Below the model the Skill.png comes
    # from, it reaches down over an ensemble's further models. One outline
    # per benchmark keeps the antialiased edges from overlapping.
    if TRANSFER_SOURCE in rows:
        top = rows[TRANSFER_SOURCE] - half - gap_y / 2
        radius = (CELL_ROUNDING, CELL_ROUNDING * aspect)
        for benchmark in benchmarks:
            steps = [(columns[(benchmark, key)],
                      max(rows[TRANSFER_SOURCE],
                          *(rows[model] for model in HEATMAP_SOURCES[key]
                            if model in rows)) + half + gap_y / 2)
                     for key in HEATMAP_METHODS]
            first_x, first_bottom = steps[0]
            last_x, last_bottom = steps[-1]
            corners = [(first_x - 0.5, top), (last_x + 0.5, top),
                       (last_x + 0.5, last_bottom)]
            # Right to left, the bottom steps between columns.
            for (x, bottom), (_, after) in pairwise(steps[::-1]):
                if bottom != after:
                    corners += [(x - 0.5, bottom), (x - 0.5, after)]
            corners.append((first_x - 0.5, first_bottom))
            ax.add_patch(PathPatch(rounded_outline(corners, radius),
                                   facecolor=BAND, edgecolor="none",
                                   zorder=0))

    save_cropped(fig, out)


def beta1_losses(
        runs_dir: Path) -> dict[float, tuple[Path, list[int], list[float]]]:
    """Read the training losses of the images the beta1 table reports.

    Args:
        runs_dir: Directory holding the benchmark run trees.
    Returns:
        Training run, logged steps and losses, keyed by beta1 in order.
    """
    sizes = load_image_sizes(runs_dir)
    rows = [row for row in load_rows(runs_dir, SEED)
            if not is_transfer(row) and not is_ensemble(row)
            and row.get("n_distractors") in (None, DISTRACTORS)
            and row["train_seed"] == TRAIN_SEED
            and at_resolution(row, sizes, RESOLUTION)]
    selected, _ = select_latest(
        [row for row in rows
         if split_beta1(row["condition"])[1] == REPORTED_BETA1])
    # Each test row names the image it injected, which is the one its
    # training run selected on val.
    runs = {json.loads(path.read_text())["digest"]: path.parent
            for path in (runs_dir / BETA1_BENCHMARK / BETA1_MODEL)
            .glob("*/best.json")}
    losses = {}
    for (_, _, _, condition), row in select_beta1(rows, selected).items():
        name, beta1 = split_beta1(condition)
        if name != INJECTED_CONDITION:
            continue
        run = runs.get(row["image_digest"])
        if run is None:
            raise SystemExit(f"no training run under {runs_dir} holds the "
                             f"image of {row['run_name']}")
        with (run / "history.jsonl").open() as handle:
            records = [json.loads(line) for line in handle]
        # The record after the last step holds only the final val score.
        logged = [record for record in records if "loss" in record]
        losses[beta1] = (run, [record["step"] for record in logged],
                         [record["loss"] for record in logged])
    return dict(sorted(losses.items()))


def plot_beta1(losses: dict[float, tuple[Path, list[int], list[float]]],
               out: Path) -> None:
    """Draw the training loss of each beta1 on a log scale and save it.

    Args:
        losses: Training run, logged steps and losses, keyed by beta1.
        out: PNG file the figure is written to.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for beta1, (_, steps, values) in losses.items():
        color, style, label = BETA1_STYLES[beta1]
        # The reported beta1 is drawn over the others.
        ax.plot(steps, values, color=color, linestyle=style,
                linewidth=LINE_WIDTH, label=label,
                zorder=3 if beta1 == REPORTED_BETA1 else 2)
    ax.set_yscale("log")
    # The axis ends on the labelled decade below the lowest loss.
    lowest = min(min(values) for _, _, values in losses.values())
    ax.set_ylim(bottom=10 ** math.floor(math.log10(lowest)))
    ax.set_xlim(0, max(steps[-1] for _, steps, _ in losses.values()))
    ax.set_xlabel("Training Step", fontsize=FONT_SIZE)
    ax.set_ylabel("Training Loss", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(losses), frameon=False, fontsize=FONT_SIZE,
              handlelength=3, borderpad=0, borderaxespad=LEGEND_PAD)
    save_cropped(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skillopt-dir", type=Path, default=SKILLOPT_DIR,
                        help="directory holding the SkillOpt skill trees")
    parser.add_argument("--llmskill-dir", type=Path, default=LLMSKILL_DIR,
                        help="directory holding the LLM-Skill skill trees")
    parser.add_argument("--optimizer", default=OPTIMIZER,
                        help="only draw SkillOpt skills from this optimizer LLM")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="PNG file the token figure is written to")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="directory holding the benchmark run trees")
    parser.add_argument("--resolutions-out", type=Path,
                        default=RESOLUTIONS_OUT,
                        help="PNG file the resolution figure is written to")
    parser.add_argument("--ensemble-out", type=Path, default=ENSEMBLE_OUT,
                        help="PNG file the ensemble heatmap is written to")
    parser.add_argument("--beta1-out", type=Path, default=BETA1_OUT,
                        help="PNG file the beta1 loss figure is written to")
    args = parser.parse_args()

    skills = find_skills(args.skillopt_dir, args.llmskill_dir,
                         args.optimizer)
    skills = {key: paths for key, paths in skills.items()
              if key[0] not in EXCLUDED_MODELS}
    if not skills:
        raise SystemExit(f"no skills under {args.llmskill_dir} or "
                         f"{args.skillopt_dir}")
    means, spreads = average_tokens(count_tokens(skills))
    out = args.out.with_suffix(".png")
    plot_tokens(means, spreads, out)
    for model, bars in means.items():
        spread = spreads[model]
        cells = "  ".join(f"{BARS[key][0]} {bars[key]:.0f}"
                          + (f" ± {spread[key]:.0f}" if key in spread else "")
                          for key in BARS if key in bars)
        print(f"{MODELS[model][0]}: {cells}")
    print(f"wrote {out}")

    scores, spreads = resolution_scores(args.runs_dir)
    if not scores:
        raise SystemExit(f"no resolution ablation runs under {args.runs_dir}")
    resolutions_out = args.resolutions_out.with_suffix(".png")
    plot_resolutions(scores, spreads, resolutions_out)
    for benchmark, cells in scores.items():
        spread = spreads.get(benchmark, {})
        print(f"{BENCHMARKS[benchmark]}: "
              + "  ".join(f"{key} {score:.2f}"
                          + (f" ± {spread[key]:.2f}" if key in spread else "")
                          for key, score in cells.items()))
    print(f"wrote {resolutions_out}")

    ensemble = ensemble_scores(args.runs_dir)
    if not ensemble:
        raise SystemExit(f"no ensemble runs under {args.runs_dir}")
    ensemble_out = args.ensemble_out.with_suffix(".png")
    plot_ensemble(ensemble, ensemble_out)
    names = {"baseline": "No Skill", **HEATMAP_METHODS}
    for model, cells in ensemble.items():
        for benchmark, scores in cells.items():
            print(f"{MODELS[model][0]}, {BENCHMARKS[benchmark]}: "
                  + "  ".join(f"{names[key]} {score:.2f}"
                              for key, score in scores.items()))
    print(f"wrote {ensemble_out}")

    losses = beta1_losses(args.runs_dir)
    if not losses:
        raise SystemExit(f"no beta1 ablation runs under {args.runs_dir}")
    beta1_out = args.beta1_out.with_suffix(".png")
    plot_beta1(losses, beta1_out)
    for beta1, (run, _, values) in losses.items():
        print(f"beta1 {beta1}: final loss {values[-1]:.4f}, min "
              f"{min(values):.4f} in {run.name}")
    print(f"wrote {beta1_out}")


if __name__ == "__main__":
    main()
