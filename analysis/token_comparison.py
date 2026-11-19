"""Token length of the LLM-Skill and GPT-6 Luna SkillOpt skills.

Reads every baselines/llmskill/skills/<benchmark>/<run>/skill.md and every
GPT-6 Luna baselines/skillopt/skills/<benchmark>/<model>/<run>/skill.md, and
counts the tokens of each skill under the tokenizer of the model it serves.
An LLM-Skill serves every model, so it is counted under each tokenizer.
Prints one block per model, one row per skill run, one column per benchmark.
SkillOpt skills archived under old/ are not read, so a model whose skills are
being retrained shows no SkillOpt cell until the new skill lands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SKILLOPT_DIR = ROOT / "baselines" / "skillopt" / "skills"
LLMSKILL_DIR = ROOT / "baselines" / "llmskill" / "skills"

# Only these are reported; anything else on disk is ignored.
BENCHMARKS = {
    "searchqa": "SearchQA",
    "traject_bench": "TRAJECT",
    "livemathematicianbench": "LiveMath",
    "t1": "T1",
}

# Directory name to display name and Hugging Face tokenizer repo.
MODELS = {
    "qwen3.5-4B": ("Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
    "qwen3.5-9B": ("Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
    "gemma-4-E4B-it": ("Gemma-4-E4B-it", "google/gemma-4-E4B-it"),
    "nemotron-3-nano-omni-30B-A3B-reasoning": (
        "Nemotron-3-Nano-Omni-30B-A3B",
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"),
}
# Never reported. Gemma-4-12B-it was dropped, but its SkillOpt skills stay
# on disk.
EXCLUDED_MODELS = {"gemma-4-12B-it"}

# Skill source to display name, in row order within a model's block.
SOURCES = {"llmskill": "LLM-Skill", "skillopt": "SkillOpt"}

# Optimizer LLM as SkillOpt run directories spell it.
OPTIMIZER = "gpt-6-luna"
# Selection origin recorded when no optimizer step beat the seed skill.
SEED_ORIGIN = "initial_skill"
SEED_MARK = "*"


def find_skills(skillopt_dir: Path, llmskill_dir: Path,
                optimizer: str) -> dict:
    """Collect the LLM-Skill files and the skill files of one optimizer LLM.

    Args:
        skillopt_dir: Directory holding benchmark/model/run/skill.md files.
        llmskill_dir: Directory holding benchmark/run/skill.md files.
        optimizer: Optimizer LLM as SkillOpt run directories spell it.
    Returns:
        Skill paths keyed by model, source and run, then by benchmark.
    """
    skills: dict = {}
    # The fixed depth keeps archived skills under model/old/ out.
    for path in sorted(skillopt_dir.glob(f"*/*/{optimizer}_*/skill.md")):
        benchmark, model, run = path.parts[-4:-1]
        if benchmark not in BENCHMARKS or model in EXCLUDED_MODELS:
            continue
        if model not in MODELS:
            raise SystemExit(f"no tokenizer known for {model}; add it to "
                             "MODELS")
        skills.setdefault((model, "skillopt", run), {})[benchmark] = path
    for path in sorted(llmskill_dir.glob("*/*/skill.md")):
        benchmark, run = path.parts[-3:-1]
        if benchmark not in BENCHMARKS:
            continue
        for model in MODELS:
            skills.setdefault((model, "llmskill", run), {})[benchmark] = path
    return skills


def load_tokenizer(repo: str) -> Tokenizer:
    """Load the tokenizer of one Hugging Face model repo.

    Args:
        repo: Hugging Face repo id of the model.
    Returns:
        The tokenizer read from the repo's tokenizer.json.
    """
    # Reading tokenizer.json alone runs no repo code, which Nemotron's
    # config would otherwise ask to execute.
    return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))


def count_tokens(skills: dict) -> list[dict]:
    """Count the tokens of every skill under the tokenizer of its model.

    Args:
        skills: Skill paths keyed by model, source and run, then by benchmark.
    Returns:
        List of entries carrying a model, source, run and token counts.
    """
    models = list(MODELS)
    sources = list(SOURCES)
    keys = sorted(skills, key=lambda key: (models.index(key[0]),
                                           sources.index(key[1]), key[2]))
    table = []
    tokenizers = {}
    for model, source, run in keys:
        if model not in tokenizers:
            tokenizers[model] = load_tokenizer(MODELS[model][1])
        tokenizer = tokenizers[model]
        counts = {}
        seeds = set()
        for benchmark, path in skills[(model, source, run)].items():
            # Stripped, as the skill is appended to the system prompt.
            text = path.read_text().strip()
            counts[benchmark] = len(
                tokenizer.encode(text, add_special_tokens=False).ids)
            if is_seed(path):
                seeds.add(benchmark)
        table.append({"model": model, "source": source, "run": run,
                      "counts": counts, "seeds": seeds})
    return table


def is_seed(path: Path) -> bool:
    """Tell whether the optimizer kept the unchanged seed skill.

    Args:
        path: Path of a skill.md next to its skill.meta.json.
    Returns:
        True when no optimizer step beat the seed skill on validation.
    """
    meta_path = path.with_name("skill.meta.json")
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("selection", {}).get("best_origin") == SEED_ORIGIN


def format_count(entry: dict, benchmark: str) -> str:
    """Format one token count cell.

    Args:
        entry: Entry from count_tokens.
        benchmark: Benchmark of the cell.
    Returns:
        The count, marked when it is the seed skill, or - when missing.
    """
    if benchmark not in entry["counts"]:
        return "-"
    mark = SEED_MARK if benchmark in entry["seeds"] else ""
    return f"{entry['counts'][benchmark]}{mark}"


def render_terminal(table: list[dict]) -> str:
    """Render the token counts as an aligned terminal table.

    Args:
        table: Entries from count_tokens.
    Returns:
        The rendered table.
    """
    header = ["Model", "Skill", "Run", *BENCHMARKS.values()]
    labels = len(header) - len(BENCHMARKS)
    rows = []
    for index, entry in enumerate(table):
        # Name the model on the first row of its block only.
        first = not index or table[index - 1]["model"] != entry["model"]
        rows.append([MODELS[entry["model"]][0] if first else "",
                     SOURCES[entry["source"]], entry["run"],
                     *(format_count(entry, benchmark)
                       for benchmark in BENCHMARKS)])
    widths = [max(len(row[i]) for row in [header, *rows])
              for i in range(len(header))]

    def line(cells: list[str]) -> str:
        """Lay out one table row.

        Args:
            cells: Cell texts in column order.
        Returns:
            The padded row, text left-aligned and counts right-aligned.
        """
        text = [cell.ljust(widths[i]) if i < labels else cell.rjust(widths[i])
                for i, cell in enumerate(cells)]
        return "  ".join(text).rstrip()

    lines = [line(header), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    for index, row in enumerate(rows):
        if row[0] and index:
            lines.append("")
        lines.append(line(row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skillopt-dir", type=Path, default=SKILLOPT_DIR,
                        help="directory holding the SkillOpt skill trees")
    parser.add_argument("--llmskill-dir", type=Path, default=LLMSKILL_DIR,
                        help="directory holding the LLM-Skill skill trees")
    parser.add_argument("--optimizer", default=OPTIMIZER,
                        help="only report SkillOpt skills from this"
                             " optimizer LLM")
    args = parser.parse_args()

    skills = find_skills(args.skillopt_dir, args.llmskill_dir,
                         args.optimizer)
    if not skills:
        raise SystemExit(f"no skills under {args.llmskill_dir} or "
                         f"{args.skillopt_dir}")
    table = count_tokens(skills)
    print(f"Skill tokens, LLM-Skill and SkillOpt ({args.optimizer})")
    print(render_terminal(table))
    if any(entry["seeds"] for entry in table):
        print(f"{SEED_MARK} seed skill kept; no optimizer step beat it.")


if __name__ == "__main__":
    main()
