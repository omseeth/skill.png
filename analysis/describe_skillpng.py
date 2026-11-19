"""Ask a served model, in several ways, what it makes of a Skill.png.

For one model behind a vLLM server, asks every question in QUESTIONS about
each image: every MAIN_MODELS model's non-reasoning Skill.png on every
benchmark, the image the main table reports, its own and the other models'
alike, plus two controls the size the model's own images were optimized at,
uniform noise (the random-image condition's canvas) and flat gray (the
canvas optimization starts from). Every question is asked twice over: bare,
the image and the question alone in one user turn, and under a benchmark's
system prompt, where the image sits during evaluation. Skill.png images go
under their own benchmark's system prompt, the controls under every
benchmark's, so each image has a control asked the same way. Each image is
sent at the size it was optimized at, as the transfer evaluations send it;
Gemma packs every image to its own budget regardless.

Thinking is off, as in the non-reasoning tables, and each question is
sampled --samples times at --temperature. The multiple-choice questions
shuffle their options per sample, so a favored letter cannot pass for a
reading of the image; the answer is recorded as the option's key, not its
letter.

Writes one JSON line per answer to runs/answers/<model>/descriptions.jsonl,
replacing the file.

With --images ablations it asks instead about the images of the ablation
tables, into descriptions_ablations.jsonl: each benchmark's ensemble image
(optimized on the ENSEMBLE_MEMBERS at once with unit-norm gradients, the one
the tables report), OPTIMIZER_MODEL's images optimized under the
CONDITIONED_OPTIMIZER SkillOpt skill, also asked under their benchmark's
system prompt with that skill composed in, as they are scored, and
RESOLUTION_MODEL's images of every other ablation size. The controls then
come at every size those images have, under every context an image of that
size is asked in.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from analyze_runs import (
    BENCHMARKS,
    CONDITIONED_OPTIMIZER,
    DISTRACTORS,
    ENSEMBLE_MEMBERS,
    INJECTED_CONDITION,
    OPTIMIZER_MODEL,
    RESOLUTION,
    RESOLUTIONS,
    RUNS_DIR,
    SEED,
    TRANSFER_ENSEMBLE_CONDITION,
    at_resolution,
    conditioned_condition,
    ensemble_condition,
    is_ensemble,
    is_transfer,
    load_failed_images,
    load_image_sizes,
    load_rows,
    resolution_condition,
    select_latest,
    select_resolutions,
)

from skillpng.config import SKILL_TEMPLATE, load_config
from skillpng.evaluate import check_trained_under, compose_system_prompt, read_skill
from skillpng.image.param import FullImage, load_param, noise_param
from skillpng.models.vllm import VLLMClient

CONFIGS_DIR = RUNS_DIR.parent / "configs"

# What each benchmark asks for, worded for a reader who has never seen it.
# The keys double as the answers of the task question.
TASKS = {
    "t1": "Writing Python code that calls travel-planning tools for"
          " flights, hotels and restaurants",
    "searchqa": "Answering trivia questions from retrieved text passages",
    "livemathematicianbench": "Solving research-level mathematics"
                              " multiple-choice questions",
    "traject": "Choosing which API tools to call, and with which"
               " parameters, for a user's request",
    "none": "None of these: it is not related to any task",
}
KINDS = {
    "photo": "A natural photograph",
    "drawing": "A drawing or diagram",
    "text": "A document or written text",
    "noise": "Random noise or static",
    "pattern": "An abstract pattern or texture",
}
# Question id to its text and, for multiple choice, its options. The open
# ones go from what is seen to what it is for; the closed ones can be counted.
QUESTIONS = {
    "describe": ("Describe this image in detail.", None),
    "read": ("Transcribe any text, letters, numbers or symbols you can read"
             " in this image, exactly as they appear. If there are none,"
             " say so.", None),
    "instruct": ("This image was given to you as an instruction. What is it"
                 " telling you to do?", None),
    "prompt": ("Rewrite what this image conveys as a short system prompt,"
               " at most five sentences, for an AI assistant.", None),
    "words": ("List ten words or concepts this image makes you think of,"
              " separated by commas.", None),
    "task": ("Which task is this image most likely meant to help with?",
             TASKS),
    "kind": ("What kind of image is this?", KINDS),
    "meaning": ("On a scale from 1 (pure random noise) to 10 (clear,"
                " meaningful content), how meaningful is this image? Answer"
                " with a single number.", None),
}
LETTERS = "ABCDEFGH"
# The models of the main tables, whose Skill.png images are asked about.
MAIN_MODELS = ("qwen3.5-4B", "qwen3.5-9B", "gemma-4-E4B-it",
               "nemotron-3-nano-omni-30B-A3B-reasoning")
# Midpoint of the [0,1] cube: the canvas optimization starts from.
GRAY = 0.5
# Context suffix of a benchmark's system prompt with an image's trained
# skill composed in.
SKILLED = "+skill"


def png_index(runs_dir: Path) -> dict[tuple[str, str], Path]:
    """Map every trained image's digest to its PNG.

    Args:
        runs_dir: Directory holding the benchmark run trees.
    Returns:
        Path of each best.png, keyed by benchmark and the digest its sidecar
        records.
    """
    pngs = {}
    for path in runs_dir.glob("*/*/*/best.json"):
        digest = json.loads(path.read_text())["digest"]
        pngs[(path.parent.parent.parent.name, digest)] = path.with_suffix(".png")
    return pngs


def png_of(pngs: dict[tuple[str, str], Path], row: dict) -> Path:
    """Find the PNG a result row injected.

    Args:
        pngs: Paths keyed by benchmark and digest, from png_index.
        row: Result row of an injected image.
    Returns:
        The image's best.png.
    """
    key = (row["benchmark"], row["image_digest"])
    if key not in pngs:
        raise SystemExit(f"no best.png with digest {row['image_digest']}"
                         f" for {row['benchmark']} {row['model']}")
    return pngs[key]


def reported_images(runs_dir: Path) -> dict[tuple[str, str], Path]:
    """Find the Skill.png images the main non-reasoning table reports.

    Selects rows the way analyze_runs.main does, then joins each selected
    Skill.png row to its PNG through the digest in the training sidecars.

    Args:
        runs_dir: Directory holding the benchmark run trees.
    Returns:
        Path of each MAIN_MODELS model's reported image, keyed by benchmark
        and the model it was optimized on.
    """
    rows = [row for row in load_rows(runs_dir, SEED)
            if row.get("n_distractors") in (None, DISTRACTORS)
            and not is_ensemble(row) and not is_transfer(row)]
    sizes = load_image_sizes(runs_dir)
    rows = [row for row in rows if at_resolution(row, sizes, RESOLUTION)]
    selected, _ = select_latest(rows)
    pngs = png_index(runs_dir)
    return {(benchmark, model): png_of(pngs, row)
            for (benchmark, model, reasoning, condition), row
            in selected.items()
            if model in MAIN_MODELS and not reasoning
            and condition == INJECTED_CONDITION}


def ablation_images(runs_dir: Path) -> dict[tuple[str, str, str], Path]:
    """Find the images the ensemble, optimizer and resolution tables report.

    Selects rows the way analyze_runs.main does for those tables.

    Args:
        runs_dir: Directory holding the benchmark run trees.
    Returns:
        Path of each image keyed by variant, benchmark and the model it was
        optimized on: ensemble images under the members joined by +,
        conditioned ones under OPTIMIZER_MODEL and res<side> ones under
        RESOLUTION_MODEL.
    """
    loaded = [row for row in load_rows(runs_dir, SEED)
              if row.get("n_distractors") in (None, DISTRACTORS)]
    pngs = png_index(runs_dir)
    images = {}

    ensembled = [{**row, "condition": ensemble_condition(row)}
                 for row in loaded if is_ensemble(row)
                 and not row["reasoning"]
                 and row["condition"] == INJECTED_CONDITION]
    members = "+".join(ENSEMBLE_MEMBERS)
    for (benchmark, _, _, condition), row in (
            select_latest(ensembled)[0].items()):
        if condition != TRANSFER_ENSEMBLE_CONDITION:
            continue
        # Every model of the table is scored with the same image.
        path = png_of(pngs, row)
        if images.setdefault(("ensemble", benchmark, members), path) != path:
            raise SystemExit(f"{benchmark} ensemble rows differ in image")

    rows = [row for row in loaded
            if not is_ensemble(row) and not is_transfer(row)]
    sizes = load_image_sizes(runs_dir)
    selected, _ = select_latest(
        [row for row in rows if at_resolution(row, sizes, RESOLUTION)])
    conditioned = conditioned_condition(CONDITIONED_OPTIMIZER)
    for benchmark in BENCHMARKS:
        row = selected.get((benchmark, OPTIMIZER_MODEL, False, conditioned))
        if row:
            images[("conditioned", benchmark, OPTIMIZER_MODEL)] = (
                png_of(pngs, row))

    # The reported size is already among the main images.
    ablated = {resolution_condition(size): size for size in RESOLUTIONS
               if size != RESOLUTION}
    for (benchmark, model, _, condition), row in (
            select_resolutions(rows, sizes)[0].items()):
        if condition in ablated:
            images[(f"res{ablated[condition]}", benchmark, model)] = (
                png_of(pngs, row))
    return images


def trained_skill(path: Path) -> str:
    """Read the skill an image was optimized under, as its sidecar names it.

    Args:
        path: The image's best.png.
    Returns:
        The skill text, checked against the digest the sidecar records.
    """
    sidecar = json.loads(path.with_suffix(".json").read_text())
    skill = read_skill(str(RUNS_DIR.parent / sidecar["skill_path"]),
                       "skill_path")
    check_trained_under(str(path), skill)
    return skill


def system_prompt(benchmark: str) -> str:
    """Read the system turn a benchmark's evaluation runs under.

    Args:
        benchmark: Benchmark name, also the stem of its config file.
    Returns:
        The configured system prompt.
    """
    return load_config(CONFIGS_DIR / f"{benchmark}.yaml").benchmark.system_prompt


def controls(side: int) -> dict[str, FullImage]:
    """Make the control canvases of one size.

    Args:
        side: Side of the square canvases.
    Returns:
        Uniform noise and flat gray, by label.
    """
    param = load_config(None).param.model_copy(
        update={"height": side, "width": side})
    return {"noise": noise_param(param, seed=SEED),
            "gray": FullImage(torch.full((3, side, side), GRAY), init="gray")}


def main_shown(images: dict[tuple[str, str], Path], side: int) -> list[dict]:
    """List the main images and their controls with the contexts of each.

    Args:
        images: Skill.png per benchmark and the model it was optimized on.
        side: Side of the control canvases, the size of the own images.
    Returns:
        One entry per image, in the order its questions are sent.
    """
    benchmarks = sorted({benchmark for benchmark, _ in images})
    # Each image under which contexts: no system turn, or a benchmark's.
    shown = [{"label": "skillpng", "variant": "main", "benchmark": benchmark,
              "model": model, "image": load_param(path), "path": str(path),
              "contexts": ("bare", benchmark)}
             for (benchmark, model), path in images.items()]
    shown += [{"label": name, "variant": "main", "benchmark": None,
               "model": None, "image": image, "path": None,
               "contexts": ("bare", *benchmarks)}
              for name, image in controls(side).items()]
    return shown


def ablation_shown(images: dict[tuple[str, str, str], Path]) -> list[dict]:
    """List the ablation images and their controls with each's contexts.

    Args:
        images: Images keyed by variant, benchmark and the model optimized
            on, from ablation_images.
    Returns:
        One entry per image, in the order its questions are sent. An image
        trained under a skill is also asked under that skill; the controls
        of a size under every context an image of that size is asked in.
    """
    shown = []
    by_side: dict[int, set[str]] = {}
    for (variant, benchmark, model), path in sorted(images.items()):
        image = load_param(path)
        contexts = ("bare", benchmark)
        if variant == "conditioned":
            contexts += (benchmark + SKILLED,)
        shown.append({"label": "skillpng", "variant": variant,
                      "benchmark": benchmark, "model": model, "image": image,
                      "path": str(path), "contexts": contexts})
        by_side.setdefault(image.tensor().shape[-1], set()).update(contexts)
    for side, contexts in sorted(by_side.items()):
        shown += [{"label": name, "variant": "control", "benchmark": None,
                   "model": None, "image": image, "path": None,
                   "contexts": ("bare", *sorted(contexts - {"bare"}))}
                  for name, image in controls(side).items()]
    return shown


def requests(shown: list[dict], samples: int, keyed: bool) -> list[dict]:
    """Spell out every question to send: image, context, question and sample.

    Args:
        shown: Images with their contexts, from main_shown or ablation_shown.
        samples: Answers drawn per question.
        keyed: Whether the variant and size seed the option shuffle too, as
            the ablation images of one benchmark and model need; the main
            images keep the shuffles they were first asked with.
    Returns:
        One request per answer, with the image tensor it injects.
    """
    out = []
    for item in shown:
        image = item["image"]
        tensor = image.tensor()
        side = tensor.shape[-1]
        for context in item["contexts"]:
            for question_id, (question, options) in QUESTIONS.items():
                for sample in range(samples):
                    key = (f"{item['label']}|{item['benchmark']}"
                           f"|{item['model']}|{context}|{question_id}"
                           f"|{sample}")
                    if keyed:
                        key += f"|{item['variant']}|{side}"
                    order = list(options) if options else None
                    if order:
                        random.Random(zlib.crc32(key.encode())).shuffle(order)
                    out.append({
                        "image": item["label"],
                        "image_variant": item["variant"],
                        "image_benchmark": item["benchmark"],
                        "image_model": item["model"],
                        "image_path": item["path"],
                        "image_digest": image.digest(),
                        "image_side": side,
                        "context": context, "question_id": question_id,
                        "sample": sample, "options": order,
                        "question": spell(question, options, order),
                        "tensor": tensor,
                    })
    return out


def spell(question: str, options: dict | None,
          order: list[str] | None) -> str:
    """Write a question, with its options lettered in shown order.

    Args:
        question: The question's text.
        options: Option key to text, or None for an open question.
        order: Option keys in the order shown, or None.
    Returns:
        The text of the user turn after the image.
    """
    if not options:
        return question
    lines = [question, ""]
    lines += [f"{LETTERS[i]}) {options[key]}" for i, key in enumerate(order)]
    lines += ["", "Answer with a single letter."]
    return "\n".join(lines)


def parse(output: str, order: list[str] | None,
          question_id: str) -> str | int | None:
    """Read a closed question's answer off the output.

    Under a benchmark's system prompt the answer often comes in that
    benchmark's <answer> tags, after some reasoning, and under TRAJECT's as a
    tool call carrying the letter. The tags are read first, the whole output
    only when they hold nothing readable.

    Args:
        output: The model's answer.
        order: Option keys in the order shown, or None for an open question.
        question_id: Question id.
    Returns:
        The chosen option's key, the 1-10 rating, or None when unreadable or
        the question is open.
    """
    if not order and question_id != "meaning":
        return None
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", output, re.S)
    for text in (*tagged[-1:], re.sub(r"</?answer>", " ", output)):
        answer = (pick_option(text, order) if order
                  else pick_rating(text))
        if answer is not None:
            return answer
    return None


def pick_option(text: str, order: list[str]) -> str | None:
    """Find the option a multiple-choice answer chose.

    Args:
        text: The answer, or what its tags hold.
        order: Option keys in the order shown.
    Returns:
        The chosen option's key, or None when no one option is named.
    """
    letters = LETTERS[:len(order)]
    # A letter standing alone, "B)", "answer is B" or a quoted "B"; a lone
    # capital A can be the article opening an echoed option.
    match = (re.match(rf"\W*([{letters}])(?=[).:,]|\s*$)", text)
             or re.search(rf"\b([{letters}])\)", text)
             or re.search(rf"(?i:answer|option)(?: is)?:?\s*\(?([{letters}])\b",
                          text)
             or re.search(rf"\"([{letters}])\"", text))
    if match:
        return order[LETTERS.index(match.group(1))]
    # The letter twice over, bare and then in tags.
    tokens = set(re.findall(r"\w+", text))
    if len(tokens) == 1 and tokens <= set(letters):
        return order[LETTERS.index(tokens.pop())]
    texts = {**TASKS, **KINDS}
    named = [key for key in order
             if texts[key].lower().rstrip(".") in text.lower()]
    return named[0] if len(named) == 1 else None


def pick_rating(text: str) -> int | None:
    """Find the 1-10 rating a meaningfulness answer gave.

    Args:
        text: The answer, or what its tags hold.
    Returns:
        The rating, or None when the answer holds no number in range.
    """
    # Echoes of the scale, "1 (pure random noise) to 10", are no rating.
    text = re.sub(r"\b1 \(pure random noise\) to 10\b", " ", text)
    match = re.search(r"\b(10|[1-9])\b", text)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="model directory name, e.g. qwen3.5-9B")
    parser.add_argument("--set", dest="overrides", action="append",
                        default=[], help="config override for the client,"
                        " e.g. model.vllm.base_url=...")
    parser.add_argument("--samples", type=int, default=5,
                        help="answers drawn per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=32,
                        help="requests in flight at once")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--images", choices=("main", "ablations"),
                        default="main",
                        help="the main-table images, or the ablation tables'")
    args = parser.parse_args()

    images = reported_images(args.runs_dir)
    found = {model for _, model in images}
    if args.model not in found or set(MAIN_MODELS) - found:
        raise SystemExit(f"no reported Skill.png for"
                         f" {sorted(set(MAIN_MODELS) - found)}")
    prompts = {benchmark: system_prompt(benchmark)
               for benchmark, _ in images}
    if args.images == "main":
        own = {load_param(path).tensor().shape[-1]
               for (_, model), path in images.items() if model == args.model}
        if len(own) != 1:
            raise SystemExit(f"{args.model}'s images differ in size: {own}")
        shown = main_shown(images, *own)
        name = "descriptions.jsonl"
    else:
        ablations = ablation_images(args.runs_dir)
        shown = ablation_shown(ablations)
        for (variant, benchmark, _), path in ablations.items():
            if variant == "conditioned":
                prompts[benchmark + SKILLED] = compose_system_prompt(
                    prompts[benchmark], SKILL_TEMPLATE, trained_skill(path))
        name = "descriptions_ablations.jsonl"

    failed = load_failed_images(args.runs_dir)
    for item in shown:
        if not item["path"]:
            continue
        path = Path(item["path"])
        note = (" (never beat its reference on val)"
                if (item["benchmark"], path.parent.parent.name,
                    item["image"].digest()) in failed else "")
        print(f"{item['variant']} {item['benchmark']} {item['model']}:"
              f" {path}{note}")

    # One client per size sent, as the client resizes to its own.
    clients = {}
    for size in {item["image"].tensor().shape[-1] for item in shown}:
        cfg = load_config(CONFIGS_DIR / "t1.yaml", [
            *args.overrides,
            "model.name=vllm",
            f"model.resolution=[{size}, {size}]",
            f"model.max_new_tokens={args.max_new_tokens}",
            # Sampled, not greedy: base.yaml leaves do_sample off.
            "model.generate.do_sample=true",
            "model.chat_template_kwargs.enable_thinking=false",
        ])
        clients[size] = VLLMClient(cfg.model)
    todo = requests(shown, args.samples, keyed=args.images != "main")

    def ask(request: dict) -> dict:
        """Send one request to the server its image size needs.

        Args:
            request: A requests() entry.
        Returns:
            The answer row, the request without its tensor.
        """
        system = ([] if request["context"] == "bare" else
                  [{"role": "system", "content": prompts[request["context"]]}])
        user = {"role": "user",
                "content": [{"type": "image"},
                            {"type": "text", "text": request["question"]}]}
        client = clients[request["tensor"].shape[-1]]
        output = client.generate(
            [*system, user], request["tensor"],
            temperature=args.temperature,
            seed=SEED + request["sample"])
        row = {key: value for key, value in request.items()
               if key != "tensor"}
        return {"model": args.model, **row, "output": output,
                "answer": parse(output, request["options"],
                                request["question_id"])}

    out = args.runs_dir / "answers" / args.model
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool, \
            open(out / name, "w") as file:
        for done, row in enumerate(pool.map(ask, todo), 1):
            file.write(json.dumps(row) + "\n")
            if done % 100 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} answers", flush=True)
    print(f"Wrote {out / name}")


if __name__ == "__main__":
    main()
