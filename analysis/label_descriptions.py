"""Label what each model said about an image with an LLM judge.

summarize_descriptions.py reads describe_skillpng.py's answers with regular
expressions: word lists for the open questions and a letter parser for the
closed ones. This asks a judge instead, one answer at a time. The judge sees
the question and the answer, never the image or which image it was, and
returns how the answer says the image looks (one of LOOKS), whether it calls
it noise, blank or corrupted, whether it claims to read text, the concrete
things it claims to see, the subject it ties the image to (one of TOPICS),
whether it says it cannot see images, and, for the closed questions, the
letter or rating it gave.

Reads runs/answers/<model>/<--name>.jsonl and appends to
<--name>_labeled.jsonl beside it, skipping answers labeled already, so an
interrupted run resumes. The key comes from OPENROUTER_API_KEY, else
--api-key-file, and is never printed.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from analyze_runs import RUNS_DIR
from describe_skillpng import MAIN_MODELS
from judge_selfreport import JUDGE, KEY_FILE, ask, read_key

MAX_CHARS = 6000
# How an answer can say the image looks, the judge's one pick.
LOOKS = {
    "blank": "blank, empty, or one uniform solid colour",
    "noise": "random noise, static, grain or random pixels",
    "pattern": "an abstract pattern, texture or abstract art",
    "corrupted": "a glitched, corrupted, distorted or broken image",
    "photo": "a photograph or real-world scene",
    "drawing": "a drawing, diagram, chart, map or interface",
    "text": "a document, page of text, code or writing",
    "none": "the answer does not say how the image looks",
}
# Subjects an answer can tie the image to, the four benchmarks' first.
TOPICS = {
    "travel_code": "travel planning, or writing code that calls travel tools",
    "trivia": "trivia, quizzes, or answering questions from passages",
    "math": "mathematics",
    "tools": "choosing or calling API tools",
    "other": "some other subject",
    "none": "no subject",
}


def schema() -> dict:
    """Build the JSON schema the judge answers in.

    Returns:
        A strict json_schema response format.
    """
    fields = {
        "looks": {"type": "string", "enum": list(LOOKS)},
        "calls_noise": {"type": "boolean"},
        "calls_blank": {"type": "boolean"},
        "calls_corrupted": {"type": "boolean"},
        "reads_text": {"type": "boolean"},
        "objects": {"type": "array", "items": {"type": "string"},
                    "maxItems": 5},
        "topic": {"type": "string", "enum": list(TOPICS)},
        "cannot_see": {"type": "boolean"},
        "choice": {"type": "string"},
    }
    return {"type": "json_schema", "json_schema": {
        "name": "description_label", "strict": True,
        "schema": {"type": "object", "properties": fields,
                   "required": list(fields), "additionalProperties": False}}}


def prompt(row: dict) -> str:
    """Write the judge's prompt for one answer.

    Args:
        row: A descriptions.jsonl row.
    Returns:
        The user turn sent to the judge.
    """
    output = row["output"]
    if len(output) > MAX_CHARS:
        output = output[:MAX_CHARS] + "\n[... cut]"
    context = ("" if row["context"] == "bare" else
               "\nThe model was set up for a task, so its answer may come as"
               " code, a tool call or inside <answer> tags.\n")
    looks = "\n".join(f"  - {key}: {text}" for key, text in LOOKS.items())
    topics = "\n".join(f"  - {key}: {text}" for key, text in TOPICS.items())
    return f"""You are labeling what an AI vision model said about an image.
You do not see the image; judge only from the answer.

The model was shown an image and asked:
<question>
{row["question"]}
</question>
{context}
Its answer:
<answer>
{output}
</answer>

Return JSON with these fields:
- looks: how the answer says the image looks overall, the best fit of:
{looks}
  If the answer only picks an option, use that option when it says how the
  image looks. A rating alone says nothing about how the image looks: give
  "none" and leave the calls_ fields false.
- calls_noise: true if the answer calls the image, or most of it, noise,
  static, grain or random pixels.
- calls_blank: true if the answer calls the image blank, empty, or one
  uniform solid colour.
- calls_corrupted: true if the answer calls the image glitched, corrupted,
  distorted or broken.
- reads_text: true if the answer claims to see or read specific text,
  letters, numbers or symbols in the image (not "there is no text").
- objects: concrete things the answer claims to see in the image (objects,
  people, places, symbols), each a short phrase, at most 5. Leave out
  colours, shapes and textures. Empty if none.
- topic: the subject the answer ties the image to, the best fit of:
{topics}
  Give "other" only when the answer names some other subject.
- cannot_see: true if the answer says it cannot see or process images, or
  declines to look.
- choice: for a multiple-choice question, the letter the answer picks; for
  a rating, the number it gives; otherwise, or if it picks none, "".

Judge only what the answer says; do not guess what the image is."""


def row_key(row: dict) -> tuple:
    """Identify an answer within one model's file.

    Args:
        row: A descriptions.jsonl row.
    Returns:
        Image, its variant, size and owner, context, question and sample.
    """
    return (row["image"], row.get("image_variant"), row.get("image_side"),
            row["image_benchmark"], row["image_model"], row["context"],
            row["question_id"], row["sample"])


def label(client: httpx.Client, key: str, row: dict, model: str) -> dict:
    """Ask the judge to label one answer.

    Args:
        client: HTTP client.
        key: OpenRouter API key.
        row: A descriptions.jsonl row.
        model: OpenRouter model id of the judge.
    Returns:
        The row with the judge's fields under label and its token usage.
    """
    verdict, usage, error = ask(client, key, model, prompt(row), schema())
    if error:
        return {**row, "label": None, "label_model": model,
                "label_error": error}
    return {**row, "label": verdict, "label_model": model,
            "label_usage": usage}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MAIN_MODELS))
    parser.add_argument("--name", default="descriptions",
                        choices=("descriptions", "descriptions_ablations"),
                        help="the answers to label")
    parser.add_argument("--judge", default=JUDGE, help="OpenRouter model id")
    parser.add_argument("--api-key-file", type=Path, default=KEY_FILE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int,
                        help="label at most this many answers per model")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    key = read_key(args.api_key_file)

    with httpx.Client(timeout=180) as client:
        for model in args.models:
            directory = args.runs_dir / "answers" / model
            source = directory / f"{args.name}.jsonl"
            if not source.exists():
                print(f"{model}: no {source.name}, skipped")
                continue
            target = directory / f"{args.name}_labeled.jsonl"
            done = set()
            if target.exists():
                with open(target) as file:
                    done = {row_key(row) for row in map(json.loads, file)
                            if row["label"] is not None}
            with open(source) as file:
                todo = [row for row in map(json.loads, file)
                        if row_key(row) not in done][:args.limit]
            print(f"{model}: {len(todo)} to label, {len(done)} labeled",
                  flush=True)
            tokens = [0, 0]
            cost = 0.0
            failed = 0
            with ThreadPoolExecutor(args.workers) as pool, \
                    open(target, "a") as file:
                for n, row in enumerate(pool.map(
                        lambda row: label(client, key, row, args.judge),
                        todo), 1):
                    file.write(json.dumps(row) + "\n")
                    usage = row.get("label_usage") or {}
                    tokens[0] += usage.get("prompt_tokens", 0)
                    tokens[1] += usage.get("completion_tokens", 0)
                    cost += usage.get("cost") or 0
                    failed += row["label"] is None
                    if n % 200 == 0 or n == len(todo):
                        print(f"  {n}/{len(todo)} labeled, {failed} failed,"
                              f" {tokens[0]:,} in / {tokens[1]:,} out tokens,"
                              f" ${cost:.2f}",
                              flush=True)


if __name__ == "__main__":
    main()
