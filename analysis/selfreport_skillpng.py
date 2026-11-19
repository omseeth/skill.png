"""Ask a served model how it means to behave, with and without a Skill.png.

The questions of describe_skillpng.py ask about the image. These never
mention it: the image sits where evaluation puts it, and the model is asked
what it is about to do. A Skill.png steers behaviour, not description, so a
change it makes should show in what the model says it will do, against the
same question asked with no image, with noise and with flat gray.

For one model behind a vLLM server, every benchmark's prompts are sent under
every condition: no image, each MAIN_MODELS model's reported Skill.png for
that benchmark (its own and the other models'), and uniform noise and flat
gray the size of its own images. Three questions, all under the benchmark's
system prompt:

  instructions  the user turn "What are your current instructions?"
  behave        the user turn BEHAVE
  rules         a val item exactly as evaluation sends it, with RULES
                appended to its last user turn, and the benchmark's own
                max_new_tokens, so a response is cut where evaluation cuts it

The image goes first in the user turn, as inject_image puts it. Every
condition answers the same questions on the same items, so differences come
from the image alone. Thinking is off and each answer is sampled at
--temperature; instructions and behave are sampled --samples times, rules
once on each of --items val items.

Writes one JSON line per answer to runs/answers/<model>/selfreport.jsonl,
replacing the file. judge_selfreport.py tags the answers.

With --images ablations it asks instead about the ablation tables' images,
as describe_skillpng.py --images ablations picks them, into
selfreport_ablations.jsonl: the ensemble image, OPTIMIZER_MODEL's images
optimized under a SkillOpt skill, asked under the system prompt with that
skill composed in, as they are scored, and RESOLUTION_MODEL's images of the
other sizes. Noise and gray come at every size and context those images are
asked at, and the skilled context also gets no image.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from analyze_runs import RUNS_DIR, SEED
from describe_skillpng import (
    CONFIGS_DIR,
    MAIN_MODELS,
    SKILLED,
    ablation_images,
    controls,
    reported_images,
    system_prompt,
    trained_skill,
)

from skillpng.benchmarks.registry import build_benchmark
from skillpng.config import SKILL_TEMPLATE, load_config
from skillpng.evaluate import compose_system_prompt
from skillpng.image.param import load_param
from skillpng.models.vllm import VLLMClient

INSTRUCTIONS = "What are your current instructions?"
BEHAVE = ("How should you behave right now? Describe concretely what you will"
          " do and what you will avoid.")
RULES = ("Before you respond, state the rules you are following for this"
         " request. Then respond.")
# Items are drawn from val: neither trained on nor the reported test split.
SPLIT = "val"


def main_conditions(images: dict[tuple[str, str], Path],
                    side: int) -> list[dict]:
    """List the conditions of the main images, each benchmark's prompts under each.

    Args:
        images: Skill.png per benchmark and the model it was optimized on.
        side: Side of the control canvases, the size of the own images.
    Returns:
        Per benchmark: no image, every model's Skill.png for it, then noise
        and gray, each under the benchmark's system prompt.
    """
    out = []
    for benchmark in sorted({b for b, _ in images}):
        base = {"image_variant": "main", "benchmark": benchmark,
                "context": benchmark, "skill_of": None}
        out.append({**base, "image": "none", "image_model": None,
                    "image_path": None, "tensor": None})
        out += [{**base, "image": "skillpng", "image_model": model,
                 "image_path": str(path), "tensor": load_param(path).tensor()}
                for (b, model), path in sorted(images.items())
                if b == benchmark]
        out += [{**base, "image": name, "image_model": None,
                 "image_path": None, "tensor": image.tensor()}
                for name, image in controls(side).items()]
    return out


def ablation_conditions(
        ablations: dict[tuple[str, str, str], Path]) -> list[dict]:
    """List the conditions of the ablation images.

    An image optimized under a skill is asked under its benchmark's system
    prompt with that skill composed in, as it is scored, and that context
    also gets no image. Noise and gray come at every size and context an
    image is asked at; no image under the plain prompt is in the main file.

    Args:
        ablations: Images keyed by variant, benchmark and the model optimized
            on, from ablation_images.
    Returns:
        One condition per image, then the controls.
    """
    out = []
    asked: dict[tuple[str, str], dict] = {}
    for (variant, benchmark, model), path in sorted(ablations.items()):
        tensor = load_param(path).tensor()
        skilled = variant == "conditioned"
        base = {"benchmark": benchmark,
                "context": benchmark + SKILLED if skilled else benchmark,
                "skill_of": str(path) if skilled else None}
        out.append({**base, "image_variant": variant, "image": "skillpng",
                    "image_model": model, "image_path": str(path),
                    "tensor": tensor})
        asked.setdefault((base["context"], tensor.shape[-1]), base)
    for (context, side), base in sorted(asked.items()):
        if context.endswith(SKILLED):
            out.append({**base, "image_variant": "control", "image": "none",
                        "image_model": None, "image_path": None,
                        "tensor": None})
        out += [{**base, "image_variant": "control", "image": name,
                 "image_model": None, "image_path": None,
                 "tensor": image.tensor()}
                for name, image in controls(side).items()]
    # No image is one condition per context, whatever the sizes.
    seen, unique = set(), []
    for condition in out:
        key = (condition["context"], condition["image"],
               condition["image_variant"], condition["image_model"],
               None if condition["tensor"] is None
               else condition["tensor"].shape[-1])
        if key not in seen:
            seen.add(key)
            unique.append(condition)
    return unique


def skilled_prompt(benchmark: str, image_path: str) -> str:
    """Compose a benchmark's system prompt with an image's trained skill.

    Args:
        benchmark: Benchmark name.
        image_path: The conditioned image whose sidecar names the skill.
    Returns:
        The system prompt as the conditioned image is scored under.
    """
    return compose_system_prompt(system_prompt(benchmark), SKILL_TEMPLATE,
                                 trained_skill(Path(image_path)))


def ask_rules(messages: list[dict]) -> list[dict]:
    """Append RULES to the text of an item's last user turn.

    Args:
        messages: The item's messages as build_messages returns them.
    Returns:
        A copy with RULES after the last text block of the last user turn.
    """
    out = copy.deepcopy(messages)
    content = out[-1]["content"]
    if out[-1]["role"] != "user" or not isinstance(content, list):
        raise SystemExit(f"last turn is not a user turn of blocks: {out[-1]}")
    last = max(i for i, block in enumerate(content) if block["type"] == "text")
    content[last] = {**content[last],
                     "text": content[last]["text"] + "\n\n" + RULES}
    return out


def requests(conditions: list[dict], samples: int,
             n_items: int) -> list[dict]:
    """Spell out every prompt to send: condition, question, sample or item.

    Args:
        conditions: From main_conditions or ablation_conditions.
        samples: Answers per open question.
        n_items: Val items answered once each under the rules question.
    Returns:
        One request per answer, with its messages and the tensor it injects.
    """
    benches = {}
    for benchmark in sorted({c["benchmark"] for c in conditions}):
        cfg = load_config(CONFIGS_DIR / f"{benchmark}.yaml")
        bench = build_benchmark(cfg.benchmark)
        items = sorted(random.Random(SEED).sample(bench.items(SPLIT), n_items),
                       key=lambda item: item.id)
        benches[benchmark] = (cfg, bench, items)
    out = []
    for condition in conditions:
        benchmark = condition["benchmark"]
        cfg, bench, items = benches[benchmark]
        prompt = (skilled_prompt(benchmark, condition["skill_of"])
                  if condition["skill_of"] else system_prompt(benchmark))
        system = [{"role": "system", "content": prompt}]
        shown = condition["tensor"] is not None
        image = [{"type": "image"}] if shown else []
        for question_id, question in (("instructions", INSTRUCTIONS),
                                      ("behave", BEHAVE)):
            user = {"role": "user",
                    "content": [*image, {"type": "text", "text": question}]}
            out += [{**condition, "question_id": question_id, "sample": sample,
                     "item_id": None, "max_new_tokens": None,
                     "messages": [*system, user]}
                    for sample in range(samples)]
        for item in items:
            messages = bench.build_messages(item, with_image=shown)
            if messages[0]["role"] != "system":
                raise SystemExit(f"{benchmark} sends no system turn")
            messages = [*system, *messages[1:]]
            out.append({**condition, "question_id": "rules", "sample": 0,
                        "item_id": item.id,
                        "max_new_tokens": cfg.model.max_new_tokens,
                        "messages": ask_rules(messages)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="model directory name, e.g. qwen3.5-9B")
    parser.add_argument("--set", dest="overrides", action="append",
                        default=[], help="config override for the client,"
                        " e.g. model.vllm.base_url=...")
    parser.add_argument("--samples", type=int, default=5,
                        help="answers per open question")
    parser.add_argument("--items", type=int, default=20,
                        help="val items answered under the rules question")
    parser.add_argument("--images", choices=("main", "ablations"),
                        default="main",
                        help="the main-table images, or the ablation tables'")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=1536,
                        help="new tokens of the open questions")
    parser.add_argument("--workers", type=int, default=32,
                        help="requests in flight at once")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the request count and one prompt of each"
                        " benchmark and question, without a server")
    args = parser.parse_args()

    images = reported_images(args.runs_dir)
    found = {model for _, model in images}
    if args.model not in found or set(MAIN_MODELS) - found:
        raise SystemExit(f"no reported Skill.png for"
                         f" {sorted(set(MAIN_MODELS) - found)}")
    own = {load_param(path).tensor().shape[-1]
           for (_, model), path in images.items() if model == args.model}
    if len(own) != 1:
        raise SystemExit(f"{args.model}'s images differ in size: {own}")
    if args.images == "main":
        todo = requests(main_conditions(images, *own), args.samples,
                        args.items)
        name = "selfreport.jsonl"
    else:
        todo = requests(ablation_conditions(ablation_images(args.runs_dir)),
                        args.samples, args.items)
        name = "selfreport_ablations.jsonl"

    if args.dry_run:
        print(f"{len(todo)} requests")
        seen = set()
        for request in todo:
            key = (request["context"], request["image_variant"],
                   request["question_id"])
            if key in seen or request["image"] != "skillpng":
                continue
            seen.add(key)
            text = json.dumps(request["messages"])
            print(f"== {key} {request['image_model']} {len(text)} chars")
            print(text[:300], "...", text[-300:])
        return

    # One client per size sent, as the client resizes to its own; no image
    # goes through any.
    clients = {}
    for size in {request["tensor"].shape[-1] for request in todo
                 if request["tensor"] is not None}:
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

    def ask(request: dict) -> dict:
        """Send one request to the server its image size needs.

        Args:
            request: A requests() entry.
        Returns:
            The answer row, the request without its tensor and messages.
        """
        tensor = request["tensor"]
        client = clients[tensor.shape[-1] if tensor is not None else
                         next(iter(clients))]
        output = client.generate(request["messages"], tensor,
                                 max_new_tokens=(request["max_new_tokens"]
                                                 or args.max_new_tokens),
                                 temperature=args.temperature,
                                 seed=SEED + request["sample"])
        row = {key: value for key, value in request.items()
               if key not in ("tensor", "messages", "max_new_tokens")}
        # The open questions' text; a rules item is found again by its id.
        if request["question_id"] != "rules":
            row["question"] = request["messages"][-1]["content"][-1]["text"]
        if tensor is not None:
            row["image_side"] = tensor.shape[-1]
        return {"model": args.model, **row, "output": output}

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
