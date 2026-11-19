"""LLM-Skill: one skill document for one benchmark, written in a single shot.

A strong generator model, reached over OpenRouter, sees the benchmark's task
description, its system prompt and k training examples, and writes one
Markdown skill. It never sees a score or a rollout, and nothing it writes is
revised: skills/<benchmark>/<name>/ holds skill.md and skill.meta.json, and an
existing directory is never overwritten.

    uv run python baselines/llmskill/generate.py \\
        --benchmark searchqa --model openai/gpt-6-astra --k 5 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from skillpng.benchmarks.base import Benchmark, Item
from skillpng.benchmarks.registry import build_benchmark
from skillpng.benchmarks.t1.t1 import T1, fill_prompt
from skillpng.config import Config, load_config
from skillpng.results import git_sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIGS = ROOT / "configs"
PROMPTS = HERE / "prompts"
SKILLS = HERE / "skills"

API = "https://openrouter.ai/api/v1"
# Rate limits and transient upstream failures; any other status is final.
RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})
ATTEMPTS = 6
# Generation stats are recorded shortly after the response arrives.
STATS_ATTEMPTS = 5
STATS_KEYS = (
    "provider_name",
    "total_cost",
    "native_tokens_prompt",
    "native_tokens_completion",
    "native_tokens_reasoning",
)

SYSTEM = (
    "You write skills: compact, reusable Markdown guidance that is added to a language model's system prompt to improve how it performs one task."
)
# {task_prompt} is T1's alone; every other template must not hold it.
SLOT_NAMES = ("{system_prompt}", "{task_prompt}", "{examples}")
SLOTS = re.compile("|".join(map(re.escape, SLOT_NAMES)))
SKILL_TAG = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
FENCE = re.compile(r"\A```[\w-]*\n(.*)\n```\Z", re.DOTALL)
PREAMBLE = re.compile(r"\A(?:here is|here's|below is)[^\n]*:[ \t]*\n", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The parsed flags.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(path.stem for path in PROMPTS.glob("*.md")),
        help="benchmark.name registry key; one template per key in prompts/",
    )
    parser.add_argument(
        "--model", required=True, help="OpenRouter model id, e.g. openai/gpt-6-astra"
    )
    parser.add_argument("--k", type=int, default=5, help="training examples shown; 0 is zero-shot")
    parser.add_argument("--seed", type=int, default=42, help="seed for drawing the examples")
    parser.add_argument("--temperature", type=float, help="sent only when given")
    parser.add_argument("--max-tokens", type=int, help="sent only when given")
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        help="reasoning budget of a thinking model; sent only when given",
    )
    parser.add_argument("--provider", help="pin one OpenRouter provider slug, without fallbacks")
    parser.add_argument(
        "--api-key-file", type=Path, help="read the key here when OPENROUTER_API_KEY is unset"
    )
    parser.add_argument("--lint-n", type=int, default=8, help="n-gram length of the overlap lint")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the prompt and exit without calling the API"
    )
    args = parser.parse_args()

    if args.k < 0:
        parser.error("--k must not be negative")

    if args.lint_n < 1:
        parser.error("--lint-n must be at least 1")

    return args


def resolve_config(name: str) -> tuple[Path, Config]:
    """Find the one experiment config whose benchmark.name is name.

    Args:
        name: The benchmark.name registry key.

    Returns:
        The config's path and the loaded config.
    """
    matches = []

    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_config(path)

        if cfg.benchmark.name == name:
            matches.append((path, cfg))

    if len(matches) != 1:
        found = [str(path.relative_to(ROOT)) for path, _ in matches] or "none"
        raise SystemExit(
            f"expected one config under configs/ with benchmark.name={name!r}, found {found}"
        )

    return matches[0]


def sample_examples(benchmark: Benchmark, k: int, seed: int) -> list[Item]:
    """Draw k training items and prove that none of them is held out.

    Args:
        benchmark: The benchmark built from the resolved config.
        k: Number of examples to draw.
        seed: Seed for the draw.

    Returns:
        The drawn items, in draw order.
    """
    train = benchmark.items("train")

    if k > len(train):
        raise SystemExit(f"--k {k} exceeds the {len(train)} training items")

    sampled = random.Random(seed).sample(train, k)
    held_out = {item.id for split in ("val", "test") for item in benchmark.items(split)}
    leaked = sorted({item.id for item in sampled} & held_out)

    # Not an assert: the check has to survive python -O.
    if leaked:
        raise SystemExit(f"LEAK: example id(s) {leaked} are in val or test; nothing generated")

    return sampled


def user_turn(benchmark: Benchmark, item: Item) -> str:
    """The user turn the target model sees for an item, as text.

    Args:
        benchmark: Benchmark that renders the item.
        item: Item to render.

    Returns:
        The text of the item's user turn, verbatim.
    """
    message = benchmark.build_messages(item, with_image=False)[-1]
    content = message.get("content")

    if message.get("role") != "user" or not isinstance(content, list):
        raise SystemExit(f"{item.id}: the last message is not a user turn of content blocks")

    if any(block.get("type") != "text" for block in content):
        raise SystemExit(f"{item.id}: the user turn carries a block that is not text")

    return "".join(str(block["text"]) for block in content)


def example_fields(benchmark: Benchmark, item: Item, shared: str | None) -> list[tuple[str, str]]:
    """What one example shows the generator, as pairs of tag and text.

    Args:
        benchmark: Benchmark that renders the item.
        item: Item to show.
        shared: T1's task prompt, or None on every other benchmark.

    Returns:
        The user turn, or T1's two fills, followed by the gold target.
    """
    turn = user_turn(benchmark, item)

    if shared is None:
        return [("user_turn", turn), ("gold_answer", item.target)]

    conversation = str(item.meta["conversation"])
    cache = str(item.meta["cache"])

    # Shown apart, the fills must still rebuild the turn the model receives.
    if fill_prompt(shared, conversation, cache) != turn:
        raise SystemExit(f"{item.id}: the user turn is not the task prompt with its fills")

    return [
        ("conversation", conversation),
        ("cache_summary", cache),
        ("gold_answer", item.target),
    ]


def render_examples(examples: list[list[tuple[str, str]]], shared: bool) -> str:
    """The examples section: each example's fields, verbatim.

    Args:
        examples: Each example's pairs of tag and text.
        shared: Whether the examples are fills of a task prompt shown once.

    Returns:
        The section, or an empty string when there are no examples.
    """
    if not examples:
        return ""

    count = "is one task" if len(examples) == 1 else f"are {len(examples)} tasks"
    shows = (
        "the conversation and the cache summary that fill the task prompt "
        "above, exactly as the model receives them, followed by the gold "
        "answer, which scores full marks"
        if shared
        else "the user turn exactly as the model receives it after the system "
        "prompt above, followed by the gold answer the scorer accepts"
    )
    head = (
        "## Examples from the training split\n\n"
        f"Below {count} from the benchmark's training split. Each shows "
        f"{shows}. They are here so that you can see what the tasks really "
        "look like; carry nothing specific from them into the skill."
    )
    blocks = [
        f'<example number="{index}">\n'
        + "\n".join(f"<{tag}>\n{text}\n</{tag}>" for tag, text in fields)
        + "\n</example>"
        for index, fields in enumerate(examples, start=1)
    ]

    return "\n\n".join([head, *blocks])


def render_prompt(template: str, values: dict[str, str]) -> str:
    """Fill a benchmark's template in one pass.

    Args:
        template: The template, each placeholder given a value exactly once.
        values: Placeholder to its text; {examples} is empty for zero-shot.

    Returns:
        The user message sent to the generator.
    """
    for slot in SLOT_NAMES:
        if slot not in values and slot in template:
            raise SystemExit(f"the template must not contain {slot} for this benchmark")

        if slot in values and template.count(slot) != 1:
            raise SystemExit(f"the template must contain {slot} exactly once")

    if not re.search(r"^\{examples\}$", template, flags=re.MULTILINE):
        raise SystemExit("the template must hold {examples} on a line of its own")

    if not values["{examples}"]:
        # Zero-shot: the placeholder line goes with the blank line after it,
        # so no empty section is left behind.
        template = re.sub(r"^\{examples\}\n\n?", "", template, flags=re.MULTILINE)

    return SLOTS.sub(lambda match: values[match.group(0)], template)


def api_key(path: Path | None, variable: str = "OPENROUTER_API_KEY") -> str:
    """The API key: the variable's value, else the file given.

    Args:
        path: File holding the key, or None.
        variable: Environment variable holding the key.

    Returns:
        The key, which is never printed or written anywhere.
    """
    key = os.environ.get(variable, "").strip()

    if not key and path is not None:
        try:
            key = path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise SystemExit(f"cannot read --api-key-file: {error.strerror}") from error

    if not key:
        raise SystemExit(f"no API key: set {variable} or pass --api-key-file")

    return key


def first_choice(payload: dict[str, object]) -> dict[str, object]:
    """The first choice of a chat completion.

    Args:
        payload: The decoded response.

    Returns:
        The choice, or an empty dict when there is none.
    """
    choices = payload.get("choices")

    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}

    return choices[0]


def retry_after(response: httpx.Response, default: float) -> float:
    """How long a refused response asks the client to wait.

    Args:
        response: The rate-limited or unavailable response.
        default: The backoff to use when it names no wait.

    Returns:
        Seconds to wait before the next attempt.
    """
    try:
        return min(max(float(response.headers.get("retry-after", "")), 1.0), 300.0)
    except ValueError:
        return default


def complete(client: httpx.Client, body: dict[str, object]) -> dict[str, object]:
    """Post one chat completion, retrying rate limits and transient errors.

    Every attempt sends the same body, so a retry never alters the prompt.

    Args:
        client: Client carrying the Authorization header.
        body: The request body.

    Returns:
        The decoded response of the first attempt that produced a choice.
    """
    last = ""

    for attempt in range(1, ATTEMPTS + 1):
        wait = min(2.0**attempt, 60.0)

        try:
            response = client.post(f"{API}/chat/completions", json=body)
        except httpx.TransportError as error:
            last = f"{type(error).__name__}: {error}"
        else:
            if response.status_code == 200:
                payload: dict[str, object] = response.json()
                choice = first_choice(payload)

                if choice and choice.get("finish_reason") != "error":
                    return payload

                last = f"provider error: {choice.get('error') or payload.get('error')}"
            elif response.status_code in RETRYABLE:
                last = f"HTTP {response.status_code}: {response.text[:500]}"
                wait = retry_after(response, wait)
            else:
                raise SystemExit(
                    f"OpenRouter refused the request (HTTP {response.status_code}): "
                    f"{response.text[:500]}"
                )

        if attempt < ATTEMPTS:
            print(f"attempt {attempt} failed ({last}); retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)

    raise SystemExit(f"OpenRouter failed {ATTEMPTS} times; last: {last}")


def generation_stats(client: httpx.Client, generation_id: str) -> dict[str, object]:
    """Look up which provider served a generation and what it cost.

    Args:
        client: Client carrying the Authorization header.
        generation_id: The id of the chat completion.

    Returns:
        The stats record, or an empty dict when it never became available.
    """
    for attempt in range(1, STATS_ATTEMPTS + 1):
        time.sleep(2.0 * attempt)

        try:
            response = client.get(f"{API}/generation", params={"id": generation_id})
        except httpx.TransportError:
            continue

        if response.status_code == 200:
            data = response.json().get("data")

            if isinstance(data, dict):
                return data

    return {}


def extract_skill(content: str) -> tuple[str, str]:
    """Pull the skill out of the generator's reply.

    Takes the last skill tag pair when there is one, else the whole reply,
    then drops a leading preamble line such as Here is the skill: and a code
    fence around the whole document.

    Args:
        content: The reply's message content.

    Returns:
        The skill text and how it was found, tags or untagged.
    """
    tagged = SKILL_TAG.findall(content)
    text = PREAMBLE.sub("", (tagged[-1] if tagged else content).strip(), count=1).strip()
    fenced = FENCE.match(text)

    if fenced:
        text = fenced.group(1).strip()

    return text, "tags" if tagged else "untagged"


def ngram_overlap(skill: str, sources: list[str], n: int) -> dict[str, object]:
    """Measure the word n-grams the skill shares verbatim with the examples.

    Reports only; the skill is never edited on its account.

    Args:
        skill: The skill text.
        sources: Every example's user turn and gold target.
        n: N-gram length in words.

    Returns:
        Counts, the shared fraction and the shared n-grams.
    """

    def grams(text: str) -> set[tuple[str, ...]]:
        """Every run of n words, compared case-insensitively.

        Args:
            text: Text to split into words.

        Returns:
            The set of n-word tuples.
        """
        words = re.findall(r"\w+", text.casefold())
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

    own = grams(skill)
    seen: set[tuple[str, ...]] = set()

    for source in sources:
        seen |= grams(source)

    shared = sorted(" ".join(gram) for gram in own & seen)

    return {
        "n": n,
        "skill_ngrams": len(own),
        "shared_ngrams": len(shared),
        "shared_fraction": len(shared) / len(own) if own else 0.0,
        "shared": shared,
    }


def skill_name(model: str, k: int, seed: int, effort: str | None) -> str:
    """The directory name: the model without its provider, effort, k and seed.

    Args:
        model: OpenRouter model id, e.g. openai/gpt-6-astra.
        k: Number of examples shown.
        seed: Seed the examples were drawn with.
        effort: Reasoning effort asked for, or None when none was.

    Returns:
        A name such as gpt-6-astra_medium_k5_seed42.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model.rsplit("/", 1)[-1]).strip("-")
    # Left out when unset, so a model asked for no reasoning keeps a name
    # with no empty slot in it.
    parts = [slug, effort, f"k{k}", f"seed{seed}"]

    return "_".join(part for part in parts if part)


def main() -> None:
    """Generate one skill, or with --dry-run only print its prompt."""
    args = parse_args()
    config_path, cfg = resolve_config(args.benchmark)

    if cfg.benchmark.limit is not None:
        # A capped split would leave the held-out check partial.
        raise SystemExit(f"{config_path.name} sets benchmark.limit; the splits must be whole")

    benchmark = build_benchmark(cfg.benchmark)
    items = sample_examples(benchmark, args.k, args.seed)
    shared = benchmark.template if isinstance(benchmark, T1) else None
    examples = [example_fields(benchmark, item, shared) for item in items]
    template = (PROMPTS / f"{args.benchmark}.md").read_text(encoding="utf-8")
    values = {
        "{system_prompt}": cfg.benchmark.system_prompt,
        "{examples}": render_examples(examples, shared is not None),
    }

    if shared is not None:
        values["{task_prompt}"] = shared

    user = render_prompt(template, values)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    if args.dry_run:
        print(f"===== system =====\n{SYSTEM}\n\n===== user =====\n{user}")
        ids = [item.id for item in items]
        print(f"{len(SYSTEM) + len(user)} characters; examples {ids}", file=sys.stderr)
        return

    out_dir = (
        SKILLS / args.benchmark / skill_name(args.model, args.k, args.seed, args.reasoning_effort)
    )

    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; a written skill is frozen and never overwritten")

    routing: dict[str, object] = {"require_parameters": True}

    if args.provider:
        routing.update(order=[args.provider], allow_fallbacks=False)

    body: dict[str, object] = {"model": args.model, "messages": messages, "provider": routing}

    if args.temperature is not None:
        body["temperature"] = args.temperature

    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens

    # OpenRouter's unified field; it maps to whatever the provider calls it.
    if args.reasoning_effort is not None:
        body["reasoning"] = {"effort": args.reasoning_effort}

    headers = {"Authorization": f"Bearer {api_key(args.api_key_file)}"}
    sha = git_sha()
    timestamp = datetime.now(UTC).isoformat()

    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=30.0), headers=headers) as client:
        payload = complete(client, body)
        generation_id = str(payload.get("id") or "")
        stats = generation_stats(client, generation_id) if generation_id else {}

    choice = first_choice(payload)
    message = choice.get("message")
    content = str(message.get("content") or "") if isinstance(message, dict) else ""
    finish = choice.get("finish_reason")

    if finish != "stop":
        # A cut-off document is not a skill, and nothing is regenerated.
        raise SystemExit(
            f"generation {generation_id} ended with finish_reason={finish!r}; nothing written"
        )

    skill, extraction = extract_skill(content)

    if not skill:
        raise SystemExit(f"generation {generation_id} returned no skill; nothing written")

    if not stats.get("provider_name"):
        print(
            f"warning: no provider recorded; look up generation {generation_id} later",
            file=sys.stderr,
        )

    lint = ngram_overlap(skill, [text for fields in examples for _, text in fields], args.lint_n)
    meta: dict[str, object] = {
        "benchmark": args.benchmark,
        "config_path": str(config_path.relative_to(ROOT)),
        "generator_model": args.model,
        "served_model": payload.get("model"),
        "provider": stats.get("provider_name"),
        "provider_routing": routing,
        "generation_id": generation_id,
        "k": args.k,
        "seed": args.seed,
        "example_ids": [item.id for item in items],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning": body.get("reasoning"),
        "messages": messages,
        "usage": payload.get("usage"),
        "generation_stats": {key: stats.get(key) for key in STATS_KEYS},
        "finish_reason": finish,
        "timestamp_utc": timestamp,
        "git_sha": sha,
        "skill_sha256": hashlib.sha256(skill.encode()).hexdigest(),
        "skill_chars": len(skill),
        "skill_words": len(skill.split()),
        "extraction": extraction,
        "raw_content": content,
        "lint": lint,
    }

    out_dir.mkdir(parents=True)
    (out_dir / "skill.md").write_text(skill + "\n", encoding="utf-8")
    (out_dir / "skill.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {out_dir.relative_to(ROOT)}: {meta['skill_words']} words; "
        f"{lint['shared_ngrams']} of {lint['skill_ngrams']} {args.lint_n}-grams "
        "also occur in the examples"
    )


if __name__ == "__main__":
    main()
