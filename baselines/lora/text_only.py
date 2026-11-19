"""A text-only checkpoint for a family whose forward requires pixels.

vLLM serves LoRA on Nemotron 3 Nano Omni's language model class but not on
the multimodal class around it, so the adapter finetune.py fits there is
served on this: the wrapper's llm_config as config.json, its language_model.*
tensors under their own names, and every other file the checkpoint ships
(tokenizer, chat template, generation config, remote code). The towers are
left out, which a baseline served with --language-model-only never reaches
either.

    uv run python baselines/lora/text_only.py \\
        /ceph/ssd/students/seeo/nemotron-3-nano-omni-30B-A3B-reasoning-BF16 \\
        /ceph/ssd/students/seeo/nemotron-3-nano-omni-30B-A3B-reasoning-BF16-lora/text_only

One shard is held at a time. The output is written beside itself and renamed
into place at the end, and an existing one is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from skillpng.results import git_sha

PREFIX = "language_model."
LLM_CONFIG = "llm_config"
CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
PROVENANCE_NAME = "text_only.json"


def convert(source: Path, out: Path) -> Path:
    """Write the language model of source as a checkpoint of its own.

    Args:
        source: The multimodal checkpoint directory.
        out: Directory the text-only checkpoint goes to.

    Returns:
        The directory written.
    """
    if out.exists():
        raise SystemExit(f"{out} exists; a text-only checkpoint is never overwritten")

    config = json.loads((source / CONFIG_NAME).read_text())
    llm = config.get(LLM_CONFIG)

    if not isinstance(llm, dict) or not llm.get("architectures"):
        raise SystemExit(f"{source / CONFIG_NAME} holds no {LLM_CONFIG} naming an architecture")

    weight_map: dict[str, str] = json.loads((source / INDEX_NAME).read_text())["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items() if key.startswith(PREFIX)})

    if not shards:
        raise SystemExit(f"{source} holds no {PREFIX}* tensors")

    staging = out.with_name(out.name + ".partial")

    # Whatever a killed run left behind.
    if staging.exists():
        shutil.rmtree(staging)

    staging.mkdir(parents=True)
    renamed: dict[str, str] = {}
    total = 0

    for shard in shards:
        with safe_open(str(source / shard), framework="pt") as handle:
            stored = handle.keys()
            keys = [key for key in stored if key.startswith(PREFIX)]
            tensors = {key[len(PREFIX) :]: handle.get_tensor(key) for key in keys}

        save_file(tensors, str(staging / shard), metadata={"format": "pt"})

        for name, tensor in tensors.items():
            renamed[name] = shard
            total += tensor.numel() * tensor.element_size()

        print(f"{shard}: {len(tensors)} tensors", flush=True)

    expected = sum(key.startswith(PREFIX) for key in weight_map)

    if len(renamed) != expected:
        raise SystemExit(f"wrote {len(renamed)} of the {expected} {PREFIX}* tensors")

    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(renamed.items()))}
    (staging / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")

    rewritten = {CONFIG_NAME, INDEX_NAME}

    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix != ".safetensors" and path.name not in rewritten:
            shutil.copy2(path, staging / path.name)

    (staging / CONFIG_NAME).write_text(json.dumps(llm, indent=2) + "\n")
    provenance = {
        "source": str(source),
        "tensors": len(renamed),
        "shards": shards,
        "git_sha": git_sha(),
        "written_utc": datetime.now(UTC).isoformat(),
    }
    (staging / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2) + "\n")
    # One rename, so a kill never leaves a half-written checkpoint in place.
    staging.rename(out)
    print(f"wrote {out}: {len(renamed)} tensors, {total / 1e9:.1f} GB")

    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="the multimodal checkpoint")
    parser.add_argument("out", type=Path, help="where the text-only checkpoint goes")
    args = parser.parse_args(argv)
    convert(args.source, args.out)


if __name__ == "__main__":
    main()
