"""The patched Nemotron chat template the vLLM servers are started with.

vLLM's openai content format hands the template every system turn as text
blocks, and the checkpoint's own template printed those as a Python repr.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

ROOT = Path(__file__).resolve().parents[1]
NEMOTRON = "nemotron-3-nano-omni-30B-A3B-reasoning"
TEMPLATE = ROOT / "slurm" / "chat_templates" / f"{NEMOTRON}.jinja"
SYSTEM = "Answer with one letter.\nNo explanation."
USER = [{"type": "image"}, {"type": "text", "text": "Which one?"}]


def _render(system: str | list[dict[str, str]], enable_thinking: bool = True) -> str:
    """Render a system turn and a user turn with the patched template.

    Args:
        system: System turn content, a string or text blocks.
        enable_thinking: Thinking toggle passed to the template.

    Returns:
        The rendered prompt.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": USER},
    ]
    # Annotated as str, it returns (prompts, generation indices).
    prompts = render_jinja_template(
        conversations=[messages],
        chat_template=TEMPLATE.read_text(),
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )[0]

    return prompts[0]


def _serving_scripts() -> list[Path]:
    """Find the Nemotron slurm scripts that start a vLLM server.

    Returns:
        Paths of those scripts.
    """
    scripts = sorted(ROOT.glob(f"slurm/*/{NEMOTRON}/*.sh"))

    return [script for script in scripts if 'bin/vllm" serve' in script.read_text()]


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_text_blocks_render_like_the_plain_string(enable_thinking: bool) -> None:
    """vLLM's block-wrapped system turn renders as the HF plain string does.

    Args:
        enable_thinking: Thinking toggle passed to the template.
    """
    blocks = [{"type": "text", "text": SYSTEM}]
    assert _render(blocks, enable_thinking) == _render(SYSTEM, enable_thinking)


def test_system_turn_is_its_text_not_a_repr() -> None:
    """The system preamble carries the raw text with real newlines."""
    rendered = _render([{"type": "text", "text": SYSTEM}])
    assert rendered.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n")
    assert "{'type'" not in rendered


def test_every_nemotron_server_gets_the_patched_template() -> None:
    """A server started without it would bring the repr back."""
    scripts = _serving_scripts()
    assert scripts
    expected = f'CHAT_TEMPLATE="{TEMPLATE.relative_to(ROOT)}"'
    for script in scripts:
        text = script.read_text()
        assert expected in text, script
        assert '--chat-template "${CHAT_TEMPLATE}"' in text, script
