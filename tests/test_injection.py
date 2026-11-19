"""Where the image sits in the message list."""

from __future__ import annotations

import pytest

from skillpng.benchmarks.base import inject_image

MESSAGES = [
    {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
    {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
]


def test_pre_query_puts_the_image_before_the_question() -> None:
    out = inject_image(MESSAGES, "pre_query")
    assert out[1]["content"][0] == {"type": "image"}
    assert out[1]["content"][1]["text"] == "What is 2+2?"


def test_post_query_puts_it_after() -> None:
    out = inject_image(MESSAGES, "post_query")
    assert out[1]["content"][-1] == {"type": "image"}


def test_the_system_message_is_left_alone() -> None:
    """A system message cannot carry an image block, so injection never touches it."""
    for site in ("pre_query", "post_query"):
        out = inject_image(MESSAGES, site)
        assert out[0] == MESSAGES[0]


def test_a_system_site_is_rejected() -> None:
    with pytest.raises(ValueError, match="injection site"):
        inject_image(MESSAGES, "system")


def test_messages_without_a_user_turn_are_rejected() -> None:
    with pytest.raises(ValueError, match="needs a user message"):
        inject_image([MESSAGES[0]], "pre_query")


def test_string_content_is_upgraded_to_blocks() -> None:
    out = inject_image([{"role": "user", "content": "hello"}], "pre_query")
    assert out[0]["content"] == [{"type": "image"}, {"type": "text", "text": "hello"}]


def test_input_is_not_mutated() -> None:
    original = [dict(message) for message in MESSAGES]
    inject_image(MESSAGES, "pre_query")
    assert original == MESSAGES


def test_exactly_one_image_block_is_added() -> None:
    out = inject_image(MESSAGES, "pre_query")
    blocks = [
        block
        for message in out
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "image"
    ]
    assert len(blocks) == 1


def test_unknown_site_is_rejected() -> None:
    with pytest.raises(ValueError, match="injection site"):
        inject_image(MESSAGES, "somewhere")
