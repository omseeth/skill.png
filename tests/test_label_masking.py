"""The loss must cover the target span and nothing else.

Get this wrong and you are optimizing the image partly against the prompt — and the loss still
goes down, so nothing else in the system will tell you.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from skillpng.models.base import IGNORE_INDEX, masked_labels, target_span, token_ce


def test_target_span_is_the_appended_region() -> None:
    assert target_span(5, 3) == (5, 8)
    with pytest.raises(ValueError):
        target_span(0, 3)
    with pytest.raises(ValueError):
        target_span(5, 0)


def test_masked_labels_covers_exactly_the_span() -> None:
    ids = torch.arange(10).unsqueeze(0)
    labels = masked_labels(ids, (6, 9))
    assert (labels[0, :6] == IGNORE_INDEX).all()
    assert labels[0, 6:9].tolist() == [6, 7, 8]
    assert labels[0, 9] == IGNORE_INDEX


def test_masked_labels_rejects_bad_spans() -> None:
    ids = torch.arange(6).unsqueeze(0)
    for span in [(0, 0), (3, 2), (4, 99), (-1, 3)]:
        with pytest.raises(ValueError):
            masked_labels(ids, span)


def test_token_ce_matches_manual_cross_entropy_on_the_span() -> None:
    torch.manual_seed(0)
    vocab, length = 11, 9
    ids = torch.randint(0, vocab, (1, length))
    logits = torch.randn(1, length, vocab)
    span = (5, 8)

    # position i-1 predicts token i
    manual = F.cross_entropy(logits[0, 4:7], ids[0, 5:8])
    assert torch.allclose(token_ce(logits, ids, span), manual, atol=1e-6)


def test_prompt_positions_do_not_contribute() -> None:
    torch.manual_seed(0)
    vocab, length = 7, 10
    ids = torch.randint(0, vocab, (1, length))
    logits = torch.randn(1, length, vocab)
    span = (6, 9)
    baseline = token_ce(logits, ids, span)

    # Everything strictly before the position that predicts the first target token is unused...
    perturbed = logits.clone()
    perturbed[:, :5] = torch.randn_like(perturbed[:, :5]) * 10
    assert torch.allclose(token_ce(perturbed, ids, span), baseline, atol=1e-6)

    # ...as is the tail beyond the span.
    perturbed = logits.clone()
    perturbed[:, 8:] = torch.randn_like(perturbed[:, 8:]) * 10
    assert torch.allclose(token_ce(perturbed, ids, span), baseline, atol=1e-6)

    # ...but the position that predicts the first target token is very much used.
    # (Note it has to be one logit, not the whole row: a constant shift is a softmax no-op.)
    perturbed = logits.clone()
    perturbed[:, 5, 0] += 100.0
    assert not torch.allclose(token_ce(perturbed, ids, span), baseline, atol=1e-3)


def test_loss_falls_to_zero_when_the_target_is_predicted_confidently() -> None:
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    span = (3, 5)
    logits = torch.zeros(1, 5, 6)
    for position in range(4):
        logits[0, position, ids[0, position + 1]] = 50.0
    assert float(token_ce(logits, ids, span)) < 1e-6


def test_trailing_logits_give_the_same_loss_as_the_full_pass() -> None:
    torch.manual_seed(0)
    vocab, length = 11, 8
    ids = torch.randint(0, vocab, (1, length))
    logits = torch.randn(1, length, vocab)
    span = (5, 8)
    keep = span[1] - span[0] + 1  # what loss_on_target asks a forward for

    assert torch.allclose(
        token_ce(logits[:, -keep:], ids, span), token_ce(logits, ids, span), atol=1e-6
    )


def test_a_window_that_cuts_into_the_span_is_refused() -> None:
    # One position short: nothing left predicts the first target token, and
    # silently scoring the rest would quietly shrink the objective.
    torch.manual_seed(0)
    ids = torch.randint(0, 7, (1, 8))
    logits = torch.randn(1, 8, 7)
    span = (5, 8)

    with pytest.raises(ValueError, match="begins past"):
        token_ce(logits[:, -3:], ids, span)
