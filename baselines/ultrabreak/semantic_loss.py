"""UltraBreak's semantic loss, in place of CE on the target tokens.

A port of semantic_similarity_loss with mode "attention" from
github.com/kaiyuanCui/UltraBreak (optimisation/utils.py). The embedding
expected under the model's next-token distribution is pulled, by cosine
similarity, toward an attention-weighted mix of the current and every later
target embedding. Close but wrong tokens thus cost less than under CE.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# The reference's constants: attention temperature, weight of the positional
# encoding in the attention scores, and std of the noise on the expected
# embeddings.
TAU = 0.5
PE_ALPHA = 0.01
NOISE = 1e-4


def sinusoidal_encoding(positions: Tensor, dim: int) -> Tensor:
    """Encode sequence positions as sines and cosines of geometric rates.

    Args:
        positions: Absolute sequence positions, shape (T,).
        dim: Width of the encoding.

    Returns:
        The (T, dim) encoding, sines in even and cosines in odd columns.
    """
    rates = torch.exp(
        torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    angles = positions.to(torch.float32).unsqueeze(1) * rates
    encoding = torch.zeros(positions.shape[0], dim, device=positions.device)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : dim // 2])

    return encoding


def semantic_loss(
    logits: Tensor,
    input_ids: Tensor,
    span: tuple[int, int],
    embeddings: Tensor,
    tau: float = TAU,
    pe_alpha: float = PE_ALPHA,
    noise: float = NOISE,
) -> Tensor:
    """Mean of 1 - cos(expected embedding, attended target) over the target.

    Logits shorter than the sequence are read as its last positions, as
    token_ce reads them.

    Args:
        logits: Model logits (B,T,V), or (B,K,V) over the last K positions.
        input_ids: Prompt and target ids, shape (B,T).
        span: Half-open start and end index of the target in input_ids.
        embeddings: The model's input embedding matrix, shape (V,D).
        tau: Temperature of the attention over target positions.
        pe_alpha: Weight of the positional encoding in the attention scores.
        noise: Std of the Gaussian noise added to the expected embeddings.

    Returns:
        The scalar loss, averaged over target positions.
    """
    start, end = span
    offset = input_ids.shape[1] - logits.shape[1]

    if not 0 <= offset < start < end <= input_ids.shape[1]:
        raise ValueError(
            f"logits {tuple(logits.shape)} do not cover the positions predicting "
            f"span {span} of ids {tuple(input_ids.shape)}"
        )

    if logits.shape[-1] != embeddings.shape[0]:
        raise ValueError(
            f"logits score {logits.shape[-1]} tokens but the embedding matrix "
            f"has {embeddings.shape[0]} rows"
        )

    table = embeddings.detach()
    # Position i predicts token i + 1, so the target is read one earlier.
    rows = logits[:, start - 1 - offset : end - 1 - offset].float()
    probs = torch.softmax(rows, dim=-1)
    expected = (probs.to(table.dtype) @ table).float()
    expected = expected + noise * torch.randn_like(expected)
    targets = table[input_ids[:, start:end]].float()

    positions = torch.arange(start - 1, end - 1, device=rows.device)
    encoding = pe_alpha * sinusoidal_encoding(positions, table.shape[1])
    queries = F.normalize(expected + encoding, dim=-1)
    keys = F.normalize(targets + encoding, dim=-1)
    scores = queries @ keys.transpose(1, 2) / tau

    # Each position attends to its own target token and every later one.
    length = end - start
    later = torch.ones(length, length, dtype=torch.bool, device=rows.device).triu()
    weights = torch.softmax(scores.masked_fill(~later, float("-inf")), dim=-1)
    attended = weights @ targets
    similarity = F.cosine_similarity(expected, attended, dim=-1)

    return (1.0 - similarity).mean()
