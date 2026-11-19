"""VLM contract, plus the label-masking and cross-entropy it is built on."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from ..image.preprocess import PreprocessSpec

IGNORE_INDEX = -100


def target_span(prompt_len: int, target_len: int) -> tuple[int, int]:
    """Half-open [start, end) of the target in cat(prompt_ids, target_ids).

    Prompt and target are tokenized separately and concatenated, so the
    boundary is exact rather than recovered by string matching:
    retokenization can move a boundary across a merge.

    Args:
        prompt_len: Number of prompt tokens.
        target_len: Number of target tokens.

    Returns:
        Start and end index of the target span.
    """
    if prompt_len < 1:
        raise ValueError("prompt must be non-empty: nothing to condition on")
    if target_len < 1:
        raise ValueError("target must be non-empty: there would be no loss")
    return prompt_len, prompt_len + target_len


def masked_labels(
    input_ids: Tensor, span: tuple[int, int], ignore_index: int = IGNORE_INDEX
) -> Tensor:
    """Build labels that keep the span and ignore everything else.

    Args:
        input_ids: Token ids of shape (B, T).
        span: Half-open start and end index of the target.
        ignore_index: Label value for positions outside the span.

    Returns:
        Labels equal to input_ids inside span, ignore_index elsewhere.
    """
    start, end = span

    if input_ids.dim() != 2:
        raise ValueError(f"expected (B,T) input_ids, got {tuple(input_ids.shape)}")

    length = input_ids.shape[1]

    if not 0 <= start < end <= length:
        raise ValueError(f"span {span} out of range for sequence length {length}")
    labels = torch.full_like(input_ids, ignore_index)
    labels[:, start:end] = input_ids[:, start:end]

    return labels


def token_ce(
    logits: Tensor, input_ids: Tensor, span: tuple[int, int], ignore_index: int = IGNORE_INDEX
) -> Tensor:
    """Mean teacher-forced CE over the target span only.

    The causal shift is done here instead of passing labels to the model, so
    the convention stays visible and testable without weights: position i-1
    predicts token i.

    Logits shorter than the sequence are read as its last positions, which is
    what a forward given logits_to_keep hands back. What that leaves out is
    unused either way, so both call shapes give the same loss.

    Args:
        logits: Model logits of shape (B, T, V), or (B, K, V) for the last K positions.
        input_ids: Token ids of shape (B, T).
        span: Half-open start and end index of the target.
        ignore_index: Label value for positions outside the span.

    Returns:
        Scalar cross-entropy averaged over target positions.
    """
    if logits.shape[0] != input_ids.shape[0] or logits.shape[1] > input_ids.shape[1]:
        raise ValueError(
            f"logits {tuple(logits.shape)} do not align with ids {tuple(input_ids.shape)}"
        )

    # Positions the logits leave out at the front; zero for a full pass.
    offset = input_ids.shape[1] - logits.shape[1]

    if offset >= span[0]:
        raise ValueError(
            f"logits cover the last {logits.shape[1]} of {input_ids.shape[1]} positions, "
            f"which begins past the one predicting the first token of span {span}"
        )

    labels = masked_labels(input_ids, span, ignore_index)
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, offset + 1 :]

    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
    )


class VLM(ABC):
    """A frozen, local, white-box vision-language model.

    Implementations must guarantee that model parameters have
    requires_grad=False, and that the image tensor handed to loss_on_target
    is the only leaf carrying gradient.
    """

    @abstractmethod
    def loss_on_target(
        self, messages: list[dict[str, object]], image: Tensor | None, target: str
    ) -> Tensor:
        """Teacher-forced CE on target only, differentiable w.r.t. image.

        Args:
            messages: Chat turns in the processor's message format.
            image: Float image in [0,1], (3,H,W), requires_grad=True.
            target: Continuation the loss is computed on.

        Returns:
            Scalar cross-entropy on the target tokens.
        """

    @abstractmethod
    def generate(
        self, messages: list[dict[str, object]], image: Tensor | None, **kw: object
    ) -> str:
        """Continue the prompt, greedily by default.

        Args:
            messages: Chat turns in the processor's message format.
            image: Float image in [0,1], (3,H,W), or None.
            **kw: Backend-specific generation overrides.

        Returns:
            Decoded continuation without the prompt.
        """

    @property
    @abstractmethod
    def preprocess(self) -> PreprocessSpec:
        """Preprocessing derived from the loaded processor's config.

        Args:
            None.

        Returns:
            Spec describing how images are resized and normalized.
        """

    @property
    def image_token_count(self) -> int | None:
        """Image tokens the pinned resolution produces.

        Returns:
            Token count once a forward has established it, else None.
        """
        return None

    @property
    def trainable(self) -> bool:
        """Can a loss taken through this model carry gradient to the image?

        False for a backend that only hands back text, which the training
        loop checks before it spends anything.

        Returns:
            True unless loss_on_target cannot produce a gradient.
        """
        return True

    def describe(self) -> dict[str, object]:
        """Row material for results: whatever identifies this instance.

        Returns:
            Mapping of identifying fields for a results row.
        """
        return {"model": type(self).__name__, "model_dtype": "unknown"}
