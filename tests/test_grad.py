"""Exactly one leaf in the graph carries gradient, and it is the image.

The fixture is a real (tiny) `nn.Module` rather than a fake VLM: the thing being tested is that
freezing parameters and routing a differentiable tensor through a torch graph actually leaves one
leaf, which a mock would assert into existence rather than demonstrate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from skillpng.image.param import FullImage
from skillpng.image.preprocess import PreprocessSpec, preprocess_image
from skillpng.models.base import target_span, token_ce


class TinyVLM(nn.Module):
    """A vision tower + projector + language head, small enough to run in a unit test."""

    def __init__(self, vocab: int = 16, dim: int = 8, patch: int = 4) -> None:
        super().__init__()
        self.tower = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.projector = nn.Linear(dim, dim)
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids: Tensor, pixel_values: Tensor) -> Tensor:
        features = self.tower(pixel_values).flatten(2).transpose(1, 2)
        features = self.projector(features).mean(dim=1, keepdim=True)
        embeddings = self.embed(input_ids) + features  # the image reaches every position
        logits: Tensor = self.head(embeddings)
        return logits


def _graph_leaves(tensor: Tensor) -> list[Tensor]:
    """Walk the autograd graph and collect every tensor that would accumulate gradient."""
    seen: set[int] = set()
    stack = [tensor.grad_fn]
    leaves: list[Tensor] = []
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if type(node).__name__ == "AccumulateGrad":
            leaves.append(getattr(node, "variable"))  # noqa: B009
        stack.extend(child for child, _ in node.next_functions)
    return leaves


def _build_loss(model: TinyVLM, image: Tensor) -> tuple[Tensor, Tensor]:
    spec = PreprocessSpec(image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5))
    pixel_values = preprocess_image(image, spec, (16, 16))
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    logits: Tensor = model(input_ids, pixel_values)
    return token_ce(logits, input_ids, target_span(4, 2)), input_ids


def test_exactly_one_leaf_and_it_is_the_image() -> None:
    torch.manual_seed(0)
    model = TinyVLM()
    model.requires_grad_(False)  # what HFVLM does at load
    model.eval()

    param = FullImage(torch.rand(3, 16, 16))
    image = param.tensor()
    loss, _ = _build_loss(model, image)

    leaves = _graph_leaves(loss)
    assert len(leaves) == 1, [tuple(leaf.shape) for leaf in leaves]
    assert leaves[0] is param.raw


def test_backward_puts_gradient_on_the_image_and_nowhere_else() -> None:
    torch.manual_seed(0)
    model = TinyVLM()
    model.requires_grad_(False)
    param = FullImage(torch.rand(3, 16, 16))

    loss, _ = _build_loss(model, param.tensor())
    loss.backward()

    assert param.raw.grad is not None
    assert torch.isfinite(param.raw.grad).all()
    assert float(param.raw.grad.abs().sum()) > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_unfrozen_model_would_add_leaves() -> None:
    """Guards the guard: the leaf count is not vacuously one."""
    torch.manual_seed(0)
    model = TinyVLM()  # parameters left trainable
    param = FullImage(torch.rand(3, 16, 16))
    loss, _ = _build_loss(model, param.tensor())
    assert len(_graph_leaves(loss)) > 1
