"""The decoder layers split across GPUs, and nothing else about the model changes.

A ~13k-token T1 sequence outgrows one H200 in its backward, so model.num_gpus
cuts the decoder layers into contiguous runs, one per GPU. The unit tests use
the meta device as the other GPU wherever a tensor only has to arrive
somewhere; the integration test needs two real ones.
"""

from __future__ import annotations

import copy
from collections import namedtuple

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from skillpng.models.placement import (
    _Arrive,
    _Leave,
    decoder_layers,
    gpus,
    pin_layer_states,
    run_of,
    split_layers,
    to_device,
)

META = torch.device("meta")
CPU = torch.device("cpu")


class Block(nn.Module):
    """A decoder layer holding a ModuleList of experts, as an MoE layer does."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.experts = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(2))

    def forward(
        self, hidden: Tensor, scale: Tensor | None = None, past_key_values: object = None
    ) -> Tensor:
        mixed = self.proj(hidden) + self.experts[0](hidden) + self.experts[1](hidden)
        out: Tensor = hidden + mixed * (scale if scale is not None else 1.0)
        return out


class Toy(nn.Module):
    """A vision tower, embeddings, a layer stack and a head, all tiny."""

    def __init__(self, vocab: int = 11, dim: int = 8, layers: int = 5) -> None:
        super().__init__()
        self.tower = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(Block(dim) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: Tensor, image: Tensor, cache: object = None) -> Tensor:
        hidden = self.embed(ids) + self.tower[1](self.tower[0](image))
        # Built once on the first device and handed to every layer, as the
        # rotary embeddings and masks are.
        scale = torch.full((1,), 0.5, device=hidden.device)

        for layer in self.layers:
            hidden = layer(hidden, scale=scale, past_key_values=cache)

        logits: Tensor = self.head(self.norm(hidden))

        return logits


class HybridCache:
    """Nemotron's cache shape: every layer's Mamba state allocated up front."""

    def __init__(self, layers: int, device: torch.device) -> None:
        self.conv_states = {i: torch.zeros(1, 4, device=device) for i in range(layers)}
        self.ssm_states = {i: torch.zeros(1, 2, device=device) for i in range(layers)}


def test_the_heaviest_module_list_is_the_layer_stack() -> None:
    name, layers = decoder_layers(Toy())
    assert name == "layers"
    assert len(layers) == 5


def test_a_model_without_a_module_list_is_refused() -> None:
    with pytest.raises(ValueError, match="no ModuleList"):
        decoder_layers(nn.Linear(2, 2))


def test_runs_are_contiguous_and_near_equal() -> None:
    devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    # Nemotron 3 Nano's 52 layers and Qwen3.5-9B's 32 halve exactly.
    for count in (52, 32):
        runs = [run_of(i, count, devices).index for i in range(count)]
        assert runs == [0] * (count // 2) + [1] * (count // 2)
    assert [run_of(i, 5, devices).index for i in range(5)] == [0, 0, 0, 1, 1]


def test_a_split_asks_for_cuda() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        gpus(CPU, 2)


def test_a_split_wider_than_the_visible_gpus_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="gres"):
        gpus(torch.device("cuda"), 2)


def test_the_split_starts_at_the_configured_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert gpus(torch.device("cuda"), 2) == [torch.device("cuda", 0), torch.device("cuda", 1)]
    assert gpus(torch.device("cuda", 1), 2) == [torch.device("cuda", 1), torch.device("cuda", 2)]


def test_tensors_move_through_tuples_lists_and_dicts() -> None:
    cos_sin = (torch.ones(2), torch.ones(2))
    moved = to_device({"pair": cos_sin, "list": [torch.ones(1)], "flag": True}, META)
    assert isinstance(moved, dict)
    assert isinstance(moved["pair"], tuple)
    assert all(t.device == META for t in moved["pair"])
    assert moved["list"][0].device == META
    assert moved["flag"] is True


def test_a_container_subclass_is_passed_through() -> None:
    Pair = namedtuple("Pair", "first second")
    pair = Pair(torch.ones(1), torch.ones(1))
    assert to_device(pair, META) is pair


def test_a_hybrid_cache_state_follows_its_layer() -> None:
    cache = HybridCache(3, CPU)
    pin_layer_states(cache, 1, META)
    assert cache.conv_states[1].device == META
    assert cache.ssm_states[1].device == META
    assert cache.conv_states[0].device == CPU
    assert cache.ssm_states[2].device == CPU


def test_a_cache_without_layer_states_is_left_alone() -> None:
    pin_layer_states(None, 0, META)
    pin_layer_states(object(), 0, META)


def test_a_layer_receives_its_inputs_and_its_cache_state_on_its_device() -> None:
    cache = HybridCache(3, CPU)
    hook = _Arrive(2, META)
    args, kwargs = hook(nn.Identity(), (torch.ones(1),), {"past_key_values": cache})
    assert isinstance(args[0], Tensor)
    assert args[0].device == META
    assert kwargs["past_key_values"] is cache
    assert cache.conv_states[2].device == META
    assert cache.conv_states[1].device == CPU


def test_the_last_layer_hands_back_to_the_first_device() -> None:
    output = _Leave(META)(nn.Identity(), (), torch.ones(1))
    assert isinstance(output, Tensor)
    assert output.device == META


def test_a_split_model_computes_what_the_whole_one_does() -> None:
    torch.manual_seed(0)
    whole = Toy()
    split = copy.deepcopy(whole)
    placed = split_layers(split, [CPU, CPU])
    assert sorted(placed) == list(range(5))

    ids = torch.tensor([[1, 2, 3, 4]])
    image = torch.rand(1, 8)
    torch.testing.assert_close(split(ids, image), whole(ids, image))


@pytest.mark.integration
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_two_gpus_give_the_loss_and_image_gradient_of_one() -> None:
    first, second = torch.device("cuda", 0), torch.device("cuda", 1)
    torch.manual_seed(0)
    whole = Toy().requires_grad_(False)
    split = copy.deepcopy(whole)
    whole.to(first)
    split_layers(split, [first, second])

    assert split.embed.weight.device == first
    assert split.head.weight.device == first
    bottom, top = split.layers[0], split.layers[-1]
    assert isinstance(bottom, Block) and isinstance(top, Block)
    assert bottom.proj.weight.device == first
    assert top.proj.weight.device == second

    ids = torch.tensor([[1, 2, 3, 4]], device=first)
    losses, grads = [], []
    cache = HybridCache(5, first)

    for model in (whole, split):
        image = torch.rand(1, 8, generator=torch.Generator().manual_seed(1)).to(first)
        image.requires_grad_(True)
        logits = model(ids, image, cache=cache)
        loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
        loss.backward()
        losses.append(loss.detach())
        grads.append(image.grad)

    torch.testing.assert_close(losses[1], losses[0])
    torch.testing.assert_close(grads[1], grads[0])
    assert cache.conv_states[0].device == first
    assert cache.conv_states[4].device == second
