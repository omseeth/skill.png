"""Split one model's decoder layers across several GPUs.

A T1 item is ~13k tokens, and the graph its backward keeps outgrows one H200
on its own: training already accumulates one sequence at a time, so a batch
split across GPUs would not help. A second GPU helps only by holding part of
that one graph, so the decoder layers are cut into contiguous runs, one per
GPU. Everything else (the vision tower, the embeddings, the final norm and
the head) stays on the first, where the image and the ids live, so the logits
and the image gradient come out there too.

Hidden states cross between runs through forward hooks on the layers, and
autograd carries the gradient back along the same copies. Nothing about the
math changes, only where each layer runs.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)

# Per-layer states a checkpoint's own hybrid cache allocates on one device up
# front. The Mamba decoding kernels update them in place, and a state on
# another GPU than its layer is a pointer they cannot read.
_LAYER_STATES = ("conv_states", "ssm_states")


def gpus(device: torch.device, count: int) -> list[torch.device]:
    """Resolve the consecutive GPUs a split model spans.

    Checked before the weights load, which takes minutes, not after.

    Args:
        device: The configured device, which must be a CUDA one.
        count: How many GPUs to span.

    Returns:
        count consecutive CUDA devices, starting at device.
    """
    if device.type != "cuda":
        raise ValueError(
            f"model.num_gpus={count} splits the layers across CUDA devices, "
            f"but model.device resolved to {device}"
        )

    start = device.index or 0
    visible = torch.cuda.device_count()

    if start + count > visible:
        raise ValueError(
            f"model.num_gpus={count} from cuda:{start} needs {start + count} "
            f"visible GPUs and this process sees {visible}: request them with "
            "#SBATCH --gres=gpu:N"
        )

    return [torch.device("cuda", start + offset) for offset in range(count)]


def decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Find the stack of decoder layers: the model's heaviest ModuleList.

    The vision tower's blocks and each MoE layer's experts are ModuleLists
    too, but the language model's layers outweigh the former and contain
    the latter.

    Args:
        model: The loaded model.

    Returns:
        The qualified name of the layer stack, and the stack itself.
    """
    found: tuple[str, nn.ModuleList] | None = None
    heaviest = 0

    for name, module in model.named_modules():

        if not isinstance(module, nn.ModuleList):
            continue

        size = sum(param.numel() for param in module.parameters())

        if size > heaviest:
            found, heaviest = (name, module), size

    if found is None:
        raise ValueError("this model has no ModuleList of layers to split across GPUs")

    return found


def run_of(index: int, layers: int, devices: list[torch.device]) -> torch.device:
    """Pick the device of one layer: contiguous, near-equal runs, in order.

    Args:
        index: Position of the layer in the stack.
        layers: Number of layers in the stack.
        devices: The devices split across, first one first.

    Returns:
        The device holding that layer.
    """
    return devices[index * len(devices) // layers]


def split_layers(model: nn.Module, devices: list[torch.device]) -> dict[int, torch.device]:
    """Place the decoder layers in runs across devices, the rest on the first.

    The whole model reaches the first device before the later runs move off
    it, which asks no more of that GPU than the single-device load did: the
    weights fit one GPU, it is the activations that do not.

    Args:
        model: The loaded model, on any single device.
        devices: The devices to span; the first holds all but the layers.

    Returns:
        The device of each layer, by position in the stack.
    """
    name, layers = decoder_layers(model)
    first = devices[0]
    placed = {index: run_of(index, len(layers), devices) for index in range(len(layers))}
    model.to(first)

    for index, layer in enumerate(layers):
        layer.to(placed[index])
        layer.register_forward_pre_hook(_Arrive(index, placed[index]), with_kwargs=True)

    # The final norm and the head sit on the first device, so the last
    # layer's output goes back there.
    layers[-1].register_forward_hook(_Leave(first))

    runs = ", ".join(
        f"{device} layers {min(held)}-{max(held)}"
        for device in devices
        if (held := [index for index, at in placed.items() if at == device])
    )
    log.info("split %d layers of %s: %s; everything else on %s", len(layers), name, runs, first)

    return placed


def to_device(value: object, device: torch.device) -> object:
    """Move every tensor in value to device, through tuples, lists and dicts.

    Only the exact container types are rebuilt: a subclass such as a
    namedtuple or a model output is passed through as it is.

    Args:
        value: A layer argument or output.
        device: Where its tensors should be.

    Returns:
        The same structure with its tensors on device.
    """
    if isinstance(value, Tensor):
        return value.to(device)

    if type(value) is tuple:
        return tuple(to_device(item, device) for item in value)

    if type(value) is list:
        return [to_device(item, device) for item in value]

    if type(value) is dict:
        return {key: to_device(item, device) for key, item in value.items()}

    return value


def pin_layer_states(cache: object, index: int, device: torch.device) -> None:
    """Move a hybrid cache's state for one layer onto that layer's device.

    Nemotron 3 Nano Omni's own cache allocates every Mamba layer's conv and
    ssm state on the embeddings' device before the first step. The prefill
    only copies into them, which crosses devices silently; the decoding
    kernels then update them in place and would be handed the wrong GPU's
    memory. Moved at the layer's first call, a state stays with its layer.
    transformers' own caches allocate lazily, per layer, and have neither.

    Args:
        cache: The past_key_values a layer is called with, or None.
        index: Position of the layer in the stack.
        device: The device holding the layer.
    """
    for name in _LAYER_STATES:
        states = getattr(cache, name, None)

        if not isinstance(states, dict):
            continue

        state = states.get(index)

        if isinstance(state, Tensor) and state.device != device:
            states[index] = state.to(device)


class _Arrive:
    """Forward pre-hook that brings a layer's inputs onto its device."""

    def __init__(self, index: int, device: torch.device) -> None:
        """Remember which layer this hook serves and where it runs.

        Args:
            index: Position of the layer in the stack.
            device: The device holding the layer.
        """
        self.index = index
        self.device = device

    def __call__(
        self, module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        """Move the tensor arguments and this layer's cached states over.

        Args:
            module: The layer about to run.
            args: Its positional arguments.
            kwargs: Its keyword arguments.

        Returns:
            The arguments, every tensor among them on the layer's device.
        """
        pin_layer_states(kwargs.get("past_key_values"), self.index, self.device)
        moved_args = tuple(to_device(value, self.device) for value in args)
        moved_kwargs = {key: to_device(value, self.device) for key, value in kwargs.items()}

        return moved_args, moved_kwargs


class _Leave:
    """Forward hook that returns the last layer's output to the first device."""

    def __init__(self, device: torch.device) -> None:
        """Remember where the modules after the stack run.

        Args:
            device: The first device, holding the final norm and the head.
        """
        self.device = device

    def __call__(
        self, module: nn.Module, args: tuple[object, ...], output: object
    ) -> object:
        """Move the layer's output to the first device.

        Args:
            module: The last layer, which just ran.
            args: Its positional arguments.
            output: What it returned.

        Returns:
            The output, every tensor in it on the first device.
        """
        return to_device(output, self.device)
