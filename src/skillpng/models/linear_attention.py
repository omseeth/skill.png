"""FLA's kernel under the gated delta rule layers of Qwen3.5 and its kin.

Qwen3.5 interleaves full attention with gated delta rule layers, three of
every four. transformers runs those on FLA's chunked kernel when it can
import it and otherwise on a torch implementation of its own.

transformers looks the kernel up once, when a family's modeling file is
first imported, by walking fla.ops.gated_delta_rule.chunk_gated_delta_rule
attribute by attribute from the fla package. fla-core's __init__ imports
none of its ops, so on its own that walk ends at None and the fallback is
taken without a word. The ops are imported here before the load, and what
each loaded family ended on is checked after it.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys

import torch
from torch import nn

log = logging.getLogger(__name__)

# The module transformers walks to for the chunked kernel.
_FLA_OPS = "fla.ops.gated_delta_rule"
_FLA_KERNEL = "chunk_gated_delta_rule"
# What each family binds its kernel-or-fallback function to, and what it
# names the layer calling it: Qwen3.5, its MoE, Qwen3-Next and OLMo hybrid.
_RULE = "torch_chunk_gated_delta_rule"
_MIXER_SUFFIX = "GatedDeltaNet"


def load_fla_kernels() -> bool:
    """Import FLA's gated delta rule ops, so the load's lookup finds them.

    Has to run before anything imports the modeling file. Where the ops do
    not import (no triton, as on macOS), the layers take the fallback, and
    verify_fla_fast_path decides whether that is allowed.

    Returns:
        True when the ops imported.
    """
    try:
        importlib.import_module(_FLA_OPS)
    except Exception as error:
        # Not only ImportError: triton can refuse the platform at import.
        log.info("%s is not importable here (%s)", _FLA_OPS, error)
        return False

    return True


def verify_fla_fast_path(model: nn.Module, device: torch.device) -> list[str]:
    """Refuse a CUDA run whose gated delta rule layers took the fallback.

    The fallback costs memory and time, not accuracy, so a CPU run is let
    through on it with a note. On CUDA it is what ran a single-sequence T1
    backward out of an H200, and it looks healthy until that backward, so
    it is raised on at the load. Models without such layers pass.

    Args:
        model: The loaded model, of any family.
        device: The device the model is about to run on.

    Returns:
        Modeling modules confirmed on FLA's kernel.
    """
    fla = getattr(sys.modules.get(_FLA_OPS), _FLA_KERNEL, None)
    confirmed: list[str] = []
    fallback: list[str] = []

    for name in _mixer_modules(model):
        kernel = _kernel_behind(getattr(sys.modules.get(name), _RULE, None))
        (confirmed if fla is not None and kernel is fla else fallback).append(name)

    if fallback and device.type == "cuda":
        raise RuntimeError(
            "the gated delta rule layers in "
            + ", ".join(fallback)
            + " run transformers' torch implementation, whose backward holds"
            " every chunk's fp32 intermediates: "
            + _why_not_fla()
            + ". Run uv sync to install fla-core, and check that"
            " load_fla_kernels ran before the model was loaded."
        )

    if fallback:
        log.info("gated delta rule on the torch implementation in: %s", ", ".join(fallback))

    if confirmed:
        log.info("FLA's chunked gated delta rule confirmed in: %s", ", ".join(confirmed))

    return confirmed


def _mixer_modules(model: nn.Module) -> list[str]:
    """The modeling modules defining the model's gated delta rule layers.

    Args:
        model: The loaded model, of any family.

    Returns:
        Their names in sys.modules, each once.
    """
    names: list[str] = []

    for module in model.modules():
        kind = type(module)

        if kind.__name__.endswith(_MIXER_SUFFIX) and kind.__module__ not in names:
            names.append(kind.__module__)

    return names


def _kernel_behind(rule: object) -> object:
    """The function a transformers kernel wrapper hands its calls to.

    use_kernel_func_from_hub_with_fallback resolves it once, at import, and
    keeps it in the wrapper's closure as implementation.

    Args:
        rule: A family's torch_chunk_gated_delta_rule, or None.

    Returns:
        The function its calls reach, or None when that cannot be read.
    """
    if not inspect.isfunction(rule):
        return None

    return inspect.getclosurevars(rule).nonlocals.get("implementation")


def _why_not_fla() -> str:
    """Tell a missing FLA apart from one imported too late.

    Args:
        None.

    Returns:
        One clause naming the cause.
    """
    try:
        importlib.import_module(_FLA_OPS)
    except Exception as error:
        return f"{_FLA_OPS} does not import here ({type(error).__name__}: {error})"

    return f"{_FLA_OPS} imports, but only after the modeling file had looked for it"
