"""Remote checkpoint code written against an older transformers, run on this one.

A trust-remote-code checkpoint ships its own modeling file, pinned to the
transformers it was released against. Nemotron 3 Nano Omni's calls the mask
factories by the keyword names they lost in 5.4, and hands transformers a
cache class of its own that predates the query offset 5.14 asks every cache
for. Neither is a difference in the math, and the same file needs 5.6 or
newer elsewhere, so both are fixed at the call boundary rather than by
holding transformers back.

The third one is not a call at all: a buffer its vision tower computes in
its own __init__ does not survive the load this transformers does, and is
repaired after it.

The fourth is a call boundary again, and the one with teeth. The file asks
the hub for causal-conv1d and mamba-ssm, names 5.15's mapping does not
carry, and reads the None it gets back as permission to run a naive torch
scan instead. That one does change the math, so the lookup is answered from
the wheels installed here and the answer is checked rather than trusted.

The fifth is a declaration the file never makes. transformers reads sdpa
support off a class attribute that defaults to False, so the kernel is
refused however it is asked for and the load falls back to eager, which
holds a whole attention matrix per layer for the backward and is what a 5k
token sequence runs out of memory on. The declaration is repaired for the
classes whose source shows they dispatch through the attention interface,
and the kernel is then counted at the first call rather than assumed.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

# 5.4 renamed the embeddings argument the mask factories take, and 5.12
# dropped the cache_position 5.4 had already stopped reading.
_RENAMED = {"input_embeds": "inputs_embeds"}
_DROPPED = ("cache_position",)
_PATCHED = "_skillpng_legacy_kwargs"
_KERNEL_PATCHED = "_skillpng_hub_kernels"
# The Mamba2 kernels the Nemotron hybrid asks the hub for, mapped to the
# modules the wheels in make install-mamba put on the path. Submodules are
# listed because the checkpoint walks to its functions by attribute, and an
# unimported submodule is not an attribute of its package.
_HUB_KERNELS = {
    "causal-conv1d": ("causal_conv1d",),
    "mamba-ssm": (
        "mamba_ssm",
        "mamba_ssm.ops.triton.selective_state_update",
        "mamba_ssm.ops.triton.ssd_combined",
    ),
}


def patch_mask_factories() -> list[str]:
    """Let the transformers mask factories take their older keyword names.

    Returns:
        The factories wrapped by this call, empty when they already were.
    """
    from transformers import masking_utils

    wrapped: list[str] = []

    for name in dir(masking_utils):

        if not name.startswith("create_"):
            continue

        func = getattr(masking_utils, name)

        if not callable(func) or getattr(func, _PATCHED, False):
            continue

        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            continue

        if "inputs_embeds" not in params:
            continue

        _rebind(name, func, _legacy_kwargs(func))
        wrapped.append(name)

    if wrapped:
        log.info("mask factories now take the older argument names: %s", ", ".join(wrapped))

    return wrapped


def patch_remote_caches() -> list[str]:
    """Give a checkpoint's own cache class the query offset the masks ask it for.

    Returns:
        Qualified names of the cache classes patched by this call.
    """
    patched: list[str] = []

    for name, module in list(sys.modules.items()):

        if not name.startswith("transformers_modules."):
            continue

        for value in list(vars(module).values()):

            if not _is_legacy_cache(value, name):
                continue

            value.get_query_offset = _get_query_offset
            patched.append(f"{name}.{value.__name__}")

    if patched:
        log.info("remote cache classes given get_query_offset: %s", ", ".join(patched))

    return patched


def patch_hub_kernels() -> list[str]:
    """Answer the Mamba2 kernel lookups from the wheels installed here.

    5.15's _HUB_KERNEL_MAPPING registers three kernels and neither Mamba
    one, so the checkpoint's lazy_load_kernel returns None for both, its
    is_fast_path_available comes out False, and every Mamba block takes a
    naive torch scan that is slower, tens of GB heavier in the backward, and
    numerically different. Installing the wheels does not help on its own:
    nothing in that path ever looks at them.

    Returns:
        The kernel names this call answers, empty when it patched nothing.
    """
    from transformers.integrations import hub_kernels

    original = hub_kernels.lazy_load_kernel

    if getattr(original, _KERNEL_PATCHED, False):
        return []

    mapping = getattr(hub_kernels, "_HUB_KERNEL_MAPPING", {})
    loaded: dict[str, object] = {}

    for name, paths in _HUB_KERNELS.items():

        # A name transformers knows is its business, not ours.
        if name in mapping:
            continue

        try:
            for path in paths:
                importlib.import_module(path)
        except ImportError as error:
            log.warning(
                "%s is not importable here (%s), so the Mamba2 fast path"
                " stays off: run make install-mamba",
                name,
                error,
            )
            continue

        loaded[name] = sys.modules[paths[0]]

    if not loaded:
        return []

    @functools.wraps(original)
    def shim(kernel_name: str, *args: object, **kwargs: object) -> object:
        """Hand back the installed package, or fall through to the hub.

        Args:
            kernel_name: The kernel the caller is asking for.
            args: Positional arguments the original takes.
            kwargs: Keyword arguments the original takes.

        Returns:
            The module providing the kernel, or the original's own answer.
        """
        if kernel_name in loaded:
            return loaded[kernel_name]

        return original(kernel_name, *args, **kwargs)

    setattr(shim, _KERNEL_PATCHED, True)
    _rebind("lazy_load_kernel", original, shim)
    log.info("mamba kernels served from the installed wheels: %s", ", ".join(loaded))

    return list(loaded)


def verify_mamba_fast_path() -> list[str]:
    """Refuse a run whose Mamba blocks settled on the naive torch scan.

    The fallback warns once and then looks exactly like a healthy run, which
    is how it reached a training job and cost it an OOM, so it is raised on
    rather than logged. Modules with no such flag are not Mamba hybrids and
    are passed over.

    Returns:
        Qualified names of the remote modules confirmed on the fast path.
    """
    fast: list[str] = []
    slow: list[str] = []

    for name, module in list(sys.modules.items()):

        if not name.startswith("transformers_modules."):
            continue

        available = getattr(module, "is_fast_path_available", None)

        if available is None:
            continue

        (fast if available else slow).append(name)

    if slow:
        raise RuntimeError(
            "the Mamba2 fast path is off in "
            + ", ".join(slow)
            + ": the naive torch scan changes the numerics and holds a"
            " backward graph tens of GB larger. Run make install-mamba, and"
            " check that patch_hub_kernels ran before the checkpoint's"
            " modeling code was imported."
        )

    if fast:
        log.info("mamba fast path confirmed in: %s", ", ".join(fast))

    return fast


def promote_to_sdpa(model: Any) -> dict[str, str]:
    """Set sdpa on a loaded model whose class never declared support for it.

    Support is a property of the code and not of the flag: attention that
    goes through the transformers attention interface reads the
    implementation off its config at every forward and runs whichever one
    it names. transformers can answer that question about a checkpoint's
    own file, from its source, and _can_set_attn_implementation is the
    answer. The flag is repaired on the classes that pass it and left alone
    on the rest, which set_attn_implementation then skips rather than
    raises on, and which keep the eager masks their eager config asks for.

    Args:
        model: The loaded model, of any family.

    Returns:
        The implementation each submodel ended on, by qualified name.
    """
    submodels = _submodels(model)

    for _, module in submodels:

        if module._can_set_attn_implementation():
            type(module)._supports_sdpa = True

    model.set_attn_implementation("sdpa")
    landed = {name: str(module.config._attn_implementation) for name, module in submodels}
    named = ", ".join(f"{name or '<model>'}={impl}" for name, impl in landed.items())

    if "sdpa" not in landed.values():
        raise RuntimeError(
            "sdpa was asked for and nothing on this checkpoint took it: "
            f"{named}. Its attention does not go through the interface that "
            "reads the implementation off the config, so there is no kernel "
            "here to switch. Run it with model.attn_implementation=eager and "
            "cut the sequence length instead."
        )

    log.info("attention implementation after the promotion: %s", named)

    return landed


@contextmanager
def require_sdpa(what: str) -> Iterator[None]:
    """Run one model call and refuse it if it reached no sdpa kernel.

    A class can pass every check transformers makes and still take an eager
    path of its own, and the difference is not only speed: the mask
    factories answer a config saying sdpa with None wherever a plain causal
    mask can be left to the kernel, and an eager path handed None attends
    over the whole sequence in both directions. That is a loss taken on a
    model that read its own continuation, which looks like a healthy run.
    So the kernel is counted, once, and a call that reached none of it is
    raised on rather than optimized from.

    Args:
        what: The call being guarded, named in the log line and the error.
    """
    import torch

    original = torch.nn.functional.scaled_dot_product_attention
    calls = [0]

    @functools.wraps(original)
    def counting(*args: Any, **kwargs: Any) -> Any:
        """Count this call and make it.

        Args:
            args: Positional arguments the kernel takes.
            kwargs: Keyword arguments it takes.

        Returns:
            Whatever the kernel returns.
        """
        calls[0] += 1

        return original(*args, **kwargs)

    # Rebound rather than set on torch.nn.functional alone: a caller that
    # imported the name is calling its own reference to it.
    _rebind("scaled_dot_product_attention", original, counting)

    try:
        yield
    finally:
        _rebind("scaled_dot_product_attention", counting, original)

    if not calls[0]:
        raise RuntimeError(
            f"the config says sdpa and {what} reached no sdpa kernel: this "
            "family runs an attention path of its own that ignores the "
            "implementation its config names. The masks were built for a "
            "kernel that never ran, so the loss this produces is not the one "
            "that was asked for. Run it with model.attn_implementation=eager "
            "instead."
        )

    log.info("sdpa confirmed: %s reached the kernel %d times", what, calls[0])


def repair_radio_summary_idxs(model: Any) -> list[str]:
    """Rebuild the C-RADIO tower's summary index buffer from its own config.

    summary_idxs is computed in the tower's __init__ from the teachers its
    config names, and no checkpoint carries it. This transformers builds the
    model on the meta device and materializes what the checkpoint does not
    hold, so the computed value is replaced by whatever the allocation held
    -- the load report says as much, calling it MISSING and newly
    initialized. The tower then indexes its class tokens with those numbers.
    Out of range, that is a device-side assert, raised on CUDA by whichever
    kernel runs next rather than at the read itself.

    Args:
        model: The loaded model, of any family.

    Returns:
        Qualified names of the buffers this call rebuilt.
    """
    import torch

    rebuilt: list[str] = []

    for name, module in model.named_modules():
        teachers = _teachers(module)

        if not teachers:
            continue

        wanted = [i for i, teacher in enumerate(teachers) if teacher.get("use_summary", True)]

        for buffer_name, buffer in list(module.named_buffers()):

            if buffer_name.split(".")[-1] != "summary_idxs":
                continue

            # A buffer still on the meta device holds no value to read, and
            # is exactly the case this repairs.
            held = "nothing (meta)" if buffer.is_meta else buffer.tolist()

            if held == wanted:
                continue

            parent = buffer_name.rsplit(".", 1)[0] if "." in buffer_name else ""
            owner = module.get_submodule(parent) if parent else module
            device = torch.device("cpu") if buffer.is_meta else buffer.device
            log.warning(
                "%s.%s came back from the load holding %s, which does not "
                "index this tower's %d class tokens; rebuilding it as %s",
                name,
                buffer_name,
                held,
                len(teachers),
                wanted,
            )
            owner.summary_idxs = torch.tensor(wanted, dtype=buffer.dtype, device=device)
            rebuilt.append(f"{name}.{buffer_name}" if name else buffer_name)

    return rebuilt


def _submodels(model: Any) -> list[tuple[str, Any]]:
    """The model and every submodel of it that carries a config of its own.

    Args:
        model: The loaded model, of any family.

    Returns:
        Qualified name and module for each, the model itself under the
        empty name.
    """
    from transformers import PreTrainedModel

    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, PreTrainedModel)
    ]


def _teachers(module: Any) -> list[dict[str, Any]] | None:
    """The teacher list a C-RADIO config names, when this module has one.

    Args:
        module: A module of the loaded model.

    Returns:
        The teachers, or None for every module that is not a RADIO wrapper.
    """
    args = getattr(getattr(module, "config", None), "args", None)

    if not isinstance(args, dict):
        return None

    teachers = args.get("teachers")

    if not isinstance(teachers, list) or not all(isinstance(one, dict) for one in teachers):
        return None

    return teachers


def _legacy_kwargs(func: Any) -> Any:
    """Wrap a mask factory so calls in the older spelling still reach it.

    Args:
        func: The mask factory as this transformers defines it.

    Returns:
        A wrapper that renames what moved and drops what went away.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Forward the call under the names this transformers knows.

        Args:
            args: Positional arguments, which did not move.
            kwargs: Keyword arguments, in either spelling.

        Returns:
            Whatever the wrapped factory returns.
        """
        for old, new in _RENAMED.items():

            if old in kwargs and new not in kwargs:
                kwargs[new] = kwargs.pop(old)

        for name in _DROPPED:
            kwargs.pop(name, None)

        return func(*args, **kwargs)

    setattr(wrapper, _PATCHED, True)

    return wrapper


def _rebind(name: str, original: Any, shim: Any) -> None:
    """Put the shim wherever the original function is already bound.

    Module dicts are read directly rather than through getattr: a lazy
    module's __getattr__ would import half of transformers to answer.

    Args:
        name: The name the function is bound under.
        original: The function object to replace.
        shim: The wrapper to put in its place.
    """
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)

        if namespace is not None and namespace.get(name) is original:
            namespace[name] = shim


def _is_legacy_cache(value: Any, module: str) -> bool:
    """Is this a cache class of the module's own that predates the query offset?

    Args:
        value: A module attribute.
        module: Name of the module it was found in.

    Returns:
        True when the class is defined there and misses the method.
    """
    return (
        inspect.isclass(value)
        and getattr(value, "__module__", None) == module
        and hasattr(value, "get_mask_sizes")
        and hasattr(value, "get_seq_length")
        and not hasattr(value, "get_query_offset")
    )


def _get_query_offset(self: Any, layer_idx: int = 0) -> int:
    """The tokens already cached, which is where this query starts.

    Args:
        self: The cache this is bound onto.
        layer_idx: The layer whose cached length is asked for.

    Returns:
        The number of tokens the cache holds for that layer.
    """
    return int(self.get_seq_length(layer_idx))
