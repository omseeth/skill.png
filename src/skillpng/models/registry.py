"""Model registry: string key -> VLM subclass, keeping the loop generic."""

from __future__ import annotations

from .._registry import Registry
from ..config import ModelCfg
from .base import VLM

MODELS: Registry = Registry("model")


def build_model(cfg: ModelCfg) -> VLM:
    """Instantiate the model named by the config.

    Args:
        cfg: Model config whose name selects the registered wrapper.

    Returns:
        The instantiated VLM wrapper.
    """
    # Imported here so registering the wrappers does not make transformers
    # an import-time dependency of the whole package.
    from . import hf, vllm  # noqa: F401  (registration side effects)

    return MODELS.get(cfg.name)(cfg)  # type: ignore[call-arg, no-any-return]
