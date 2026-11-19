"""Model wrappers. `hf` is imported lazily by `build_model` so `transformers` stays off the
package's import path until a model is actually loaded."""

from __future__ import annotations

from .base import VLM, masked_labels, target_span, token_ce
from .registry import MODELS, build_model

__all__ = ["MODELS", "VLM", "build_model", "masked_labels", "target_span", "token_ce"]
