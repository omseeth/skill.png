"""skillpng — gradient-optimized guidance images for frozen local vision-language models.

Nothing here imports `transformers` at package import time: config, image parameterization, and the
benchmark contract are usable (and tested) without model weights.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
