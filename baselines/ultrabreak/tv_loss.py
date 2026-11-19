"""UltraBreak's total variation penalty on the image.

Added to the semantic loss with a fixed weight, it penalizes high-frequency
detail, so the image stays smooth. It is taken on the bare image, the one
that gets saved, not on a patch view.
"""

from __future__ import annotations

from torch import Tensor

# Weight of the penalty in the reference's recipe.
TV_WEIGHT = 0.5


def total_variation(image: Tensor) -> Tensor:
    """Anisotropic total variation: mean |dx| plus mean |dy|.

    Args:
        image: A [0,1] image tensor whose last two dims are height and width.

    Returns:
        A scalar tensor, zero for a flat image.
    """
    dx = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
    dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()

    return dx + dy
