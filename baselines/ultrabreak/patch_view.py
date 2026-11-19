"""UltraBreak's random patch view: the image pasted as a patch on a canvas.

Each item's loss sees the image shrunk to a fraction of the canvas side,
rotated and zoomed by the angle and factor drawn for it, and moved by a
random whole-pixel offset over a plain background. This follows
apply_random_patch of github.com/kaiyuanCui/UltraBreak, which pastes a 224
px patch on a 336 px white image. The canvas has the image's own size, so
the saved image keeps the size the model is fed at test.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor

WHITE = 1.0
# Side of the reference's patch relative to its canvas, 224 / 336.
PATCH_FRACTION = 2.0 / 3.0
# The reference's rotation range in degrees and its zoom range.
ROTATION_RANGE = (-15.0, 15.0)
SCALE_RANGE = (0.8, 1.2)


def sample_view(
    rng: random.Random,
    rotation_range: tuple[float, float],
    scale_range: tuple[float, float],
) -> tuple[float, float]:
    """Draw one rotation angle and one zoom factor, each uniformly.

    Args:
        rng: Generator the draws are taken from.
        rotation_range: (low, high) angle in degrees.
        scale_range: (low, high) zoom factor.

    Returns:
        The angle in degrees and the zoom factor.
    """
    degrees = rng.uniform(*rotation_range)
    scale = rng.uniform(*scale_range)

    return degrees, scale


def rotate_scale(image: Tensor, degrees: float, scale: float, fill: float = WHITE) -> Tensor:
    """Rotate and zoom the image about its center, differentiably.

    One bilinear resample covers both, so the view is blurred once, not
    twice. The output keeps the input's size; pixels the transform uncovers
    take the fill value.

    Args:
        image: A [0,1] (3,H,W) image tensor.
        degrees: Rotation angle in degrees.
        scale: Zoom factor; above 1 zooms in, below 1 zooms out.
        fill: Value of the uncovered pixels, white by default.

    Returns:
        The transformed (3,H,W) image, the input itself for the identity.
    """
    if degrees == 0.0 and scale == 1.0:
        return image

    height, width = image.shape[-2:]
    cos = math.cos(math.radians(degrees)) / scale
    sin = math.sin(math.radians(degrees)) / scale
    # affine_grid maps output to input in coordinates normalized per axis, so
    # the off-diagonal terms carry the aspect ratio to keep the rotation rigid
    # on non-square images.
    theta = torch.tensor(
        [[[cos, sin * height / width, 0.0], [-sin * width / height, cos, 0.0]]],
        dtype=image.dtype,
        device=image.device,
    )
    # Zero padding around a shifted image puts the fill value outside.
    shifted = (image - fill).unsqueeze(0)
    grid = F.affine_grid(theta, list(shifted.shape), align_corners=False)
    sampled = F.grid_sample(
        shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    return sampled.squeeze(0) + fill


def box_extent(height: int, width: int, degrees: float, zoom: float) -> tuple[float, float]:
    """Measure the box a rotated, zoomed patch covers.

    Args:
        height: Patch height in pixels before the zoom.
        width: Patch width in pixels before the zoom.
        degrees: Rotation angle in degrees.
        zoom: Total zoom factor, the canvas fraction included.

    Returns:
        The height and width of the bounding box in pixels.
    """
    cos = abs(math.cos(math.radians(degrees)))
    sin = abs(math.sin(math.radians(degrees)))

    return zoom * (height * cos + width * sin), zoom * (width * cos + height * sin)


def max_offset(height: int, width: int, degrees: float, zoom: float) -> tuple[int, int]:
    """Find the largest whole-pixel offset that keeps the patch on the canvas.

    Args:
        height: Canvas and patch height in pixels.
        width: Canvas and patch width in pixels.
        degrees: Rotation angle in degrees.
        zoom: Total zoom factor, the canvas fraction included.

    Returns:
        The largest vertical and horizontal offset from the canvas center.
    """
    box_height, box_width = box_extent(height, width, degrees, zoom)

    if box_height > height or box_width > width:
        raise ValueError(
            f"a patch zoomed by {zoom:.3f} and rotated by {degrees:.1f} degrees "
            f"covers {box_height:.0f}x{box_width:.0f} px, more than the "
            f"{height}x{width} canvas; lower the patch fraction or the zoom"
        )

    return math.floor((height - box_height) / 2), math.floor((width - box_width) / 2)


def sample_offset(
    rng: random.Random, height: int, width: int, degrees: float, zoom: float
) -> tuple[int, int]:
    """Draw where the patch center goes, uniformly over the spots that fit.

    Args:
        rng: Generator the draw is taken from.
        height: Canvas and patch height in pixels.
        width: Canvas and patch width in pixels.
        degrees: Rotation angle in degrees.
        zoom: Total zoom factor, the canvas fraction included.

    Returns:
        The vertical and horizontal offset from the canvas center in pixels.
    """
    room_y, room_x = max_offset(height, width, degrees, zoom)

    return rng.randint(-room_y, room_y), rng.randint(-room_x, room_x)


def shift(image: Tensor, dy: int, dx: int, fill: float) -> Tensor:
    """Move the image content by whole pixels, filling what it uncovers.

    Args:
        image: A (3,H,W) image tensor.
        dy: Downward offset in pixels.
        dx: Rightward offset in pixels.
        fill: Value of the uncovered pixels.

    Returns:
        The shifted image, the size of the input.
    """
    height, width = image.shape[-2:]
    padded = F.pad(image, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)), value=fill)
    top, left = max(-dy, 0), max(-dx, 0)

    return padded[..., top : top + height, left : left + width]


def place_patch(
    image: Tensor,
    degrees: float,
    scale: float,
    offset: tuple[int, int],
    fraction: float = PATCH_FRACTION,
    background: float = WHITE,
) -> Tensor:
    """Paste the image as a rotated, zoomed patch on a plain canvas.

    One bilinear resample does rotation and zoom; the offset is whole pixels,
    so it blurs nothing.

    Args:
        image: The [0,1] (3,H,W) patch; the canvas has the same size.
        degrees: Rotation angle in degrees.
        scale: Zoom factor drawn for this view.
        offset: Vertical and horizontal offset of the patch center in pixels.
        fraction: Side of the unzoomed patch relative to the canvas side.
        background: Value of the canvas around the patch.

    Returns:
        The (3,H,W) canvas the model sees.
    """
    centered = rotate_scale(image, degrees, fraction * scale, fill=background)

    return shift(centered, *offset, fill=background)
