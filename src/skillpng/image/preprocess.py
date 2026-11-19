"""The single, differentiable, torch-only preprocessing path."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor

# PIL resample enum -> torch interpolate mode. Lanczos/Hamming have no torch
# equivalent; bicubic is the closest, and the residual difference is what
# test_preprocess_parity's tolerance covers.
_RESAMPLE_TO_TORCH = {
    0: "nearest",
    1: "bicubic",
    2: "bilinear",
    3: "bicubic",
    4: "area",
    5: "bilinear",
}


@dataclass(frozen=True)
class PreprocessSpec:
    """Everything needed to turn a [0,1] (3,H,W) tensor into the tower's pixel_values.

    Derived from the processor at load time; never written down by us.
    """
    image_mean: tuple[float, ...] = (0.0, 0.0, 0.0)
    image_std: tuple[float, ...] = (1.0, 1.0, 1.0)
    rescale_factor: float = 1 / 255
    do_rescale: bool = True
    do_normalize: bool = True
    height: int | None = None
    width: int | None = None
    size_divisor: int | None = None
    patch_size: int | None = None
    merge_size: int | None = None
    max_soft_tokens: int | None = None
    temporal_patch_size: int | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    resample: str = "bicubic"
    raw: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def from_processor(processor: object) -> PreprocessSpec:
        """Read every preprocessing constant off a loaded processor's config.

        Args:
            processor: A processor or image processor exposing the HF config attributes.

        Returns:
            A spec populated from the processor, with defaults where fields are absent.
        """
        ip = getattr(processor, "image_processor", processor)
        size = _as_dict(getattr(ip, "size", None))
        crop = _as_dict(getattr(ip, "crop_size", None))
        height, width = _resolve_size(
            size, crop, do_crop=bool(getattr(ip, "do_center_crop", False))
        )
        resample = getattr(ip, "resample", 3)
        raw = {
            "class": type(ip).__name__,
            "size": size,
            "crop_size": crop,
            **{
                key: getattr(ip, key)
                for key in (
                    "image_mean",
                    "image_std",
                    "norm_mean",
                    "norm_std",
                    "rescale_factor",
                    "do_rescale",
                    "do_normalize",
                    "do_resize",
                    "do_center_crop",
                    "do_convert_rgb",
                    "resample",
                    "patch_size",
                    "merge_size",
                    "pooling_kernel_size",
                    "max_soft_tokens",
                    "temporal_patch_size",
                    "size_divisor",
                    "min_pixels",
                    "max_pixels",
                )
                if hasattr(ip, key)
            },
        }

        # Nemotron 3 Nano Omni keeps its constants under norm_mean and
        # norm_std, leaves the standard fields at its base class's None, and
        # rescales and normalizes in its own _preprocess.
        mean = getattr(ip, "image_mean", None) or getattr(ip, "norm_mean", None)
        std = getattr(ip, "image_std", None) or getattr(ip, "norm_std", None)

        return PreprocessSpec(
            image_mean=tuple(mean or (0.0, 0.0, 0.0)),
            image_std=tuple(std or (1.0, 1.0, 1.0)),
            rescale_factor=float(getattr(ip, "rescale_factor", 1 / 255) or 1 / 255),
            do_rescale=_flag(getattr(ip, "do_rescale", None)),
            do_normalize=_flag(getattr(ip, "do_normalize", None)),
            height=height,
            width=width,
            size_divisor=_first_int(getattr(ip, "size_divisor", None)),
            patch_size=_first_int(
                getattr(ip, "patch_size", None) or getattr(processor, "patch_size", None)
            ),
            merge_size=_first_int(
                getattr(ip, "merge_size", None)
                # Gemma 4's name for the same thing: patches per merged token.
                or getattr(ip, "pooling_kernel_size", None)
                or getattr(processor, "merge_size", None)
            ),
            max_soft_tokens=_first_int(getattr(ip, "max_soft_tokens", None)),
            temporal_patch_size=_first_int(getattr(ip, "temporal_patch_size", None)),
            min_pixels=_first_int(size.get("min_pixels") or getattr(ip, "min_pixels", None)),
            max_pixels=_first_int(size.get("max_pixels") or getattr(ip, "max_pixels", None)),
            resample=_RESAMPLE_TO_TORCH.get(
                int(resample) if isinstance(resample, int) else 3, "bicubic"
            ),
            raw=raw,
        )

    @property
    def max_patches(self) -> int | None:
        """Rows this processor pads every image out to: its token budget, unmerged.

        Returns:
            The padded patch-sequence length, or None when no budget is exposed.
        """
        if not self.max_soft_tokens:
            return None

        merge = self.merge_size or 1
        return self.max_soft_tokens * merge * merge

    @property
    def alignment(self) -> int:
        """Resolution must be a multiple of this: patch * merge, else size_divisor.

        Returns:
            The pixel multiple every resolution has to round down to, at least 1.
        """
        if self.patch_size:
            return self.patch_size * (self.merge_size or 1)
        
        return self.size_divisor or 1

    def resolve_resolution(self, requested: tuple[int, int] | None = None) -> tuple[int, int]:
        """Round a resolution to the alignment and clamp it into the pixel budget.

        Args:
            requested: Height and width to use instead of the spec's own fixed size.

        Returns:
            An aligned height and width satisfying min_pixels and max_pixels.
        """
        height, width = requested or (self.height or 0, self.width or 0)

        if height <= 0 or width <= 0:
            if not (self.max_patches and self.patch_size):
                raise ValueError(
                    "no resolution available: the processor config did not expose a fixed size, "
                    "so set model.resolution: [height, width] in your config."
                )
            # No fixed size, but a token budget. The fit below grows any square
            # into the largest one that fills it.
            height = width = self.alignment
        
        align = max(self.alignment, 1)
        height, width = _round_to(height, align), _round_to(width, align)

        if self.max_patches and self.patch_size:
            # A budget-packing family resizes every image to the largest
            # aligned grid that fills its budget, so any other resolution is
            # resampled before the tower sees it. Land on the size it keeps.
            height, width = _fit_patch_budget(
                height, width, self.patch_size, self.max_patches, align
            )

        if self.max_pixels and height * width > self.max_pixels:
            scale = sqrt(self.max_pixels / (height * width))
            height, width = (
                _round_to(int(height * scale), align),
                _round_to(int(width * scale), align),
            )

        if self.min_pixels and height * width < self.min_pixels:
            scale = sqrt(self.min_pixels / (height * width))
            height = _round_to(int(height * scale) + align, align)
            width = _round_to(int(width * scale) + align, align)

        return height, width

    def token_count(self, height: int, width: int) -> int | None:
        """Count the image tokens a resolution turns into, per patch and merge size.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            The token count, or None when the spec exposes no patch size.
        """
        if not self.patch_size:
            return None
        
        merge = self.merge_size or 1
        return (height // self.patch_size) * (width // self.patch_size) // (merge * merge)

    def describe(self) -> dict[str, object]:
        """Flatten the spec into a JSON-friendly record for logging and artifacts.

        Returns:
            A dict of every resolved field plus the raw processor fields it came from.
        """
        return {
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "rescale_factor": self.rescale_factor,
            "do_rescale": self.do_rescale,
            "do_normalize": self.do_normalize,
            "size": [self.height, self.width],
            "alignment": self.alignment,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "max_soft_tokens": self.max_soft_tokens,
            "max_patches": self.max_patches,
            "temporal_patch_size": self.temporal_patch_size,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "resample": self.resample,
            "processor_fields": self.raw,
        }


def quantize_ste(image: Tensor) -> Tensor:
    """Straight-through uint8 quantization.

    Forward: exactly the 8-bit grid a PNG can store, so the saved artifact *is*
    the thing that was optimized. Backward: identity. Without this the optimizer
    chases sub-quantization detail that does not survive save().

    Args:
        image: A [0,1] image tensor of any shape.

    Returns:
        The image snapped to the 8-bit grid, with an identity gradient.
    """
    quantized = torch.round(image.clamp(0.0, 1.0) * 255.0) / 255.0
    return image + (quantized - image).detach()


def resize_image(image: Tensor, spec: PreprocessSpec, size: tuple[int, int]) -> Tensor:
    """[0,1] (3,H,W) -> [0,1] (1,3,h,w) at the fed resolution. Differentiable.

    The resize on its own, because the two backends need it and nothing else
    in common: models.hf goes on to normalize the result into pixel_values,
    models.vllm encodes it as the PNG the server preprocesses itself.

    Args:
        image: A [0,1] image tensor of shape (3,H,W), or (1,3,H,W).
        spec: The preprocessing constants read off the model's processor.
        size: Target height and width to resize to.

    Returns:
        A batched [0,1] tensor of shape (1,3,h,w) carrying gradients.
    """
    if image.dim() == 4:

        if image.shape[0] != 1:
            raise ValueError(f"expected a single image, got batch of {image.shape[0]}")
        
        image = image[0]

    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"expected (3,H,W) in [0,1], got {tuple(image.shape)}")

    x = image.unsqueeze(0)
    height, width = size

    if x.shape[-2:] != (height, width):
        antialias = spec.resample in {"bilinear", "bicubic"}
        x = F.interpolate(x, size=(height, width), mode=spec.resample, antialias=antialias)
        x = x.clamp(0.0, 1.0)  # bicubic overshoots outside [0,1]

    return x


def preprocess_image(image: Tensor, spec: PreprocessSpec, size: tuple[int, int]) -> Tensor:
    """[0,1] (3,H,W) -> pixel_values (1,3,h,w). Differentiable end to end.

    Mirrors the HF image-processor pipeline: resize, rescale from the 0-255 range, then normalize.

    Args:
        image: A [0,1] image tensor of shape (3,H,W), or (1,3,H,W).
        spec: The preprocessing constants read off the model's processor.
        size: Target height and width to resize to.

    Returns:
        A batched pixel_values tensor of shape (1,3,h,w) carrying gradients.
    """
    # Tensor is [0,1]; the processor's pipeline starts from [0,255].
    x = resize_image(image, spec, size) * 255.0
    if spec.do_rescale:
        x = x * spec.rescale_factor

    if spec.do_normalize:
        mean = torch.tensor(spec.image_mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(spec.image_std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        x = (x - mean) / std

    return x


def pack_patch_sequence(pixel_values: Tensor, spec: PreprocessSpec) -> Tensor:
    """(1,3,h,w) -> (num_patches, temporal*3*patch*patch). Differentiable.

    Dynamic-resolution towers do not take a picture, they take a flat sequence of patches, one row per patch. Rebuilding that layout in torch — rather 
    letting the processor build it in NumPy — is what keeps the gradient 
    attached to the canvas. Two invariants decide whether the tower sees the 
    same image the processor would have handed it, and both fail silently when broken:

        1. Patches come out in (gh//merge, gw//merge, merge_h, merge_w) order, 
        so every consecutive merge**2 rows form one spatial block that the patch
        merger folds into a single LLM token. A plain row-major flatten over the
        gh x gw grid groups horizontal strips instead and destroys locality.
        2. Inside a row the channel axis is outer to the temporal one, i.e.
        (C, T, ph, pw). Swapping the two feeds the first vision conv interleaved
        frames where it expects channels.

    A still image is repeated temporal_patch_size times, which is exactly what 
    the processor does for a single frame.

    Args:
        pixel_values: Normalized pixels of shape (1,3,h,w).
        spec: The preprocessing constants read off the model's processor.

    Returns:
        A 2-D patch sequence carrying gradients back to pixel_values.
    """
    if pixel_values.dim() != 4 or pixel_values.shape[0] != 1:
        raise ValueError(f"expected (1,3,h,w), got {tuple(pixel_values.shape)}")

    patch = spec.patch_size
    
    if not patch:
        raise ValueError(
            "this processor returns a patch sequence but exposes no patch_size, so the "
            "packing layout cannot be derived from its config; override "
            "HFVLM._pack_pixel_values for this family."
        )

    merge = spec.merge_size or 1
    temporal = spec.temporal_patch_size or 1
    channels, height, width = pixel_values.shape[1:]
    align = patch * merge

    if height % align or width % align:
        raise ValueError(
            f"{height}x{width} is not a multiple of patch_size*merge_size "
            f"({align}); set model.resolution to a multiple of {align}."
        )

    gh, gw = height // patch, width // patch
    # (T, C, gH//m, m_h, ph, gW//m, m_w, pw); the frame axis is a stride-0
    # expand, so repeating the image costs nothing until the reshape below.
    patches = pixel_values.expand(temporal, -1, -1, -1).reshape(
        temporal, channels, gh // merge, merge, patch, gw // merge, merge, patch
    )
    # Token order outer to inner: gH//m, gW//m, m_h, m_w. Per-row content
    # order: C, T, ph, pw.
    patches = patches.permute(2, 5, 3, 6, 1, 0, 4, 7)

    return patches.reshape(gh * gw, temporal * channels * patch * patch)


def pack_padded_patch_sequence(
    pixel_values: Tensor, spec: PreprocessSpec, merged: bool = False
) -> Tensor:
    """(1,3,h,w) -> (1, rows, pixels-per-row), zero-padded. Differentiable.

    The other kind of patch-sequence tower: instead of a row per patch and
    nothing else, this one pads every image out to a fixed budget of rows and
    tells the encoder which of them are real through a separate position-id
    key. Two invariants, both silent when broken:

        1. Rows are plain row-major over the patch grid — no merge blocking
        within the row order. Pooling into soft tokens is driven by the
        position ids, and those ids are built row-major alongside.
        2. Inside a row the pixel axes are outer to the channel one, i.e.
        (ph, pw, C): one pixel's three channels sit next to each other. This is
        the opposite of pack_patch_sequence's (C, T, ph, pw).

    Some processors of that family fold the merge x merge block of patches
    into the soft token it becomes before handing it over: a row is then one
    block of pixels, and the budget counts soft tokens rather than patches.
    That is what merged selects, and it is the same packing with a merge times
    larger patch — the block is contiguous either way, so nothing else moves.

    The padding rows are zeros, matching the processor. They carry position id
    -1, and the encoder masks them out before pooling, so nothing downstream
    reads them and no gradient comes back through them.

    Args:
        pixel_values: Normalized pixels of shape (1,3,h,w).
        spec: The preprocessing constants read off the model's processor.
        merged: Whether a row holds a whole merge x merge block of patches.

    Returns:
        A padded, batched patch sequence carrying gradients back to pixel_values.
    """
    if pixel_values.dim() != 4 or pixel_values.shape[0] != 1:
        raise ValueError(f"expected (1,3,h,w), got {tuple(pixel_values.shape)}")

    patch, max_patches = spec.patch_size, spec.max_patches

    if not patch or not max_patches:
        raise ValueError(
            "this processor pads its patch sequence to a token budget but " \
            "exposes no patch_size or max_soft_tokens, so the packing layout " \
            "cannot be derived from its config; override " \
            "HFVLM_pack_pixel_values for this family."
        )

    merge = spec.merge_size or 1
    channels, height, width = pixel_values.shape[1:]
    align = patch * merge

    if height % align or width % align:
        raise ValueError(
            f"{height}x{width} is not a multiple of patch_size*merge_size "
            f"({align}); set model.resolution to a multiple of {align}."
        )

    # One row per merged block and a budget in soft tokens, or one row per
    # patch and a budget in patches.
    block = align if merged else patch
    rows = (spec.max_soft_tokens or 0) if merged else max_patches
    gh, gw = height // block, width // block

    if gh * gw > rows:
        raise ValueError(
            f"{height}x{width} is {gh * gw} rows of pixels, over this "
            f"processor's budget of {rows}; lower model.resolution."
        )

    patches = pixel_values[0].reshape(channels, gh, block, gw, block)
    # Token order: gh, gw. Per-row content order: ph, pw, C.
    patches = patches.permute(1, 3, 2, 4, 0).reshape(gh * gw, channels * block * block)
    padding = rows - gh * gw

    if padding:
        patches = F.pad(patches, (0, 0, 0, padding))

    return patches.unsqueeze(0)


def _as_dict(value: object) -> dict[str, int]:
    """Keep a size-like config field only if it is already a dict, else drop it.

    Args:
        value: A processor config field that may or may not be a mapping.

    Returns:
        A shallow copy of the dict, or an empty dict.
    """
    return dict(value) if isinstance(value, dict) else {}


def _flag(value: object) -> bool:
    """Read a processor's do_ flag, taking an unset one as on.

    None is a flag the processor never set, not one it turned off. With no
    mean or std found, normalizing on is the identity anyway.

    Args:
        value: The flag as the processor holds it, possibly None.

    Returns:
        False only when the processor set the flag off.
    """
    return True if value is None else bool(value)


def _first_int(
    value: int | float | str | dict[str, int] | list[int] | tuple[int, ...] | None,
) -> int | None:
    """Pull a single int out of the several shapes processor configs use.

    Args:
        value: A number, a size dict, a sequence, or None.

    Returns:
        The first usable int, or None when the value holds none.
    """
    if value is None:
        return None
    
    if isinstance(value, dict):
        for key in ("height", "shortest_edge", "width"):
            if key in value:
                return int(value[key])
            
        return None
    
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_size(
    size: dict[str, int], crop: dict[str, int], do_crop: bool
) -> tuple[int | None, int | None]:
    """Pick the fixed resolution a processor implies, crop winning over size.

    Args:
        size: The processor's size dict.
        crop: The processor's crop_size dict.
        do_crop: Whether the processor center-crops, making crop the final size.

    Returns:
        A height and width, or None for both when no fixed size is exposed.
    """
    if do_crop and "height" in crop and "width" in crop:
        return int(crop["height"]), int(crop["width"])
    if "height" in size and "width" in size:
        return int(size["height"]), int(size["width"])
    if "shortest_edge" in size:
        edge = int(size["shortest_edge"])
        return edge, edge
    if "longest_edge" in size:
        edge = int(size["longest_edge"])
        return edge, edge
    
    return None, None


def _round_to(value: int, multiple: int) -> int:
    """Round a value down to a multiple, never below the multiple itself.

    Args:
        value: The value to round.
        multiple: The alignment to round down to.

    Returns:
        The largest multiple not exceeding value, at minimum one multiple.
    """
    return max(multiple, (value // multiple) * multiple)


def _fit_patch_budget(
    height: int, width: int, patch: int, max_patches: int, align: int
) -> tuple[int, int]:
    """Largest aligned size of this aspect ratio whose patch count fits the budget.

    Mirrors the resize a budget-packing processor does itself, iterated to its
    fixed point: flooring each side to the alignment leaves budget unspent, and
    another pass takes it. Neither side ever shrinks, so this settles at once
    for a square and in a round or two otherwise.

    Args:
        height: Requested height in pixels.
        width: Requested width in pixels.
        patch: Patch size in pixels.
        max_patches: Patch rows the processor pads every image out to.
        align: Pixel multiple both sides have to stay on.

    Returns:
        A height and width the processor would hand to the tower unresized.
    """
    target = max_patches * patch * patch

    for _ in range(8):
        scale = sqrt(target / (height * width))
        fitted = _round_to(int(height * scale), align), _round_to(int(width * scale), align)

        if fitted == (height, width):
            break

        height, width = fitted

    return height, width
