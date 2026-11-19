"""ImageParam: the only leaf in the graph that carries gradient.

tensor() is quantization-straight-through (see quantize_ste), so what the
model sees in the forward pass is exactly the 8-bit image save() writes.
save() always emits a PNG plus a JSON sidecar recording how the image came to
exist.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from ..config import ParamCfg
from .preprocess import quantize_ste

# Mid-gray, the value a fresh canvas starts from.
GRAY = 0.5


class ImageParam(ABC):
    """A differentiable image in [0,1], shape (3,H,W)."""

    kind: str

    @abstractmethod
    def tensor(self) -> Tensor:
        """Produce the image the model sees in the forward pass.

        Returns:
            A differentiable [0,1] (3,H,W) tensor, quantized
            straight-through to the uint8 grid.
        """

    @abstractmethod
    def leaves(self) -> list[Tensor]:
        """Expose the tensors an optimizer steps.

        Returns:
            Exactly the leaf tensors carrying requires_grad=True.
        """

    @abstractmethod
    def project_(self) -> None:
        """Enforce this parameterization's constraints in place."""

    @abstractmethod
    def provenance(self) -> dict[str, object]:
        """Describe how this image came to exist.

        Returns:
            Sidecar fields specific to this parameterization.
        """

    @property
    def shape(self) -> tuple[int, int, int]:
        """Report the shape of the image this parameter produces.

        Returns:
            The (3,H,W) shape as a tuple of ints.
        """
        return tuple(self.tensor().shape)  # type: ignore[return-value]

    def to_uint8(self) -> np.ndarray:
        """Convert the current image to the 8-bit array that gets written.

        Returns:
            An (H,W,3) uint8 array with channels last, clamped to [0,1]
            before scaling.
        """
        array = (self.tensor().detach().float().clamp(0, 1) * 255).round().to(torch.uint8)
        return array.permute(1, 2, 0).cpu().numpy()

    def digest(self) -> str:
        """Hash the uint8 image so every results row can join on content.

        Returns:
            The first 16 hex characters of the image's SHA-256 digest.
        """
        return hashlib.sha256(self.to_uint8().tobytes()).hexdigest()[:16]

    def save(self, path: str | Path, meta: dict[str, object] | None = None) -> Path:
        """Write the image as a PNG beside a JSON provenance sidecar.

        Args:
            path: Destination path; the suffix is forced to .png.
            meta: Extra sidecar fields, merged last so they win on conflict.

        Returns:
            The path of the PNG that was written.
        """
        path = Path(path).with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.to_uint8(), mode="RGB").save(path)
        sidecar = {
            "kind": self.kind,
            "digest": self.digest(),
            "shape": list(self.shape),
            "created": datetime.now(UTC).isoformat(),
            **self.provenance(),
            **(meta or {}),
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> ImageParam:
        """Reconstruct a parameter from a saved PNG and its sidecar.

        Args:
            path: Path to the saved PNG.
            device: Torch device the reconstructed tensors are placed on.

        Returns:
            An ImageParam holding the loaded image.
        """


class FullImage(ImageParam):
    """Unconstrained canvas — the only parameterization.

    The optimization is constrained by nothing but the pixel cube: project_() is a plain clamp
    into [0,1] after every step. No perturbation budget, no projection onto an L-inf ball.
    """

    kind: str = "full"

    def __init__(self, raw: Tensor, init: str = "unknown") -> None:
        """Wrap a (3,H,W) tensor as a trainable canvas.

        Args:
            raw: The [0,1] (3,H,W) image copied into a trainable leaf.
            init: Label recording where the initial pixels came from.
        """
        _check_chw(raw)
        self.raw = raw.detach().clone().float().requires_grad_(True)
        self.init = init

    @classmethod
    def new(
        cls,
        height: int,
        width: int,
        init: str = "gray",
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> FullImage:
        """Create a fresh canvas from gray, noise, or an image on disk.

        Args:
            height: Canvas height in pixels.
            width: Canvas width in pixels.
            init: "gray", "random", or a path to a PNG resized into place.
            seed: Seed used by the "random" initialization.
            device: Torch device the canvas is placed on.

        Returns:
            A FullImage initialized as requested.
        """
        if init == "gray":
            raw = torch.full((3, height, width), GRAY)
        elif init == "random":
            generator = torch.Generator().manual_seed(seed)
            raw = torch.rand((3, height, width), generator=generator)
        else:
            raw = _resize_to(read_png(init), height, width)

        return cls(raw.to(device), init=init)

    def tensor(self) -> Tensor:
        """Quantize the raw canvas straight-through to the uint8 grid.

        Returns:
            A differentiable [0,1] (3,H,W) tensor.
        """
        return quantize_ste(self.raw)

    def leaves(self) -> list[Tensor]:
        """Expose the single trainable canvas tensor.

        Returns:
            A one-element list holding the raw canvas.
        """
        return [self.raw]

    def project_(self) -> None:
        """Clamp the canvas back into the pixel cube after a step."""
        with torch.no_grad():
            self.raw.clamp_(0.0, 1.0)

    def provenance(self) -> dict[str, object]:
        """Record how this canvas was initialized.

        Returns:
            A dict carrying the init label.
        """
        return {"init": self.init}

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> FullImage:
        """Rebuild a canvas from the pixels of a saved PNG.

        Args:
            path: Path to the saved PNG.
            device: Torch device the canvas is placed on.

        Returns:
            A FullImage whose init label is the source path.
        """
        return cls(read_png(path).to(device), init=str(path))


PARAM_KINDS: dict[str, type[ImageParam]] = {"full": FullImage}


def load_param(path: str | Path, device: str | torch.device = "cpu") -> ImageParam:
    """Load whichever parameterization the sidecar says this PNG is.

    Args:
        path: Path to the PNG; the sidecar is read from the .json beside it.
        device: Torch device the loaded tensors are placed on.

    Returns:
        The ImageParam subclass named by the sidecar's kind field.
    """
    path = Path(path).with_suffix(".png")
    kind = str(read_sidecar(path).get("kind", "full"))

    if kind not in PARAM_KINDS:
        raise ValueError(f"unknown param kind {kind!r} in {path.with_suffix('.json')}")
    
    return PARAM_KINDS[kind].load(path, device=device)


def build_param(cfg: ParamCfg, seed: int = 0, device: str | torch.device = "cpu") -> ImageParam:
    """Build a fresh parameter from config.

    Args:
        cfg: Parameter config naming the kind, size, and initialization.
        seed: Seed forwarded to random initialization.
        device: Torch device the new parameter is placed on.

    Returns:
        A freshly initialized ImageParam.
    """
    if cfg.kind == "full":
        return FullImage.new(cfg.height, cfg.width, init=cfg.init, seed=seed, device=device)
    raise ValueError(f"unknown param.kind {cfg.kind!r}")


def noise_param(cfg: ParamCfg, seed: int = 0, device: str | torch.device = "cpu") -> ImageParam:
    """Build the noise condition: a random canvas on the [0,1] cube.

    Args:
        cfg: Parameter config supplying the canvas height and width.
        seed: Seed for the uniform noise, also recorded in the init label.
        device: Torch device the canvas is placed on.

    Returns:
        A FullImage filled with uniform noise.
    """
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand((3, cfg.height, cfg.width), generator=generator)
    return FullImage(raw.to(device), init=f"noise:seed={seed}")


def read_png(path: str | Path) -> Tensor:
    """Read a PNG off disk as a float image tensor.

    Args:
        path: Path to the PNG, converted to RGB on read.

    Returns:
        A [0,1] (3,H,W) float tensor.
    """
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB")).copy()

    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def _resize_to(image: Tensor, height: int, width: int) -> Tensor:
    """Resize an image to the target size, skipping the no-op case.

    Args:
        image: A [0,1] (3,H,W) float tensor.
        height: Target height in pixels.
        width: Target width in pixels.

    Returns:
        The image at (3,height,width), bicubic-resampled and clamped.
    """
    if image.shape[-2:] == (height, width):
        return image
    
    resized = F.interpolate(
        image.unsqueeze(0), size=(height, width), mode="bicubic", antialias=True
    )

    return resized.squeeze(0).clamp(0, 1)


def _check_chw(image: Tensor) -> None:
    """Reject any tensor that is not a 3-channel CHW image.

    Args:
        image: The tensor to validate.
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"expected (3,H,W), got {tuple(image.shape)}")


def read_sidecar(path: str | Path) -> dict[str, object]:
    """Read the JSON sidecar sitting beside a saved PNG.

    Args:
        path: Path to the PNG; the .json beside it is read.

    Returns:
        The parsed sidecar, or an empty dict when there is none.
    """
    sidecar = Path(path).with_suffix(".json")

    if not sidecar.exists():
        return {}
    
    data: dict[str, object] = json.loads(sidecar.read_text())

    return data


