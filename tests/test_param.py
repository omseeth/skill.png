"""The saved PNG must be the thing that was optimized.

We optimize floats and ship uint8. The straight-through quantization surrogate is what keeps those
two the same image; these tests check the round trip and the surrogate agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from skillpng.config import ParamCfg
from skillpng.image.param import FullImage, build_param, load_param
from skillpng.image.preprocess import quantize_ste


def test_png_roundtrip_full_image(tmp_path: Path) -> None:
    torch.manual_seed(0)
    param = FullImage(torch.rand(3, 24, 32))
    original = param.tensor().detach().clone()

    path = param.save(tmp_path / "image")
    reloaded = load_param(path)

    assert isinstance(reloaded, FullImage)
    # Forward values are already on the uint8 grid, so the round trip is exact.
    assert torch.equal(reloaded.tensor().detach(), original)
    assert reloaded.digest() == param.digest()


def test_ste_forward_is_the_uint8_grid_and_backward_is_identity() -> None:
    raw = torch.rand(3, 8, 8, requires_grad=True)
    quantized = quantize_ste(raw)

    expected = torch.round(raw.detach() * 255) / 255
    assert torch.equal(quantized.detach(), expected)

    quantized.sum().backward()
    assert raw.grad is not None
    assert torch.equal(raw.grad, torch.ones_like(raw))


def test_saved_bytes_match_the_forward_tensor(tmp_path: Path) -> None:
    param = FullImage(torch.rand(3, 16, 16))
    path = param.save(tmp_path / "image")
    from skillpng.image.param import read_png

    assert torch.equal(read_png(path), param.tensor().detach())


def test_sidecar_records_provenance(tmp_path: Path) -> None:
    param = FullImage(torch.rand(3, 8, 8), init="gray")
    path = param.save(tmp_path / "image", meta={"step": 7})
    sidecar = json.loads(path.with_suffix(".json").read_text())

    assert sidecar["kind"] == "full"
    assert sidecar["digest"] == param.digest()
    assert sidecar["shape"] == [3, 8, 8]
    assert sidecar["step"] == 7


def test_project_is_a_plain_clamp_into_the_unit_cube() -> None:
    """The only constraint on the optimization: pixels stay in `[0,1]`, nothing else."""
    param = FullImage(torch.rand(3, 12, 12))
    with torch.no_grad():
        param.raw.copy_(torch.linspace(-2.0, 3.0, param.raw.numel()).reshape(param.raw.shape))
    param.project_()

    assert float(param.raw.detach().min()) == 0.0
    assert float(param.raw.detach().max()) == 1.0
    # Values already inside the cube are untouched — no ball to project onto.
    inside = FullImage(torch.rand(3, 8, 8))
    before = inside.raw.detach().clone()
    inside.project_()
    assert torch.equal(inside.raw.detach(), before)


def test_leaves_are_the_only_optimized_tensors() -> None:
    param = FullImage(torch.rand(3, 8, 8))
    assert [id(leaf) for leaf in param.leaves()] == [id(param.raw)]
    assert param.raw.requires_grad


def test_build_param_from_config() -> None:
    param = build_param(ParamCfg(kind="full", height=9, width=7, init="gray"))
    assert param.shape == (3, 9, 7)
    assert torch.allclose(param.tensor(), torch.full((3, 9, 7), 0.5), atol=1 / 255)
