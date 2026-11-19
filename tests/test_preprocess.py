"""Preprocessing: spec derivation without weights, and parity with the real processor with them.

`test_preprocess_parity` is the test that catches "works in training, wrong at eval" — it is the
only place our torch pipeline and the model's own `AutoProcessor` are compared. It needs real
weights, so it is an integration test:

    SKILLPNG_TEST_MODEL=/path/to/local/vlm uv run pytest -m integration
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from skillpng.image.preprocess import (
    PreprocessSpec,
    pack_padded_patch_sequence,
    pack_patch_sequence,
    preprocess_image,
)
from skillpng.models.hf import HFVLM


def _fake_processor(**overrides: object) -> SimpleNamespace:
    defaults = {
        "size": {"height": 336, "width": 336},
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 1 / 255,
        "do_rescale": True,
        "do_normalize": True,
        "resample": 3,
        "patch_size": 14,
    }
    defaults.update(overrides)
    return SimpleNamespace(image_processor=SimpleNamespace(**defaults))


def test_spec_is_read_off_the_processor() -> None:
    spec = PreprocessSpec.from_processor(_fake_processor())
    assert (spec.height, spec.width) == (336, 336)
    assert spec.image_mean == (0.5, 0.5, 0.5)
    assert spec.resample == "bicubic"
    assert spec.patch_size == 14
    assert spec.raw["class"] == "SimpleNamespace"


def test_shortest_edge_and_crop_size_forms() -> None:
    spec = PreprocessSpec.from_processor(_fake_processor(size={"shortest_edge": 224}))
    assert (spec.height, spec.width) == (224, 224)

    cropped = PreprocessSpec.from_processor(
        _fake_processor(
            size={"shortest_edge": 256},
            crop_size={"height": 224, "width": 224},
            do_center_crop=True,
        )
    )
    assert (cropped.height, cropped.width) == (224, 224)


def test_resolution_rounds_to_the_patch_grid() -> None:
    spec = PreprocessSpec.from_processor(_fake_processor(patch_size=14, merge_size=2))
    assert spec.alignment == 28
    assert spec.resolve_resolution((450, 450)) == (448, 448)
    assert spec.token_count(448, 448) == (448 // 14) ** 2 // 4


def test_resolution_respects_a_pixel_budget() -> None:
    spec = PreprocessSpec.from_processor(_fake_processor(patch_size=14, max_pixels=200_000))
    height, width = spec.resolve_resolution((1024, 1024))
    assert height * width <= 200_000
    assert height % 14 == 0 and width % 14 == 0


def test_missing_resolution_is_a_loud_error() -> None:
    spec = PreprocessSpec.from_processor(_fake_processor(size={}))
    with pytest.raises(ValueError, match=r"model\.resolution"):
        spec.resolve_resolution()


def test_preprocess_normalizes_and_keeps_the_gradient() -> None:
    spec = PreprocessSpec.from_processor(_fake_processor(size={"height": 8, "width": 8}))
    image = torch.rand(3, 8, 8, requires_grad=True)
    pixel_values = preprocess_image(image, spec, (8, 8))

    assert pixel_values.shape == (1, 3, 8, 8)
    expected = (image.detach() * 255 * (1 / 255) - 0.5) / 0.5
    assert torch.allclose(pixel_values.detach(), expected, atol=1e-6)

    pixel_values.sum().backward()
    assert image.grad is not None and float(image.grad.abs().sum()) > 0


_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
_NORM_STD = (0.26862954, 0.26130258, 0.27577711)


def _norm_mean_processor() -> SimpleNamespace:
    """A processor shaped like Nemotron 3 Nano Omni's.

    It keeps its constants under norm_mean and norm_std and leaves the
    standard fields at the None its base class sets.

    Returns:
        A processor whose standard preprocessing fields are all unset.
    """
    return _fake_processor(
        size={},
        image_mean=None,
        image_std=None,
        do_rescale=None,
        do_normalize=None,
        norm_mean=list(_NORM_MEAN),
        norm_std=list(_NORM_STD),
        patch_size=16,
    )


def _norm_mean_reference(array: torch.Tensor) -> torch.Tensor:
    """What that processor's own _preprocess makes of uint8 pixels.

    Args:
        array: Pixels of shape (3,H,W), 0-255.

    Returns:
        Its pixel_values, of shape (1,3,H,W).
    """
    mean = torch.tensor(_NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_NORM_STD).view(1, 3, 1, 1)
    return (array.float().unsqueeze(0) / 255.0 - mean) / std


def test_spec_reads_norm_mean_when_the_standard_fields_are_unset() -> None:
    """None flags are unset, not off; norm_mean and norm_std stand in."""
    spec = PreprocessSpec.from_processor(_norm_mean_processor())
    assert spec.image_mean == _NORM_MEAN
    assert spec.image_std == _NORM_STD
    assert spec.do_rescale and spec.do_normalize
    assert spec.raw["norm_mean"] == list(_NORM_MEAN)


def test_flags_set_off_stay_off() -> None:
    """A processor that turns normalization off explicitly is obeyed."""
    spec = PreprocessSpec.from_processor(_fake_processor(do_normalize=False))
    assert not spec.do_normalize


def test_preprocess_matches_a_norm_mean_processor() -> None:
    """The regression: these pixels went in as raw 0-255 values."""
    spec = PreprocessSpec.from_processor(_norm_mean_processor())
    array = torch.randint(0, 256, (3, 16, 16), generator=torch.Generator().manual_seed(0))
    ours = preprocess_image(array.float() / 255.0, spec, (16, 16))
    assert torch.allclose(ours, _norm_mean_reference(array), atol=1e-5)


def _probe_checking_vlm(spec: PreprocessSpec) -> HFVLM:
    """A model carrying only what the pixel parity check reads: no weights.

    Args:
        spec: The spec it builds its pixels from.

    Returns:
        An HFVLM whose _check_pixel_parity can be called.
    """
    vlm = HFVLM.__new__(HFVLM)
    vlm._spec = spec
    vlm._size = (16, 16)
    vlm._ref_pv_shape = (1, 3, 16, 16)
    return vlm


def _probe_reference() -> torch.Tensor:
    """The norm_mean processor's pixel_values for the model's probe image.

    Returns:
        Its pixel_values for the probe, of shape (1,3,16,16).
    """
    probe = _probe_checking_vlm(PreprocessSpec())._probe_image()
    array = torch.from_numpy(np.array(probe, dtype=np.uint8)).permute(2, 0, 1)
    return _norm_mean_reference(array)


def test_pixel_parity_passes_when_the_constants_are_read() -> None:
    """The fixed spec rebuilds the processor's probe pixels."""
    spec = PreprocessSpec.from_processor(_norm_mean_processor())
    _probe_checking_vlm(spec)._check_pixel_parity(_probe_reference())


def test_pixel_parity_catches_a_misread_constant() -> None:
    """The spec the Nemotron images were trained with now fails at load."""
    unnormalized = PreprocessSpec(do_rescale=False, do_normalize=False)
    with pytest.raises(RuntimeError, match="pixel_values for the probe"):
        _probe_checking_vlm(unnormalized)._check_pixel_parity(_probe_reference())

    # One channel's constant wrong is caught too.
    shifted = PreprocessSpec(
        image_mean=(_NORM_MEAN[0], _NORM_MEAN[1], 0.5), image_std=_NORM_STD
    )
    with pytest.raises(RuntimeError, match="pixel_values for the probe"):
        _probe_checking_vlm(shifted)._check_pixel_parity(_probe_reference())


def test_preprocess_resizes_when_the_canvas_differs() -> None:
    spec = PreprocessSpec.from_processor(_fake_processor())
    image = torch.rand(3, 64, 64, requires_grad=True)
    pixel_values = preprocess_image(image, spec, (32, 48))
    assert pixel_values.shape == (1, 3, 32, 48)
    pixel_values.sum().backward()
    assert image.grad is not None


def test_preprocess_rejects_wrong_shapes() -> None:
    spec = PreprocessSpec.from_processor(_fake_processor())
    with pytest.raises(ValueError):
        preprocess_image(torch.rand(1, 8, 8), spec, (8, 8))
    with pytest.raises(ValueError):
        preprocess_image(torch.rand(2, 3, 8, 8), spec, (8, 8))


def _patch_sequence_spec() -> PreprocessSpec:
    """A processor of the dynamic-resolution kind: 2x2 pixel patches, merged 2x2.

    Returns:
        A spec whose pixel_values are a flat patch sequence.
    """
    return PreprocessSpec.from_processor(
        _fake_processor(patch_size=2, merge_size=2, temporal_patch_size=2)
    )


def test_spec_reads_the_temporal_patch_size() -> None:
    assert _patch_sequence_spec().temporal_patch_size == 2
    assert PreprocessSpec.from_processor(_fake_processor()).temporal_patch_size is None


def test_pack_patch_sequence_keeps_the_merge_block_layout() -> None:
    """Rows must come out in merge blocks, each laid out (C, T, ph, pw)."""
    spec = _patch_sequence_spec()
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    packed = pack_patch_sequence(image, spec)

    # 4x4 patch grid, each row 2 frames * 3 channels * 2 * 2 pixels.
    assert packed.shape == (16, 24)

    for index in range(16):
        block_row, block_col = index // 8, (index // 4) % 2
        row, col = (index // 2) % 2, index % 2
        grid_row, grid_col = block_row * 2 + row, block_col * 2 + col
        expected = image[0, :, grid_row * 2 : grid_row * 2 + 2, grid_col * 2 : grid_col * 2 + 2]
        patch = packed[index].reshape(3, 2, 2, 2)  # (C, T, ph, pw)
        assert torch.equal(patch[:, 0], expected)
        assert torch.equal(patch[:, 1], expected)  # the frame is repeated


def test_pack_patch_sequence_keeps_the_gradient() -> None:
    spec = _patch_sequence_spec()
    image = torch.rand(3, 8, 8, requires_grad=True)
    packed = pack_patch_sequence(preprocess_image(image, spec, (8, 8)), spec)

    assert packed.shape == (16, 24)
    packed.sum().backward()
    assert image.grad is not None and float(image.grad.abs().min()) > 0


def test_pack_patch_sequence_rejects_an_unaligned_resolution() -> None:
    with pytest.raises(ValueError, match="patch_size\\*merge_size"):
        pack_patch_sequence(torch.rand(1, 3, 6, 8), _patch_sequence_spec())


def _soft_token_spec(**overrides: object) -> PreprocessSpec:
    """A processor of the budget-packing kind: 2x2 patches, pooled 2x2, 5 tokens.

    Args:
        **overrides: Processor fields to change for this spec.

    Returns:
        A spec whose pixel_values are a patch sequence padded to a token budget.
    """
    fields: dict[str, object] = {
        "patch_size": 2,
        "pooling_kernel_size": 2,
        "max_soft_tokens": 5,
        "do_normalize": False,
        "size": {},
    }
    fields.update(overrides)
    return PreprocessSpec.from_processor(_fake_processor(**fields))


def test_spec_reads_the_token_budget() -> None:
    spec = _soft_token_spec()
    assert spec.max_soft_tokens == 5
    # pooling_kernel_size is this family's name for merge_size.
    assert spec.merge_size == 2
    assert spec.max_patches == 20
    assert PreprocessSpec.from_processor(_fake_processor()).max_patches is None


def test_resolution_lands_on_the_size_the_budget_implies() -> None:
    """Under budget is resized up, over it down: only the fixed point is fed."""
    spec = _soft_token_spec()
    assert spec.alignment == 4
    assert spec.resolve_resolution((4, 4)) == (8, 8)
    assert spec.resolve_resolution((64, 64)) == (8, 8)
    assert spec.token_count(8, 8) == 4

    fitted = spec.resolve_resolution((32, 16))
    assert spec.resolve_resolution(fitted) == fitted


def test_pack_padded_patch_sequence_keeps_the_row_major_layout() -> None:
    """Rows are row-major over the grid, each laid out (ph, pw, C), then padded."""
    spec = _soft_token_spec()
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    packed = pack_padded_patch_sequence(image, spec)

    # 4x4 patch grid padded to the 20-row budget, each row 3 channels * 2 * 2.
    assert packed.shape == (1, 20, 12)

    for index in range(16):
        grid_row, grid_col = index // 4, index % 4
        expected = image[0, :, grid_row * 2 : grid_row * 2 + 2, grid_col * 2 : grid_col * 2 + 2]
        patch = packed[0, index].reshape(2, 2, 3)  # (ph, pw, C)
        assert torch.equal(patch.permute(2, 0, 1), expected)

    assert torch.equal(packed[0, 16:], torch.zeros(4, 12))


def test_pack_padded_patch_sequence_merges_rows_into_soft_tokens() -> None:
    """Merged, a row is the whole merge x merge block and the budget counts tokens."""
    spec = _soft_token_spec()
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    packed = pack_padded_patch_sequence(image, spec, merged=True)

    # 2x2 grid of 4x4 blocks padded to the 5-token budget, each row 3 * 4 * 4.
    assert packed.shape == (1, 5, 48)

    for index in range(4):
        grid_row, grid_col = index // 2, index % 2
        expected = image[0, :, grid_row * 4 : grid_row * 4 + 4, grid_col * 4 : grid_col * 4 + 4]
        block = packed[0, index].reshape(4, 4, 3)  # (ph, pw, C)
        assert torch.equal(block.permute(2, 0, 1), expected)

    assert torch.equal(packed[0, 4:], torch.zeros(1, 48))


def test_pack_padded_patch_sequence_keeps_the_gradient() -> None:
    spec = _soft_token_spec()
    image = torch.rand(3, 8, 8, requires_grad=True)
    packed = pack_padded_patch_sequence(preprocess_image(image, spec, (8, 8)), spec)

    packed.sum().backward()
    assert image.grad is not None and float(image.grad.abs().min()) > 0


def test_pack_padded_patch_sequence_rejects_an_over_budget_resolution() -> None:
    with pytest.raises(ValueError, match="budget"):
        pack_padded_patch_sequence(torch.rand(1, 3, 16, 16), _soft_token_spec())


def test_pack_padded_patch_sequence_rejects_an_unaligned_resolution() -> None:
    with pytest.raises(ValueError, match="patch_size\\*merge_size"):
        pack_padded_patch_sequence(torch.rand(1, 3, 6, 8), _soft_token_spec())


def _budget_packing_vlm(ref_shape: tuple[int, ...]) -> HFVLM:
    """A model carrying only the two fields the packing choice reads: no weights.

    Args:
        ref_shape: The pixel_values shape the processor produced for the probe.

    Returns:
        An HFVLM whose _pack_pixel_values can be called.
    """
    vlm = HFVLM.__new__(HFVLM)
    vlm._spec = _soft_token_spec()
    vlm._ref_pv_shape = ref_shape
    return vlm


def test_the_processor_row_count_picks_the_layout() -> None:
    """A row per patch or a row per soft token: only the probe's shape says which."""
    image = torch.rand(1, 3, 8, 8)

    assert _budget_packing_vlm((1, 20, 12))._pack_pixel_values(image).shape == (1, 20, 12)
    assert _budget_packing_vlm((1, 5, 48))._pack_pixel_values(image).shape == (1, 5, 48)


def test_padded_patch_parity_with_a_real_budget_packing_processor() -> None:
    """Parity against the processor class itself, which needs no weights to build."""
    import numpy as np

    module = pytest.importorskip(
        "transformers.models.gemma4.image_processing_pil_gemma4",
        reason="no budget-packing processor in this transformers",
    )
    image_processor = module.Gemma4ImageProcessorPil()
    spec = PreprocessSpec.from_processor(image_processor)
    height, width = spec.resolve_resolution((768, 768))

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    theirs = image_processor(images=[Image.fromarray(array, mode="RGB")], return_tensors="pt")
    ours = pack_padded_patch_sequence(preprocess_image(tensor, spec, (height, width)), spec)

    assert tuple(ours.shape) == tuple(theirs["pixel_values"].shape)
    assert torch.allclose(ours, theirs["pixel_values"].float(), atol=1e-6)
    # The rows the processor calls real are the ones we filled.
    valid = int((theirs["image_position_ids"] != -1).all(dim=-1).sum())
    assert valid == (height // spec.patch_size) * (width // spec.patch_size)


def test_merged_patch_parity_with_a_real_soft_token_processor() -> None:
    """The other processor of that family folds the block itself; same parity check."""
    import numpy as np

    module = pytest.importorskip(
        "transformers.models.gemma4_unified.image_processing_gemma4_unified",
        reason="no soft-token-packing processor in this transformers",
    )
    image_processor = module.Gemma4UnifiedImageProcessor()
    spec = PreprocessSpec.from_processor(image_processor)
    height, width = spec.resolve_resolution((768, 768))

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    theirs = image_processor(images=[Image.fromarray(array, mode="RGB")], return_tensors="pt")
    ours = pack_padded_patch_sequence(
        preprocess_image(tensor, spec, (height, width)), spec, merged=True
    )

    assert tuple(ours.shape) == tuple(theirs["pixel_values"].shape)
    assert torch.allclose(ours, theirs["pixel_values"].float(), atol=1e-6)
    # The rows the processor calls real are the ones we filled.
    valid = int((theirs["image_position_ids"] != -1).all(dim=-1).sum())
    assert valid == spec.token_count(height, width)


@pytest.mark.integration
def test_preprocess_parity() -> None:
    """Our torch preprocessing must match the model's own AutoProcessor."""
    model_id = os.environ.get("SKILLPNG_TEST_MODEL")
    if not model_id:
        pytest.skip("set SKILLPNG_TEST_MODEL to a local VLM checkpoint")

    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    spec = PreprocessSpec.from_processor(processor)
    height, width = spec.resolve_resolution()

    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    pil = Image.fromarray(array, mode="RGB")
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    theirs = processor.image_processor(images=[pil], return_tensors="pt")["pixel_values"]
    ours = preprocess_image(tensor, spec, (height, width))
    if theirs.dim() == 2:
        # A dynamic-resolution tower: compare the packed patch sequence, since
        # the packing is exactly where a wrong-but-plausible layout hides.
        ours = pack_patch_sequence(ours, spec)
    elif theirs.dim() == 3:
        # The same, for a tower that pads the sequence to a token budget —
        # merged when its rows are soft tokens rather than patches.
        ours = pack_padded_patch_sequence(
            ours, spec, merged=theirs.shape[1] == spec.max_soft_tokens != spec.max_patches
        )

    assert ours.shape == theirs.shape, (
        f"shape mismatch {tuple(ours.shape)} vs {tuple(theirs.shape)}: this family packs pixels "
        "differently — override HFVLM._pack_pixel_values."
    )
    # No resize is involved here, so only rescale/normalize can differ: the tolerance is tight.
    assert torch.allclose(ours, theirs.float(), atol=1e-4), float(
        (ours - theirs.float()).abs().max()
    )
