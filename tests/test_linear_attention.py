"""Qwen3.5's gated delta rule layers run FLA's kernel, or the load says so.

transformers picks the kernel once, when a family's modeling file is
imported, by walking attributes down from the fla package. The unit tests
drive that real lookup against a fake fla and a fake modeling file; the
integration tests need a GPU and fla-core itself.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import Tensor, nn
from transformers.integrations.hub_kernels import use_kernel_func_from_hub_with_fallback

from skillpng.models.linear_attention import (
    _kernel_behind,
    load_fla_kernels,
    verify_fla_fast_path,
)

CUDA = torch.device("cuda")
CPU = torch.device("cpu")
MODELING = "fake_modeling_qwen"


def fla_kernel(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) -> str:
    return "fla"


def torch_rule(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) -> str:
    return "torch"


def install_fla(monkeypatch: pytest.MonkeyPatch, ops_imported: bool) -> None:
    """Put a fake fla on the path, with or without its ops imported.

    fla-core's __init__ imports none of its ops, so a bare import leaves
    fla.ops unset, and that is the trap.
    """
    package = types.ModuleType("fla")
    ops = types.ModuleType("fla.ops")
    rule = types.ModuleType("fla.ops.gated_delta_rule")
    rule.chunk_gated_delta_rule = fla_kernel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fla", package)

    if ops_imported:
        package.ops = ops  # type: ignore[attr-defined]
        ops.gated_delta_rule = rule  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fla.ops", ops)
        monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", rule)
    else:
        monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", None)


def import_modeling(monkeypatch: pytest.MonkeyPatch) -> nn.Module:
    """Import a fake modeling file the way transformers writes one.

    Returns a model holding one of its gated delta rule layers.
    """
    module = types.ModuleType(MODELING)
    decorate = use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
    module.torch_chunk_gated_delta_rule = decorate(torch_rule)  # type: ignore[attr-defined]
    layer = type("ToyGatedDeltaNet", (nn.Module,), {"__module__": MODELING})
    monkeypatch.setitem(sys.modules, MODELING, module)

    return nn.Sequential(nn.Linear(2, 2), layer())


def test_the_lookup_finds_fla_once_its_ops_are_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fla(monkeypatch, ops_imported=True)
    model = import_modeling(monkeypatch)
    assert verify_fla_fast_path(model, CUDA) == [MODELING]


def test_a_bare_fla_import_leaves_the_fallback_and_cuda_refuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fla(monkeypatch, ops_imported=False)
    model = import_modeling(monkeypatch)
    rule = sys.modules[MODELING].torch_chunk_gated_delta_rule
    assert _kernel_behind(rule) is torch_rule
    with pytest.raises(RuntimeError, match="does not import here"):
        verify_fla_fast_path(model, CUDA)


def test_ops_imported_after_the_modeling_file_are_told_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fla(monkeypatch, ops_imported=False)
    model = import_modeling(monkeypatch)
    install_fla(monkeypatch, ops_imported=True)
    with pytest.raises(RuntimeError, match="only after the modeling file"):
        verify_fla_fast_path(model, CUDA)


def test_a_cpu_run_is_let_through_on_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fla(monkeypatch, ops_imported=False)
    model = import_modeling(monkeypatch)
    assert verify_fla_fast_path(model, CPU) == []


def test_a_model_without_gated_delta_rule_layers_passes() -> None:
    assert verify_fla_fast_path(nn.Linear(2, 2), CUDA) == []


def test_loading_reports_whether_the_ops_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fla(monkeypatch, ops_imported=False)
    assert load_fla_kernels() is False
    install_fla(monkeypatch, ops_imported=True)
    assert load_fla_kernels() is True


def gated_delta_inputs(device: torch.device, length: int = 300) -> list[Tensor]:
    """q, k, v, g and beta in Qwen3.5's layout and dtypes, gradients on.

    300 tokens is not a multiple of the 64-token chunk, so the padding the
    torch implementation does is exercised too.
    """
    generator = torch.Generator().manual_seed(0)
    heads, dim = 4, 128

    def draw(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator)

    q = draw(1, length, heads, dim).to(device, torch.bfloat16)
    k = draw(1, length, heads, dim).to(device, torch.bfloat16)
    v = draw(1, length, heads, dim).to(device, torch.bfloat16)
    # Log decays, negative, and computed in fp32 as the layer does.
    g = -nn.functional.softplus(draw(1, length, heads)).to(device)
    beta = draw(1, length, heads).sigmoid().to(device, torch.bfloat16)

    return [x.requires_grad_(True) for x in (q, k, v, g, beta)]


def relative_error(got: Tensor, want: Tensor) -> float:
    return float((got.float() - want.float()).norm() / want.float().norm())


@pytest.mark.integration
def test_qwen3_5_finds_fla_when_it_is_imported_first() -> None:
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    name = "transformers.models.qwen3_5.modeling_qwen3_5"

    if name in sys.modules:
        pytest.skip("an earlier test imported the modeling file before FLA")

    assert load_fla_kernels()
    from transformers.models.qwen3_5 import modeling_qwen3_5

    rule = modeling_qwen3_5.torch_chunk_gated_delta_rule
    assert _kernel_behind(rule) is fla.chunk_gated_delta_rule


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fla_computes_the_torch_implementation_and_its_gradient() -> None:
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    from transformers.models.qwen3_5 import modeling_qwen3_5

    reference = modeling_qwen3_5.torch_chunk_gated_delta_rule.__wrapped__
    outputs, grads = [], []
    upstream = torch.randn(1, 300, 4, 128, generator=torch.Generator().manual_seed(1))

    for rule in (fla.chunk_gated_delta_rule, reference):
        inputs = gated_delta_inputs(CUDA)
        out, _ = rule(*inputs, use_qk_l2norm_in_kernel=True)
        (out.float() * upstream.to(CUDA)).sum().backward()
        outputs.append(out)
        grads.append([x.grad for x in inputs])

    assert relative_error(outputs[0], outputs[1]) < 0.02

    # Looser: bf16 gradients, and g's sums a whole chunk of them.
    for got, want in zip(grads[0], grads[1], strict=True):
        assert got is not None and want is not None
        assert relative_error(got, want) < 0.05


@pytest.mark.integration
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_fla_runs_on_the_second_gpu_of_a_split() -> None:
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    first, second = torch.device("cuda", 0), torch.device("cuda", 1)
    results = []

    # The current device stays the first, as in a split model, so the
    # kernel has to follow its tensors onto the second itself.
    with torch.cuda.device(first):
        for device in (first, second):
            out, _ = fla.chunk_gated_delta_rule(
                *gated_delta_inputs(device), use_qk_l2norm_in_kernel=True
            )
            results.append(out.to(first))

    torch.testing.assert_close(results[1], results[0])
