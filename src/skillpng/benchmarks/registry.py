"""BENCHMARKS — string key -> Benchmark subclass."""

from __future__ import annotations

from .._registry import Registry
from ..config import BenchmarkCfg
from .base import Benchmark

BENCHMARKS: Registry = Registry("benchmark")


def build_benchmark(cfg: BenchmarkCfg) -> Benchmark:
    """Look up the Benchmark subclass registered under cfg.name and instantiate it with cfg.

    Args:
        cfg: Benchmark configuration whose name selects the registered benchmark.

    Returns:
        Benchmark instance built from cfg.
    """
    if not cfg.name:
        raise ValueError("benchmark.name is empty: set it to a registered benchmark key")
    
    return BENCHMARKS.get(cfg.name)(cfg)  # type: ignore[call-arg, no-any-return]
