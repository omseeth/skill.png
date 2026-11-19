"""Benchmark contract and registry. Import your own adapter modules before building one."""

from __future__ import annotations

from .base import Benchmark, Item, Result, inject_image, split_by_group
from .livemathematicianbench.livemathematicianbench import LiveMathematicianBench
from .registry import BENCHMARKS, build_benchmark
from .searchqa.searchqa import SearchQA
from .t1.t1 import T1
from .trajectbench.traject import TrajectBench

__all__ = [
    "BENCHMARKS",
    "T1",
    "Benchmark",
    "Item",
    "LiveMathematicianBench",
    "Result",
    "SearchQA",
    "TrajectBench",
    "build_benchmark",
    "inject_image",
    "split_by_group",
]
