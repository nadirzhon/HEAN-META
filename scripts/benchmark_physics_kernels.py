#!/usr/bin/env python3
"""Benchmark Python vs active backend physics kernels."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.physics.physics_kernels import (
    backend_name,
    calc_entropy,
    calc_entropy_py,
    calc_temperature,
    calc_temperature_py,
    detect_phase,
    detect_phase_py,
)


Batch = tuple[list[float], list[float]]


def make_batches(batch_count: int, lookback: int, seed: int) -> list[Batch]:
    random.seed(seed)
    batches: list[Batch] = []
    for _ in range(batch_count):
        prices = [10000.0]
        volumes = [1.0]
        for _ in range(lookback - 1):
            prices.append(prices[-1] + random.uniform(-25.0, 25.0))
            volumes.append(max(0.001, random.uniform(0.001, 8.0)))
        batches.append((prices, volumes))
    return batches


def run(
    batches: Sequence[Batch],
    iterations: int,
    temp_fn: Callable[[Sequence[float], Sequence[float]], float],
    entropy_fn: Callable[[Sequence[float]], float],
    phase_fn: Callable[[float, float], str],
) -> tuple[float, float]:
    checksum = 0.0
    start = time.perf_counter()
    for _ in range(iterations):
        for prices, volumes in batches:
            t = temp_fn(prices, volumes)
            e = entropy_fn(volumes)
            phase = phase_fn(t, e)
            checksum += (t * 1e-9) + (e * 1e-6) + (1 if phase == "VAPOR" else 0)
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    batches = make_batches(args.batches, args.lookback, args.seed)
    total_calls = args.batches * args.iterations

    py_elapsed, py_checksum = run(
        batches,
        args.iterations,
        calc_temperature_py,
        calc_entropy_py,
        detect_phase_py,
    )
    active_elapsed, active_checksum = run(
        batches,
        args.iterations,
        calc_temperature,
        calc_entropy,
        detect_phase,
    )

    py_ops = total_calls / py_elapsed
    active_ops = total_calls / active_elapsed
    speedup = py_elapsed / active_elapsed if active_elapsed > 0 else 0.0

    print(f"Backend active: {backend_name()}")
    print(f"Workload calls: {total_calls}")
    print(f"Python kernels:  {py_elapsed:.4f}s ({py_ops:.1f} calls/s), checksum={py_checksum:.6f}")
    print(
        f"Active kernels:  {active_elapsed:.4f}s ({active_ops:.1f} calls/s), "
        f"checksum={active_checksum:.6f}"
    )
    print(f"Speedup (active vs python): {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
