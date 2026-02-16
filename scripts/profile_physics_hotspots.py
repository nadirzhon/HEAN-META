#!/usr/bin/env python3
"""Profile physics hot path under synthetic load.

`py-spy` is preferred when available. In restricted/offline environments this
script falls back to cProfile and still reports top hotspots.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.physics.physics_kernels import (
    calc_entropy,
    calc_temperature,
    detect_phase,
)


def make_batches(batch_count: int, lookback: int, seed: int) -> list[tuple[list[float], list[float]]]:
    random.seed(seed)
    batches: list[tuple[list[float], list[float]]] = []
    for _ in range(batch_count):
        prices = [10000.0]
        volumes = [1.0]
        for _ in range(lookback - 1):
            prices.append(prices[-1] + random.uniform(-25.0, 25.0))
            volumes.append(max(0.001, random.uniform(0.001, 8.0)))
        batches.append((prices, volumes))
    return batches


def run_workload(
    batches: Sequence[tuple[list[float], list[float]]],
    iterations: int,
) -> float:
    checksum = 0.0
    for _ in range(iterations):
        for prices, volumes in batches:
            t = calc_temperature(prices, volumes)
            e = calc_entropy(volumes)
            phase = detect_phase(t, e)
            checksum += (t * 1e-9) + (e * 1e-6) + (1 if phase == "VAPOR" else 0)
    return checksum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    batches = make_batches(args.batches, args.lookback, args.seed)

    profiler = cProfile.Profile()
    profiler.enable()
    checksum = run_workload(batches, args.iterations)
    profiler.disable()

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(args.top)

    print(f"checksum={checksum:.6f}")
    print(stats_stream.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
