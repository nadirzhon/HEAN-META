"""Pure compute kernels for physics microservice.

This module keeps hot-path calculations stateless and testable:
- temperature
- entropy
- phase detection

If a compiled C++ module (`physics_cpp`) is available, wrappers below
dispatch to it; otherwise they use the pure Python implementations.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from typing import Sequence


def calc_temperature_py(prices: Sequence[float], volumes: Sequence[float]) -> float:
    """Kinetic-energy-like market temperature from price delta and volume."""
    n = min(len(prices), len(volumes))
    if n < 2:
        return 0.0

    kinetic_energy = 0.0
    for i in range(1, n):
        dp = float(prices[i]) - float(prices[i - 1])
        vm = float(volumes[i])
        x = dp * vm
        kinetic_energy += x * x

    return kinetic_energy / float(n)


def calc_entropy_py(volumes: Sequence[float]) -> float:
    """Shannon entropy over positive volume distribution."""
    positives = [float(v) for v in volumes if v > 0]
    if not positives:
        return 0.0

    total = sum(positives)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for v in positives:
        p = v / total
        entropy -= p * math.log(p)
    return entropy


def detect_phase_py(temperature: float, entropy: float) -> str:
    """Simple phase classifier used by microservice outputs."""
    if temperature < 400 and entropy < 2.5:
        return "ICE"
    if temperature >= 800 and entropy >= 3.5:
        return "VAPOR"
    return "WATER"


def extract_price_volume(data: object) -> tuple[float, float] | None:
    """Extract (price, volume) from collector payload variants."""
    if isinstance(data, list) and data:
        row = data[0]
        if isinstance(row, dict):
            return float(row.get("p", 0) or 0), float(row.get("v", 0) or 0)
        return None

    if isinstance(data, dict):
        price = float(data.get("lastPrice", data.get("p", 0)) or 0)
        volume = float(data.get("volume24h", data.get("v", 1)) or 0)
        return price, volume

    return None


def _maybe_import_cpp():
    disabled = os.getenv("PHYSICS_CPP_DISABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if disabled:
        return None

    # Local repo module location when built via cpp_core install target.
    repo_root = Path(__file__).resolve().parents[2]
    cpp_module_dir = repo_root / "src" / "hean" / "cpp_modules"
    if cpp_module_dir.exists():
        module_path = str(cpp_module_dir)
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

    try:
        return importlib.import_module("physics_cpp")
    except Exception:
        return None


_CPP = _maybe_import_cpp()


def backend_name() -> str:
    return "cpp" if _CPP is not None else "python"


def calc_temperature(prices: Sequence[float], volumes: Sequence[float]) -> float:
    if _CPP is not None:
        return float(_CPP.calc_temperature(list(prices), list(volumes)))
    return calc_temperature_py(prices, volumes)


def calc_entropy(volumes: Sequence[float]) -> float:
    if _CPP is not None:
        return float(_CPP.calc_entropy(list(volumes)))
    return calc_entropy_py(volumes)


def detect_phase(temperature: float, entropy: float) -> str:
    if _CPP is not None:
        return str(_CPP.detect_phase(float(temperature), float(entropy)))
    return detect_phase_py(temperature, entropy)
