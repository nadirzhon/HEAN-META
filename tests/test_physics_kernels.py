from __future__ import annotations

import random

from services.physics import physics_kernels as kernels


def _sample_data(seed: int = 42, n: int = 120) -> tuple[list[float], list[float]]:
    random.seed(seed)
    prices = [10000.0]
    volumes = [1.0]
    for _ in range(n - 1):
        prices.append(prices[-1] + random.uniform(-35.0, 35.0))
        volumes.append(max(0.001, random.uniform(0.001, 10.0)))
    return prices, volumes


def test_python_kernels_basic_invariants() -> None:
    prices, volumes = _sample_data()

    temp = kernels.calc_temperature_py(prices, volumes)
    entropy = kernels.calc_entropy_py(volumes)
    phase = kernels.detect_phase_py(temp, entropy)

    assert temp >= 0.0
    assert entropy >= 0.0
    assert phase in {"ICE", "WATER", "VAPOR"}


def test_extract_price_volume_variants() -> None:
    pv = kernels.extract_price_volume([{"p": "123.5", "v": "0.9"}])
    assert pv == (123.5, 0.9)

    pv = kernels.extract_price_volume({"lastPrice": "456.1", "volume24h": "321.0"})
    assert pv == (456.1, 321.0)

    assert kernels.extract_price_volume("bad") is None


def test_active_backend_matches_python_semantics() -> None:
    prices, volumes = _sample_data()

    temp_py = kernels.calc_temperature_py(prices, volumes)
    ent_py = kernels.calc_entropy_py(volumes)
    phase_py = kernels.detect_phase_py(temp_py, ent_py)

    temp_active = kernels.calc_temperature(prices, volumes)
    ent_active = kernels.calc_entropy(volumes)
    phase_active = kernels.detect_phase(temp_active, ent_active)

    # Allow tiny floating-point drift between Python and C++ backends.
    assert abs(temp_active - temp_py) < 1e-9
    assert abs(ent_active - ent_py) < 1e-9
    assert phase_active == phase_py

