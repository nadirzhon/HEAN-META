"""Rust bridge for physics calculations.

Provides graceful fallback to pure Python when Rust module unavailable.
"""

import logging
import math

logger = logging.getLogger(__name__)

# Try to import Rust module
try:
    import hean_core
    RUST_AVAILABLE = True
    logger.info("Rust core module loaded (100x speedup enabled)")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("Rust core not available, using pure Python (slower)")


def market_temperature(prices: list[float], volumes: list[float]) -> float:
    """Calculate market temperature: T = KE / N where KE = Sum (dP * V)^2."""
    if RUST_AVAILABLE:
        return hean_core.market_temperature(prices, volumes)

    if len(prices) < 2:
        return 0.0
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    n = len(prices)
    kinetic_energy = 0.0
    for i in range(1, len(prices)):
        price_change = prices[i] - prices[i - 1]
        volume = volumes[i]
        kinetic_energy += (price_change * volume) ** 2

    return kinetic_energy / n


def market_entropy(volumes: list[float]) -> float:
    """Calculate market entropy: S = -Sum p_i * log(p_i)."""
    if RUST_AVAILABLE:
        return hean_core.market_entropy(volumes)

    if not volumes:
        return 0.0

    total_volume = sum(volumes)
    if total_volume == 0.0:
        return 0.0

    entropy = 0.0
    for vol in volumes:
        if vol > 0.0:
            p = vol / total_volume
            entropy -= p * math.log(p)

    return entropy


def detect_phase(
    temperature: float,
    entropy: float,
    temp_history: list[float],
    entropy_history: list[float],
) -> str:
    """Detect market phase based on temperature and entropy."""
    if RUST_AVAILABLE:
        return hean_core.detect_phase(temperature, entropy, temp_history, entropy_history)

    temp_threshold = 1.0
    if temp_history:
        avg_temp = sum(temp_history) / len(temp_history)
        temp_threshold = avg_temp * 1.5

    entropy_threshold = 2.0
    if entropy_history:
        avg_entropy = sum(entropy_history) / len(entropy_history)
        entropy_threshold = avg_entropy * 1.2

    if temperature < temp_threshold * 0.5 and entropy < entropy_threshold * 0.8:
        return "ICE"
    elif temperature > temp_threshold and entropy > entropy_threshold:
        return "VAPOR"
    elif temperature > temp_threshold * 0.7:
        return "WATER"
    else:
        return "TRANSITION"


def szilard_profit(temperature: float, information_bits_val: float) -> float:
    """Calculate Szilard profit: Work = k * T * ln(2) * information_bits."""
    if RUST_AVAILABLE:
        return hean_core.szilard_profit(temperature, information_bits_val)

    K_CRYPTO = 0.1
    LN_2 = math.log(2)
    return K_CRYPTO * temperature * LN_2 * information_bits_val


def information_bits(prediction_accuracy: float) -> float:
    """Calculate information content from prediction accuracy."""
    if RUST_AVAILABLE:
        return hean_core.information_bits(prediction_accuracy)

    if prediction_accuracy <= 0.0 or prediction_accuracy >= 1.0:
        return 0.0
    return -math.log2(1.0 - prediction_accuracy)


def thermal_efficiency(profit: float, risk_capital: float) -> float:
    """Calculate thermal efficiency: eta = W / Q_in."""
    if RUST_AVAILABLE:
        return hean_core.thermal_efficiency(profit, risk_capital)

    if risk_capital <= 0.0:
        return 0.0
    return profit / risk_capital
