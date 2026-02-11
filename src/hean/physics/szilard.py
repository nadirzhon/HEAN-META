"""Szilard's Formula for Maximum Extractable Profit.

Based on Szilard's engine and information theory:
    MAX_PROFIT = T * I
where:
    T = market temperature (energy available)
    I = information in bits = log2(1/p)
    p = probability of event
"""

import math
from dataclasses import dataclass

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SzilardProfit:
    max_profit: float
    temperature: float
    information_bits: float
    probability: float
    confidence: float


class SzilardEngine:
    """Calculate maximum extractable profit using Szilard's formula."""

    PRACTICAL_SCALE = 0.001
    MIN_PROBABILITY = 0.01
    MAX_PROBABILITY = 0.99

    def calculate_max_profit(
        self,
        temperature: float,
        probability: float,
        capital: float = 1000.0,
    ) -> SzilardProfit:
        probability = max(self.MIN_PROBABILITY, min(self.MAX_PROBABILITY, probability))
        information_bits = math.log2(1.0 / probability)
        theoretical_profit = temperature * information_bits
        max_profit = theoretical_profit * self.PRACTICAL_SCALE * capital / 1000.0

        temp_confidence = 1.0 - abs(temperature - 600) / 600
        prob_confidence = 1.0 - 2 * abs(probability - 0.5)
        confidence = max(0.0, min(1.0, (temp_confidence + prob_confidence) / 2))

        result = SzilardProfit(
            max_profit=max_profit,
            temperature=temperature,
            information_bits=information_bits,
            probability=probability,
            confidence=confidence,
        )

        logger.debug(
            f"[Szilard] MAX_PROFIT=${max_profit:.2f} "
            f"(T={temperature:.1f}, I={information_bits:.2f} bits, "
            f"p={probability:.2%}, confidence={confidence:.2f})"
        )

        return result

    def calculate_edge_from_physics(
        self,
        temperature: float,
        entropy: float,
        signal_confidence: float = 0.5,
    ) -> float:
        entropy_factor = max(0, 1.0 - (entropy / 5.0))
        temp_factor = 1.0 - abs(temperature - 600) / 600
        temp_factor = max(0, min(1, temp_factor))
        physics_edge = (entropy_factor * 0.6 + temp_factor * 0.4) * 100
        return physics_edge * signal_confidence

    def should_trade(
        self,
        temperature: float,
        entropy: float,
        phase: str,
        min_edge_bps: float = 5.0,
        ssd_mode: str = "normal",
        resonance_strength: float = 0.0,
    ) -> tuple[bool, str]:
        # SSD: Laplace mode overrides — deterministic trading
        if ssd_mode == "laplace":
            edge = self.calculate_edge_from_physics(
                temperature, entropy, 0.5 + resonance_strength * 0.45
            )
            return True, (
                f"SSD LAPLACE: deterministic (resonance={resonance_strength:.3f}, "
                f"edge={edge:.1f} bps)"
            )

        # SSD: Silent mode — absolute block
        if ssd_mode == "silent":
            return False, "SSD SILENT: entropy diverging, noise regime"

        if phase == "vapor":
            return False, "VAPOR phase - too chaotic"

        if phase == "water" and entropy < 2.5:
            edge = self.calculate_edge_from_physics(temperature, entropy, 0.8)
            if edge >= min_edge_bps:
                return True, f"ICE->WATER transition (edge={edge:.1f} bps)"

        if phase == "ice" and entropy < 2.0:
            return False, "ICE phase - waiting for breakout"

        if phase == "water":
            edge = self.calculate_edge_from_physics(temperature, entropy, 0.6)
            if edge >= min_edge_bps:
                return True, f"WATER phase trending (edge={edge:.1f} bps)"

        return False, "Conditions not favorable"

    def calculate_optimal_size_multiplier(
        self, temperature: float, entropy: float, phase: str,
        ssd_mode: str = "normal", resonance_strength: float = 0.0,
    ) -> float:
        # SSD: Silent → zero size
        if ssd_mode == "silent":
            return 0.0

        base = 0.5
        if phase == "vapor":
            base = 0.1
        elif phase == "ice":
            base = 0.5 if entropy < 2.0 else 0.3
        elif phase == "water":
            entropy_mult = max(0.5, 1.0 - (entropy - 2.0) / 3.0)
            temp_mult = max(0.5, min(1.5, 1.0 - abs(temperature - 600) / 600))
            base = min(2.0, entropy_mult * temp_mult)

        # SSD: Laplace mode → boost proportional to resonance
        if ssd_mode == "laplace":
            base = min(2.0, base * (1.0 + resonance_strength))

        return base
