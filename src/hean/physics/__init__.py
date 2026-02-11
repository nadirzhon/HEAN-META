"""Physics-based market analysis module.

Provides thermodynamic analysis of market states:
- Temperature: Market kinetic energy
- Entropy: Volume distribution disorder
- Phase Detection: ICE/WATER/VAPOR transitions
- Szilard: Maximum extractable profit
- Participant Classifier: X-Ray into market players
- Temporal Stack: Multi-timeframe analysis
"""

from hean.physics.engine import PhysicsEngine, PhysicsState
from hean.physics.rust_bridge import (
    market_temperature,
    market_entropy,
    detect_phase,
    szilard_profit,
    information_bits,
    thermal_efficiency,
    RUST_AVAILABLE,
)
from hean.physics.participant_classifier import ParticipantClassifier
from hean.physics.anomaly_detector import MarketAnomalyDetector
from hean.physics.temporal_stack import TemporalStack
from hean.physics.cross_market import CrossMarketImpulse
from hean.physics.emotion_arbitrage import EmotionArbitrage
from hean.physics.phase_detector import SSDMode, ResonanceState

__all__ = [
    "PhysicsEngine",
    "PhysicsState",
    "ParticipantClassifier",
    "MarketAnomalyDetector",
    "TemporalStack",
    "CrossMarketImpulse",
    "EmotionArbitrage",
    "SSDMode",
    "ResonanceState",
    "market_temperature",
    "market_entropy",
    "detect_phase",
    "szilard_profit",
    "information_bits",
    "thermal_efficiency",
    "RUST_AVAILABLE",
]
