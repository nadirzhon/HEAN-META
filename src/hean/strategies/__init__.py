"""Trading strategies module."""

from hean.strategies.base import BaseStrategy
from hean.strategies.basis_arbitrage import BasisArbitrage
from hean.strategies.correlation_arb import CorrelationArbitrage
from hean.strategies.edge_confirmation import EdgeConfirmationLoop
from hean.strategies.enhanced_grid import EnhancedGridStrategy
from hean.strategies.funding_harvester import FundingHarvester
from hean.strategies.hf_scalping import HFScalpingStrategy
from hean.strategies.impulse_engine import ImpulseEngine
from hean.strategies.inventory_neutral_mm import InventoryNeutralMM
from hean.strategies.liquidity_sweep import LiquiditySweepDetector
from hean.strategies.momentum_trader import MomentumTrader
from hean.strategies.multi_factor_confirmation import MultiFactorConfirmation
from hean.strategies.rebate_farmer import RebateFarmer
from hean.strategies.sentiment_strategy import SentimentStrategy

__all__ = [
    "BaseStrategy",
    "BasisArbitrage",
    "CorrelationArbitrage",
    "EdgeConfirmationLoop",
    "EnhancedGridStrategy",
    "FundingHarvester",
    "HFScalpingStrategy",
    "ImpulseEngine",
    "InventoryNeutralMM",
    "LiquiditySweepDetector",
    "MomentumTrader",
    "MultiFactorConfirmation",
    "RebateFarmer",
    "SentimentStrategy",
]
