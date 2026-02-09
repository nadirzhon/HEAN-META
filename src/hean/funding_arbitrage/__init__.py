"""
Multi-Exchange Funding Arbitrage

Monitors funding rates across multiple exchanges and identifies arbitrage opportunities.

Usage:
    from hean.funding_arbitrage import FundingArbitrageAggregator

    aggregator = FundingArbitrageAggregator()
    await aggregator.initialize()

    opportunities = await aggregator.find_opportunities("BTC")
    for opp in opportunities:
        if opp.should_trade:
            await execute_trade(opp)
"""

from .aggregator import FundingArbitrageAggregator
from .binance_funding import BinanceFundingClient
from .bybit_funding import BybitFundingClient
from .models import (
    ArbitrageSignal,
    ExchangeFundingRate,
    ExchangeName,
    FundingOpportunity,
)
from .okx_funding import OKXFundingClient
from .strategy import FundingArbitrageStrategy

__all__ = [
    "ExchangeFundingRate",
    "FundingOpportunity",
    "ArbitrageSignal",
    "ExchangeName",
    "BybitFundingClient",
    "BinanceFundingClient",
    "OKXFundingClient",
    "FundingArbitrageAggregator",
    "FundingArbitrageStrategy",
]
