"""
Example: Order Book Analysis

Shows how to:
1. Detect whale walls
2. Calculate order book imbalance
3. Detect hidden liquidity
4. Calculate VPIN (toxic flow)
5. Find support/resistance levels

Author: HEAN Team
"""

import asyncio
from datetime import datetime

import numpy as np
from loguru import logger

from hean.market_data import (
    OrderBookAnalyzer,
    OrderBookSnapshot,
)


def generate_realistic_orderbook(
    mid_price: float = 50000,
    spread_pct: float = 0.02,
    levels: int = 50,
) -> OrderBookSnapshot:
    """Generate realistic order book data."""
    spread = mid_price * spread_pct / 100
    best_bid = mid_price - spread / 2
    best_ask = mid_price + spread / 2

    # Generate bids (descending prices)
    bids = []
    for i in range(levels):
        price = best_bid - i * 10
        # Exponential distribution of sizes
        size = np.random.exponential(1.0) + 0.1

        # Add occasional whale orders
        if np.random.random() < 0.05:  # 5% chance
            size *= np.random.uniform(5, 15)  # 5-15x larger

        bids.append((price, size))

    # Generate asks (ascending prices)
    asks = []
    for i in range(levels):
        price = best_ask + i * 10
        size = np.random.exponential(1.0) + 0.1

        # Add occasional whale orders
        if np.random.random() < 0.05:
            size *= np.random.uniform(5, 15)

        asks.append((price, size))

    return OrderBookSnapshot(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        bids=bids,
        asks=asks,
    )


async def example_whale_detection() -> None:
    """Example: Detect whale walls."""
    logger.info("=== Whale Wall Detection ===")

    analyzer = OrderBookAnalyzer(whale_threshold_ratio=5.0)

    # Generate order book with whales
    snapshot = generate_realistic_orderbook()

    # Detect whales
    whale_signals = analyzer.detect_whale_walls(snapshot, max_distance_pct=1.0)

    logger.info(f"Detected {len(whale_signals)} whale orders")

    for signal in whale_signals[:5]:
        logger.info(f"  {signal}")
        logger.info(f"    Size: {signal.size:.2f} ({signal.size_ratio:.1f}x average)")


async def example_imbalance_analysis() -> None:
    """Example: Order book imbalance analysis."""
    logger.info("\n=== Order Book Imbalance ===")

    analyzer = OrderBookAnalyzer(imbalance_window=10)

    # Create imbalanced order book (more bids = bullish)
    mid = 50000
    bids = [(mid - i*10, np.random.uniform(2, 5)) for i in range(20)]
    asks = [(mid + i*10, np.random.uniform(0.5, 1.5)) for i in range(20)]

    snapshot = OrderBookSnapshot(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        bids=bids,
        asks=asks,
    )

    # Calculate imbalance
    imbalance = analyzer.calculate_imbalance(snapshot)

    logger.info(f"Imbalance: {imbalance}")
    logger.info(f"  Bid volume: {imbalance.bid_volume:.2f}")
    logger.info(f"  Ask volume: {imbalance.ask_volume:.2f}")
    logger.info(f"  Ratio: {imbalance.imbalance_ratio:.3f}")
    logger.info(f"  Direction: {imbalance.predicted_direction}")


async def example_vpin_calculation() -> None:
    """Example: VPIN (toxic flow) calculation."""
    logger.info("\n=== VPIN Calculation ===")

    analyzer = OrderBookAnalyzer(vpin_window=100)

    # Generate trade history
    trades = []
    for i in range(200):
        # Simulate informed trading (more buys when price rising)
        if np.random.random() < 0.7:  # 70% buy pressure
            side = 'buy'
        else:
            side = 'sell'

        trades.append({
            'price': 50000 + np.random.randn() * 10,
            'size': np.random.exponential(1.0),
            'side': side,
        })

    # Calculate VPIN
    vpin = analyzer.calculate_vpin(trades, volume_bucket=10.0)

    logger.info(f"VPIN: {vpin}")
    logger.info(f"  Value: {vpin.vpin:.3f}")
    logger.info(f"  Toxicity: {vpin.toxicity_level.value}")
    logger.info(f"  Warning: {'⚠️ HIGH TOXIC FLOW' if vpin.warning else '✅ Normal flow'}")


async def example_support_resistance() -> None:
    """Example: Support/resistance detection."""
    logger.info("\n=== Support & Resistance Levels ===")

    analyzer = OrderBookAnalyzer()

    # Create order book with clustering
    mid = 50000

    # Cluster bids around 49900 (support)
    bids = []
    for i in range(10):
        bids.append((49900 + np.random.uniform(-5, 5), 2.0))
    for i in range(10, 30):
        bids.append((mid - i*20, 0.5))

    # Cluster asks around 50100 (resistance)
    asks = []
    for i in range(10):
        asks.append((50100 + np.random.uniform(-5, 5), 2.5))
    for i in range(10, 30):
        asks.append((mid + i*20, 0.5))

    snapshot = OrderBookSnapshot(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        bids=sorted(bids, key=lambda x: -x[0]),
        asks=sorted(asks, key=lambda x: x[0]),
    )

    # Find levels
    levels = analyzer.find_support_resistance(snapshot)

    logger.info(f"Support levels: {[f'${l:.0f}' for l in levels['support'][:5]]}")
    logger.info(f"Resistance levels: {[f'${l:.0f}' for l in levels['resistance'][:5]]}")


async def example_comprehensive_analysis() -> None:
    """Example: Comprehensive order book analysis."""
    logger.info("\n=== Comprehensive Analysis ===")

    analyzer = OrderBookAnalyzer()

    # Generate realistic order book
    snapshot = generate_realistic_orderbook(mid_price=50000)

    # Get full summary
    summary = analyzer.get_summary(snapshot)

    logger.info(f"Symbol: {summary['symbol']}")
    logger.info(f"Mid Price: ${summary['mid_price']:.2f}")
    logger.info(f"Spread: ${summary['spread']:.2f} ({summary['spread_pct']:.4f}%)")
    logger.info(f"\nWhale Signals: {len(summary['whale_signals'])}")
    logger.info(f"Imbalance: {summary['imbalance'].imbalance_ratio:.3f} ({summary['imbalance'].predicted_direction})")
    logger.info(f"Bid Depth: {summary['depth']['bid_depth']:.2f}")
    logger.info(f"Ask Depth: {summary['depth']['ask_depth']:.2f}")
    logger.info(f"Depth Imbalance: {summary['depth']['depth_imbalance']:.3f}")


async def example_trading_strategy() -> None:
    """Example: Trading strategy using order book signals."""
    logger.info("\n=== Trading Strategy Integration ===")

    analyzer = OrderBookAnalyzer(
        whale_threshold_ratio=5.0,
        imbalance_window=10,
    )

    signals_generated = []

    # Simulate market updates
    for i in range(10):
        snapshot = generate_realistic_orderbook(mid_price=50000 + i * 50)

        # Analyze
        whales = analyzer.detect_whale_walls(snapshot)
        imbalance = analyzer.calculate_imbalance(snapshot)
        depth = analyzer.get_orderbook_depth(snapshot)

        # === Trading Logic ===

        # Signal 1: Whale bid wall + bullish imbalance
        if (any(w.side.value == "BID" and w.strength.value in ["STRONG", "EXTREME"]
                for w in whales) and
            imbalance.imbalance_ratio > 0.3):

            signals_generated.append({
                'type': 'BUY',
                'reason': 'Whale bid support + bullish imbalance',
                'confidence': abs(imbalance.imbalance_ratio),
                'price': snapshot.mid_price,
            })

        # Signal 2: Whale ask wall + bearish imbalance
        elif (any(w.side.value == "ASK" and w.strength.value in ["STRONG", "EXTREME"]
                  for w in whales) and
              imbalance.imbalance_ratio < -0.3):

            signals_generated.append({
                'type': 'SELL',
                'reason': 'Whale ask resistance + bearish imbalance',
                'confidence': abs(imbalance.imbalance_ratio),
                'price': snapshot.mid_price,
            })

        # Signal 3: Strong depth imbalance (no whales needed)
        elif abs(depth['depth_imbalance']) > 0.5:
            direction = 'BUY' if depth['depth_imbalance'] > 0 else 'SELL'
            signals_generated.append({
                'type': direction,
                'reason': 'Strong depth imbalance',
                'confidence': abs(depth['depth_imbalance']),
                'price': snapshot.mid_price,
            })

    logger.info(f"Generated {len(signals_generated)} trading signals")
    for sig in signals_generated[:5]:
        logger.info(f"  {sig['type']} @ ${sig['price']:.0f}")
        logger.info(f"    Reason: {sig['reason']}")
        logger.info(f"    Confidence: {sig['confidence']:.1%}")


async def main() -> None:
    """Run all examples."""
    await example_whale_detection()
    await example_imbalance_analysis()
    await example_vpin_calculation()
    await example_support_resistance()
    await example_comprehensive_analysis()
    await example_trading_strategy()

    logger.info("\n" + "="*60)
    logger.info("✅ All order book analysis examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())
