"""
Example: Phase 2 Advanced ML Features

Demonstrates:
1. Sentiment Analysis (Twitter, Reddit, News, Fear & Greed)
2. On-Chain Metrics (Exchange flows, MVRV, Funding rates)
3. Optuna Hyperparameter Optimization
4. Dynamic Position Sizing (Kelly Criterion)

Author: HEAN Team
"""

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


async def example_sentiment_analysis() -> None:
    """Example: Multi-source sentiment analysis."""
    logger.info("=== Sentiment Analysis ===")

    from hean.alternative_data import SentimentEngine, SentimentConfig

    # Configure
    config = SentimentConfig(
        twitter_enabled=True,
        reddit_enabled=True,
        fear_greed_enabled=True,
        min_sample_size=10,
    )

    engine = SentimentEngine(config)

    # Analyze aggregate sentiment
    signal = await engine.analyze_sentiment(symbol="BTC")

    logger.info(f"Aggregate Sentiment: {signal}")
    logger.info(f"  Direction: {signal.direction}")
    logger.info(f"  Strength: {signal.strength:.1%}")
    logger.info(f"  Score: {signal.aggregate_score:.2f}")
    logger.info(f"  Sources: {len(signal.sources)}")

    # Individual sources
    logger.info("\nIndividual Sources:")
    for score in signal.sources:
        logger.info(f"  {score.source.value}: {score.label.value} ({score.score:.2f})")

    # Trading decision
    if signal.direction == "BUY" and signal.strength > 0.6:
        logger.info("\n✅ STRONG BUY SIGNAL from sentiment!")
    elif signal.direction == "SELL" and signal.strength > 0.6:
        logger.info("\n⚠️ STRONG SELL SIGNAL from sentiment!")


async def example_onchain_analysis() -> None:
    """Example: On-chain metrics analysis."""
    logger.info("\n=== On-Chain Analysis ===")

    from hean.alternative_data.onchain_metrics import (
        OnChainCollector,
        OnChainConfig,
    )

    # Configure
    config = OnChainConfig(
        whale_inflow_threshold_btc=100.0,
        mvrv_overbought=3.5,
        funding_bearish=0.05,
    )

    collector = OnChainCollector(config)

    # Get metrics
    metrics = await collector.get_metrics("BTC")

    logger.info("On-Chain Metrics:")
    logger.info(f"  Exchange Inflow (24h): {metrics.exchange_inflow_24h:.2f} BTC")
    logger.info(f"  Exchange Outflow (24h): {metrics.exchange_outflow_24h:.2f} BTC")
    logger.info(f"  Net Flow: {metrics.net_flow_24h:.2f} BTC")
    logger.info(f"  MVRV Ratio: {metrics.mvrv_ratio:.2f}")
    logger.info(f"  Funding Rate: {metrics.funding_rate:.4f}%")
    logger.info(f"  Open Interest: ${metrics.open_interest/1e9:.2f}B")
    logger.info(f"  Long/Short Ratio: {metrics.long_short_ratio:.2f}")

    # Analyze signals
    signals = await collector.analyze_signals(metrics)

    logger.info(f"\nOn-Chain Signals: {len(signals)}")
    for signal in signals:
        logger.info(f"  {signal}")

    # Strongest signal
    if signals:
        strongest = max(signals, key=lambda s: s.strength)
        logger.info(f"\n🎯 Strongest Signal: {strongest}")


async def example_hyperparameter_optimization() -> None:
    """Example: Optuna hyperparameter optimization."""
    logger.info("\n=== Hyperparameter Optimization ===")

    from hean.optimization import HyperparameterTuner, OptunaConfig

    # Simulated backtest function
    def backtest_strategy(params):
        """Simulate backtest with given parameters."""
        # In production, run actual backtest
        # Here, simulate with some randomness + parameter influence

        rsi_period = params['rsi_period']
        oversold = params['oversold']
        overbought = params['overbought']

        # Simulate Sharpe ratio (better params = higher Sharpe)
        base_sharpe = 1.5
        rsi_bonus = (rsi_period - 20) / 50  # Optimal around 14
        threshold_bonus = (70 - abs(oversold - 30)) / 100
        noise = np.random.randn() * 0.2

        sharpe = base_sharpe + rsi_bonus + threshold_bonus + noise

        return sharpe

    # Define search space
    search_space = {
        'rsi_period': (10, 30, 'int'),
        'oversold': (20, 35, 'int'),
        'overbought': (65, 80, 'int'),
    }

    # Optimize
    config = OptunaConfig(
        study_name="rsi_optimization",
        n_trials=50,
        direction="maximize",
    )

    tuner = HyperparameterTuner(config)

    logger.info("Starting optimization (50 trials)...")
    result = tuner.optimize(
        objective_func=backtest_strategy,
        search_space=search_space,
    )

    logger.info(f"\nOptimization Results:")
    logger.info(f"  Best Sharpe: {result.best_value:.3f}")
    logger.info(f"  Best Params: {result.best_params}")
    logger.info(f"  Total Trials: {result.n_trials}")

    if result.param_importances:
        logger.info("\nParameter Importances:")
        for param, importance in result.param_importances.items():
            logger.info(f"  {param}: {importance:.3f}")


async def example_multi_objective_optimization() -> None:
    """Example: Multi-objective optimization (Sharpe + Drawdown)."""
    logger.info("\n=== Multi-Objective Optimization ===")

    from hean.optimization import HyperparameterTuner, OptunaConfig

    def backtest_multi(params):
        """Return [sharpe, max_drawdown]."""
        rsi_period = params['rsi_period']

        # Simulate metrics
        sharpe = 2.0 + (rsi_period - 14) / 10 + np.random.randn() * 0.2
        drawdown = 0.15 - (rsi_period - 14) / 100 + np.random.rand() * 0.05

        return [sharpe, drawdown]

    search_space = {
        'rsi_period': (10, 30, 'int'),
        'position_size': (0.01, 0.05, 'float'),
    }

    config = OptunaConfig(
        study_name="multi_objective",
        multi_objective=True,
        objectives=["sharpe_ratio", "max_drawdown"],
        directions=["maximize", "minimize"],
        n_trials=30,
    )

    tuner = HyperparameterTuner(config)

    logger.info("Running multi-objective optimization...")
    result = tuner.optimize_multi_objective(
        objective_func=backtest_multi,
        search_space=search_space,
    )

    logger.info(f"\nBest Solution (Pareto-optimal):")
    logger.info(f"  Sharpe: {result.best_values[0]:.3f}")
    logger.info(f"  Max DD: {result.best_values[1]:.1%}")
    logger.info(f"  Params: {result.best_params}")


async def example_dynamic_position_sizing() -> None:
    """Example: Dynamic position sizing."""
    logger.info("\n=== Dynamic Position Sizing ===")

    from hean.risk_advanced import DynamicPositionSizer, PositionSizeConfig

    # Configure
    config = PositionSizeConfig(
        kelly_fraction=0.25,  # Use 25% of Kelly
        volatility_scaling=True,
        confidence_scaling=True,
        max_position_size=0.20,
    )

    sizer = DynamicPositionSizer(config)

    # Example: Strategy with 58% win rate
    logger.info("Strategy Stats:")
    logger.info("  Win Rate: 58%")
    logger.info("  Avg Win: 2%")
    logger.info("  Avg Loss: 1%")
    logger.info("  ML Confidence: 65%")

    # Calculate position size
    size = sizer.calculate_size(
        win_rate=0.58,
        avg_win=0.02,
        avg_loss=0.01,
        account_balance=10000,
        price=50000,
        confidence=0.65,
    )

    logger.info(f"\nPosition Size: {size}")
    logger.info(f"  Capital: {size.size:.1%} (${10000 * size.size:.0f})")
    logger.info(f"  Units: {size.size_units:.4f} BTC")
    logger.info(f"  Risk: {size.risk_per_trade:.2%} per trade")
    logger.info(f"  Method: {size.method.value}")

    # Different confidence levels
    logger.info("\nPosition Size vs Confidence:")
    for conf in [0.55, 0.60, 0.65, 0.70, 0.80]:
        size = sizer.calculate_size(
            win_rate=0.58,
            avg_win=0.02,
            avg_loss=0.01,
            account_balance=10000,
            price=50000,
            confidence=conf,
        )
        logger.info(f"  {conf:.0%} confidence → {size.size:.1%} position")


async def example_kelly_from_history() -> None:
    """Example: Calculate Kelly from trade history."""
    logger.info("\n=== Kelly Criterion from History ===")

    from hean.risk_advanced import DynamicPositionSizer

    # Simulated trade history (returns)
    returns = [
        0.02, -0.01, 0.03, -0.01, 0.01,
        0.02, -0.02, 0.04, -0.01, 0.02,
        -0.01, 0.03, -0.01, 0.02, -0.01,
    ]

    sizer = DynamicPositionSizer()

    # Calculate Kelly
    kelly = sizer.calculate_kelly_from_history(returns)

    logger.info(f"Historical Returns: {returns[:5]}...")
    logger.info(f"Kelly Fraction: {kelly:.2%}")

    # Optimize Kelly fraction
    optimal_fraction = sizer.optimize_kelly_fraction(returns)

    logger.info(f"Optimal Kelly Fraction: {optimal_fraction:.0%}")


async def example_integrated_system() -> None:
    """Example: Integrate all Phase 2 components."""
    logger.info("\n=== Integrated Trading System ===")

    from hean.alternative_data import SentimentEngine
    from hean.alternative_data.onchain_metrics import OnChainCollector
    from hean.risk_advanced import DynamicPositionSizer

    # Initialize components
    sentiment_engine = SentimentEngine()
    onchain_collector = OnChainCollector()
    position_sizer = DynamicPositionSizer()

    # 1. Get sentiment
    sentiment = await sentiment_engine.analyze_sentiment("BTC")

    # 2. Get on-chain metrics
    onchain_metrics = await onchain_collector.get_metrics("BTC")
    onchain_signals = await onchain_collector.analyze_signals(onchain_metrics)

    # 3. Combined decision
    logger.info("Integrated Analysis:")
    logger.info(f"  Sentiment: {sentiment.direction} ({sentiment.strength:.1%})")
    logger.info(f"  On-Chain Signals: {len(onchain_signals)}")

    # Trading decision
    bullish_sentiment = sentiment.direction == "BUY" and sentiment.strength > 0.6
    bullish_onchain = any(
        s.direction == "BUY" and s.strength > 0.5
        for s in onchain_signals
    )

    if bullish_sentiment and bullish_onchain:
        logger.info("\n✅ STRONG BUY SIGNAL (Sentiment + On-Chain)")

        # Calculate position size
        size = position_sizer.calculate_size(
            win_rate=0.58,
            avg_win=0.02,
            avg_loss=0.01,
            account_balance=10000,
            price=50000,
            confidence=sentiment.strength,  # Use sentiment strength
        )

        logger.info(f"  Position Size: {size.size:.1%}")
        logger.info(f"  Units: {size.size_units:.4f} BTC")

    else:
        logger.info("\n⏸️ No strong signal - staying in cash")


async def main() -> None:
    """Run all Phase 2 examples."""
    await example_sentiment_analysis()
    await example_onchain_analysis()
    await example_hyperparameter_optimization()
    await example_multi_objective_optimization()
    await example_dynamic_position_sizing()
    await example_kelly_from_history()
    await example_integrated_system()

    logger.info("\n" + "="*60)
    logger.info("✅ All Phase 2 examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())
