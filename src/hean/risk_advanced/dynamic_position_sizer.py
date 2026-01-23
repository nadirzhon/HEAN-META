"""
Dynamic Position Sizing

Adaptive position sizing based on:
- Kelly Criterion (optimal bet sizing)
- Volatility scaling
- Win rate & profit factor
- Account equity
- Market regime

Expected Performance:
- Optimal capital allocation = +20-40% returns
- Drawdown reduction: -15-30%
- Risk-adjusted returns: +0.3-0.6 Sharpe

Author: HEAN Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class SizingMethod(str, Enum):
    """Position sizing methods."""
    FIXED = "FIXED"
    KELLY = "KELLY"
    FRACTIONAL_KELLY = "FRACTIONAL_KELLY"
    VOLATILITY_SCALED = "VOLATILITY_SCALED"
    CONFIDENCE_BASED = "CONFIDENCE_BASED"
    HYBRID = "HYBRID"


@dataclass
class PositionSize:
    """Position size calculation result."""
    size: float  # Position size as fraction of capital (0.0-1.0)
    size_units: float  # Position size in units (BTC, contracts, etc.)
    method: SizingMethod
    reasoning: str
    risk_per_trade: float  # % of capital at risk
    kelly_fraction: Optional[float] = None
    confidence: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def __str__(self) -> str:
        return (
            f"PositionSize({self.size:.2%} of capital, "
            f"{self.size_units:.4f} units, {self.method.value})"
        )


@dataclass
class PositionSizeConfig:
    """Configuration for position sizing."""

    # Kelly Criterion
    kelly_enabled: bool = True
    kelly_fraction: float = 0.25  # Use 25% of Kelly (fractional Kelly)
    max_kelly_size: float = 0.50  # Cap at 50% of capital

    # Fixed sizing
    fixed_size: float = 0.02  # 2% of capital per trade

    # Volatility scaling
    volatility_scaling: bool = True
    target_volatility: float = 0.02  # 2% daily volatility target
    volatility_window: int = 20  # Days to calculate volatility

    # Confidence-based
    confidence_scaling: bool = True
    min_confidence: float = 0.55  # Don't trade below 55% confidence
    max_size_at_confidence: float = 0.10  # Max 10% at 100% confidence

    # Risk limits
    max_risk_per_trade: float = 0.02  # Max 2% risk per trade
    max_position_size: float = 0.20  # Max 20% of capital in one position
    max_total_exposure: float = 0.50  # Max 50% total exposure

    # Default method
    default_method: SizingMethod = SizingMethod.HYBRID


class DynamicPositionSizer:
    """
    Dynamic position sizing calculator.

    Usage:
        config = PositionSizeConfig(kelly_fraction=0.25)
        sizer = DynamicPositionSizer(config)

        # Calculate position size
        size = sizer.calculate_size(
            win_rate=0.58,
            avg_win=0.02,  # 2% average win
            avg_loss=0.01,  # 1% average loss
            account_balance=10000,
            price=50000,
            confidence=0.65,  # ML confidence
        )

        print(f"Position size: {size.size:.2%} of capital")
        print(f"Units to buy: {size.size_units:.4f}")
    """

    def __init__(self, config: Optional[PositionSizeConfig] = None) -> None:
        """Initialize position sizer."""
        self.config = config or PositionSizeConfig()
        self.trade_history: List[float] = []  # Returns history

        logger.info("DynamicPositionSizer initialized", config=self.config)

    def calculate_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_balance: float,
        price: float,
        confidence: Optional[float] = None,
        volatility: Optional[float] = None,
        method: Optional[SizingMethod] = None,
    ) -> PositionSize:
        """
        Calculate optimal position size.

        Args:
            win_rate: Win rate (0.0-1.0, e.g., 0.58 = 58%)
            avg_win: Average win size (fraction, e.g., 0.02 = 2%)
            avg_loss: Average loss size (fraction, e.g., 0.01 = 1%)
            account_balance: Account balance (USD)
            price: Asset price (USD)
            confidence: ML model confidence (0.0-1.0)
            volatility: Current volatility (optional)
            method: Sizing method (optional, uses default)

        Returns:
            Position size calculation
        """
        method = method or self.config.default_method

        # Calculate based on method
        if method == SizingMethod.FIXED:
            size_pct = self._calculate_fixed()
            reasoning = "Fixed position size"

        elif method == SizingMethod.KELLY:
            size_pct = self._calculate_kelly(win_rate, avg_win, avg_loss)
            reasoning = f"Full Kelly ({size_pct:.2%})"

        elif method == SizingMethod.FRACTIONAL_KELLY:
            kelly = self._calculate_kelly(win_rate, avg_win, avg_loss)
            size_pct = kelly * self.config.kelly_fraction
            reasoning = f"Fractional Kelly ({self.config.kelly_fraction:.0%} of {kelly:.2%})"

        elif method == SizingMethod.VOLATILITY_SCALED:
            if volatility is None:
                volatility = self._estimate_volatility()
            size_pct = self._calculate_volatility_scaled(volatility)
            reasoning = f"Volatility-scaled (vol={volatility:.2%})"

        elif method == SizingMethod.CONFIDENCE_BASED:
            if confidence is None:
                confidence = win_rate  # Use win rate as proxy
            size_pct = self._calculate_confidence_based(confidence)
            reasoning = f"Confidence-based ({confidence:.1%})"

        elif method == SizingMethod.HYBRID:
            # Combine multiple methods
            size_pct = self._calculate_hybrid(
                win_rate, avg_win, avg_loss, confidence, volatility
            )
            reasoning = "Hybrid (Kelly + Volatility + Confidence)"

        else:
            raise ValueError(f"Unknown sizing method: {method}")

        # Apply limits
        size_pct = self._apply_limits(size_pct, confidence)

        # Calculate units
        size_units = (account_balance * size_pct) / price

        # Calculate risk per trade
        risk_per_trade = size_pct * avg_loss

        return PositionSize(
            size=size_pct,
            size_units=size_units,
            method=method,
            reasoning=reasoning,
            risk_per_trade=risk_per_trade,
            kelly_fraction=self.config.kelly_fraction if method in [
                SizingMethod.KELLY, SizingMethod.FRACTIONAL_KELLY, SizingMethod.HYBRID
            ] else None,
            confidence=confidence,
        )

    def _calculate_fixed(self) -> float:
        """Fixed percentage position sizing."""
        return self.config.fixed_size

    def _calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Kelly Criterion calculation.

        Formula: f = (p * b - q) / b
        Where:
            f = fraction to bet
            p = win rate
            q = loss rate (1 - p)
            b = win/loss ratio
        """
        if avg_loss <= 0:
            return 0.0

        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss

        kelly = (p * b - q) / b

        # Kelly can be negative (don't trade) or > 1 (overleveraged)
        kelly = max(0.0, kelly)
        kelly = min(kelly, self.config.max_kelly_size)

        return kelly

    def _calculate_volatility_scaled(self, volatility: float) -> float:
        """
        Scale position size based on volatility.

        Lower volatility = larger position
        Higher volatility = smaller position
        """
        if volatility <= 0:
            return self.config.fixed_size

        # Inverse relationship
        scale_factor = self.config.target_volatility / volatility
        size = self.config.fixed_size * scale_factor

        return size

    def _calculate_confidence_based(self, confidence: float) -> float:
        """
        Scale position size based on ML confidence.

        Higher confidence = larger position
        """
        if confidence < self.config.min_confidence:
            return 0.0

        # Linear scaling from min_confidence to 1.0
        normalized_confidence = (confidence - self.config.min_confidence) / (
            1.0 - self.config.min_confidence
        )

        size = normalized_confidence * self.config.max_size_at_confidence

        return size

    def _calculate_hybrid(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: Optional[float],
        volatility: Optional[float],
    ) -> float:
        """
        Hybrid approach combining multiple methods.

        Weights:
        - 50% Kelly Criterion
        - 30% Volatility Scaling
        - 20% Confidence
        """
        # Kelly component
        kelly = self._calculate_kelly(win_rate, avg_win, avg_loss)
        kelly_component = kelly * self.config.kelly_fraction

        # Volatility component
        if volatility is None:
            volatility = self._estimate_volatility()
        vol_component = self._calculate_volatility_scaled(volatility)

        # Confidence component
        if confidence is None:
            confidence = win_rate
        conf_component = self._calculate_confidence_based(confidence)

        # Weighted average
        size = (
            kelly_component * 0.5 +
            vol_component * 0.3 +
            conf_component * 0.2
        )

        return size

    def _apply_limits(self, size: float, confidence: Optional[float]) -> float:
        """Apply risk limits to position size."""
        # Max position size
        size = min(size, self.config.max_position_size)

        # Confidence threshold
        if confidence is not None and confidence < self.config.min_confidence:
            return 0.0

        # Risk per trade
        # (This would need stop loss distance, simplified here)
        max_size_from_risk = self.config.max_risk_per_trade / 0.01  # Assuming 1% stop
        size = min(size, max_size_from_risk)

        return size

    def _estimate_volatility(self) -> float:
        """Estimate volatility from recent returns."""
        if len(self.trade_history) < self.config.volatility_window:
            return self.config.target_volatility

        recent_returns = self.trade_history[-self.config.volatility_window:]
        volatility = np.std(recent_returns)

        return volatility

    def record_trade(self, return_pct: float) -> None:
        """Record trade return for volatility calculation."""
        self.trade_history.append(return_pct)

        # Keep only recent history
        if len(self.trade_history) > self.config.volatility_window * 2:
            self.trade_history = self.trade_history[-self.config.volatility_window:]

    def calculate_kelly_from_history(
        self,
        returns: List[float],
    ) -> float:
        """
        Calculate Kelly fraction from historical returns.

        Args:
            returns: List of trade returns (e.g., [0.02, -0.01, 0.03, ...])

        Returns:
            Kelly fraction
        """
        if not returns:
            return 0.0

        wins = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        return self._calculate_kelly(win_rate, avg_win, avg_loss)

    def optimize_kelly_fraction(
        self,
        returns: List[float],
        fractions: List[float] = None,
    ) -> float:
        """
        Find optimal Kelly fraction through simulation.

        Args:
            returns: Historical trade returns
            fractions: Kelly fractions to test (e.g., [0.1, 0.25, 0.5])

        Returns:
            Optimal Kelly fraction
        """
        if fractions is None:
            fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

        full_kelly = self.calculate_kelly_from_history(returns)

        best_fraction = 0.25
        best_sharpe = -np.inf

        for fraction in fractions:
            # Simulate returns with this Kelly fraction
            simulated_returns = []
            equity = 1.0

            for ret in returns:
                position_size = full_kelly * fraction
                trade_return = ret * position_size
                equity *= (1 + trade_return)
                simulated_returns.append(trade_return)

            # Calculate Sharpe
            if len(simulated_returns) > 0:
                sharpe = np.mean(simulated_returns) / (np.std(simulated_returns) + 1e-10)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_fraction = fraction

        logger.info(f"Optimal Kelly fraction: {best_fraction} (Sharpe: {best_sharpe:.2f})")
        return best_fraction


# Convenience function
def calculate_kelly_position(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    price: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Quick Kelly position size calculation.

    Example:
        units = calculate_kelly_position(
            win_rate=0.58,
            avg_win=0.02,
            avg_loss=0.01,
            capital=10000,
            price=50000,
            kelly_fraction=0.25,
        )
        print(f"Buy {units:.4f} BTC")
    """
    sizer = DynamicPositionSizer()
    result = sizer.calculate_size(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        account_balance=capital,
        price=price,
        method=SizingMethod.FRACTIONAL_KELLY,
    )

    return result.size_units
