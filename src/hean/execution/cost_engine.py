"""Cost estimation engine for small capital mode.

Estimates total trading costs including:
- Fee costs (maker/taker)
- Spread costs
- Slippage estimates
- Total cost basis points for cost/edge comparison
"""

from typing import Literal

from hean.config import settings
from hean.core.types import Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class CostEngine:
    """Estimates trading costs for small capital profit mode."""

    def __init__(self) -> None:
        """Initialize cost engine with conservative defaults."""
        # Fee structure (Bybit spot/perpetual)
        self._maker_fee_bps = 1.0  # 0.01% = 1 bps (maker fee, often negative/rebate on Bybit)
        self._taker_fee_bps = 6.0  # 0.06% = 6 bps (taker fee on Bybit)

        # For conservative estimation, use positive maker fee (safer assumption)
        # In reality, Bybit offers maker rebates on many pairs
        self._conservative_maker_fee_bps = 1.0

        # Slippage model parameters
        self._slippage_volatility_multiplier = 50.0  # Scales volatility to bps
        self._slippage_base_bps = 2.0  # Base slippage for normal conditions

        # Liquidity proxy thresholds
        self._low_liquidity_spread_bps = 15.0  # Spread above this indicates low liquidity

        # Running metrics
        self._total_cost_sum = 0.0
        self._cost_count = 0

    def estimate_total_cost_bps(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        order_type: Literal["maker", "taker"],
        tick: Tick | None = None,
        volatility_proxy: float = 0.0,
    ) -> dict[str, float]:
        """Estimate total trading cost in basis points.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            side: Order side ("buy" or "sell")
            qty: Order quantity
            order_type: "maker" (post-only limit) or "taker" (market/aggressive)
            tick: Current market tick (for spread calculation)
            volatility_proxy: Volatility proxy (e.g., ATR / price) for slippage estimation

        Returns:
            Dictionary with breakdown:
            - fee_bps: Fee cost (one-way)
            - spread_bps: Half-spread cost
            - slippage_bps: Estimated slippage
            - total_cost_bps: Total round-trip cost estimate
            - breakdown: Detailed string description
        """
        # 1. Fee estimation
        if order_type == "maker":
            fee_bps = self._conservative_maker_fee_bps
        else:  # taker
            fee_bps = self._taker_fee_bps

        # 2. Spread estimation
        spread_bps = self._estimate_spread_bps(tick)

        # 3. Slippage estimation
        slippage_bps = self._estimate_slippage_bps(
            symbol=symbol,
            order_type=order_type,
            spread_bps=spread_bps,
            volatility_proxy=volatility_proxy,
        )

        # 4. Total cost (round-trip: entry + exit)
        # We double fee because we enter AND exit
        # Spread is paid once (half-spread on entry, half-spread on exit = full spread)
        # Slippage is paid on each leg
        total_cost_bps = (2 * fee_bps) + spread_bps + slippage_bps

        # Track metrics
        self._total_cost_sum += total_cost_bps
        self._cost_count += 1

        breakdown = (
            f"fee={fee_bps:.1f} bps ({order_type}), "
            f"spread={spread_bps:.1f} bps, "
            f"slippage={slippage_bps:.1f} bps"
        )

        result = {
            "fee_bps": fee_bps,
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "total_cost_bps": total_cost_bps,
            "breakdown": breakdown,
        }

        logger.debug(
            f"Cost estimate for {symbol} {side} {order_type}: "
            f"total={total_cost_bps:.1f} bps ({breakdown})"
        )

        return result

    def _estimate_spread_bps(self, tick: Tick | None) -> float:
        """Estimate bid-ask spread cost in basis points.

        Returns half-spread (the cost of crossing spread once).
        """
        if not tick or not tick.bid or not tick.ask or tick.price <= 0:
            # Conservative fallback if no tick data
            return 5.0  # Assume 5 bps half-spread (10 bps full spread)

        spread = tick.ask - tick.bid
        spread_bps = (spread / tick.price) * 10000 if tick.price > 0 else 0

        # Half-spread (cost of crossing once)
        half_spread_bps = spread_bps / 2.0

        return max(half_spread_bps, 1.0)  # Minimum 1 bps

    def _estimate_slippage_bps(
        self,
        symbol: str,
        order_type: Literal["maker", "taker"],
        spread_bps: float,
        volatility_proxy: float,
    ) -> float:
        """Estimate execution slippage in basis points.

        Slippage depends on:
        - Order type: maker has low slippage (limit order), taker has higher
        - Market conditions: spread width (liquidity proxy), volatility
        - Conservative estimation for risk management

        Args:
            symbol: Trading symbol
            order_type: "maker" or "taker"
            spread_bps: Current spread in bps (liquidity proxy)
            volatility_proxy: Volatility measure (e.g., ATR/price or rolling volatility)

        Returns:
            Estimated slippage in basis points
        """
        if order_type == "maker":
            # Maker orders: minimal slippage (you set the price)
            # But there's "adverse selection" risk and partial fill risk
            # Use small base slippage
            base_slippage = self._slippage_base_bps * 0.5
        else:  # taker
            # Taker orders: cross the spread and eat into orderbook
            # Higher slippage, especially in wide spreads
            base_slippage = self._slippage_base_bps * 1.5

        # Liquidity penalty: wider spreads = more slippage
        liquidity_penalty = 0.0
        if spread_bps > self._low_liquidity_spread_bps:
            # Low liquidity detected
            liquidity_penalty = (spread_bps - self._low_liquidity_spread_bps) * 0.5

        # Volatility penalty: higher volatility = more price uncertainty
        volatility_penalty = volatility_proxy * self._slippage_volatility_multiplier
        volatility_penalty = min(volatility_penalty, 10.0)  # Cap at 10 bps

        total_slippage = base_slippage + liquidity_penalty + volatility_penalty

        # Conservative ceiling for small capital mode
        return min(total_slippage, settings.max_slippage_estimate_bps)

    def get_avg_cost_bps(self) -> float:
        """Return average cost across all estimations."""
        if self._cost_count == 0:
            return 0.0
        return self._total_cost_sum / self._cost_count

    def should_block_by_spread(self, tick: Tick | None) -> tuple[bool, str]:
        """Check if trade should be blocked due to wide spread.

        Returns:
            (should_block, reason_code)
        """
        if not settings.small_capital_mode:
            return False, ""

        if not tick or not tick.bid or not tick.ask or tick.price <= 0:
            return True, "STALE_MARKET_DATA"

        spread_bps = ((tick.ask - tick.bid) / tick.price) * 10000

        if spread_bps > settings.max_spread_bps:
            return True, "SPREAD_TOO_WIDE"

        return False, ""

    def should_block_by_min_notional(
        self,
        qty: float,
        price: float,
    ) -> tuple[bool, str]:
        """Check if order meets minimum notional requirements.

        Returns:
            (should_block, reason_code)
        """
        if not settings.small_capital_mode:
            return False, ""

        notional = qty * price
        if notional < settings.min_notional_usd:
            return True, "MIN_NOTIONAL_LIMIT"

        return False, ""

    def get_metrics(self) -> dict[str, float]:
        """Return cost engine metrics."""
        return {
            "avg_cost_bps": self.get_avg_cost_bps(),
            "cost_estimates_count": float(self._cost_count),
        }

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._total_cost_sum = 0.0
        self._cost_count = 0
