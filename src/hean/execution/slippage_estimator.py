"""Slippage Estimator.

Estimates expected slippage in basis points based on:
1. Orderbook depth analysis
2. Historical slippage data (actual vs expected prices)

Blends both sources for robust estimation.
"""

from collections import deque

from hean.logging import get_logger

logger = get_logger(__name__)


class SlippageEstimator:
    """Estimate slippage for order execution."""

    def __init__(self, history_window: int = 100) -> None:
        """Initialize slippage estimator.

        Args:
            history_window: Number of historical trades to track per symbol
        """
        self._history_window = history_window
        # Historical slippage: deque of (side, slippage_bps) tuples
        self._slippage_history: dict[str, deque] = {}

    def estimate_slippage(
        self,
        symbol: str,
        side: str,
        size: float,
        orderbook_depth: list[tuple[float, float]] | None = None,
    ) -> float:
        """Estimate slippage in basis points.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            size: Order size in base currency
            orderbook_depth: List of (price, qty) tuples from orderbook

        Returns:
            Estimated slippage in basis points
        """
        # Get historical average slippage
        historical_avg = self._get_historical_average(symbol, side)

        # Get orderbook-based estimate
        if orderbook_depth:
            orderbook_est = self._estimate_from_orderbook(
                side, size, orderbook_depth
            )
        else:
            orderbook_est = None

        # Blend estimates
        if orderbook_est is not None and historical_avg is not None:
            # Blend: 70% orderbook, 30% historical
            blended = orderbook_est * 0.7 + historical_avg * 0.3
        elif orderbook_est is not None:
            blended = orderbook_est
        elif historical_avg is not None:
            blended = historical_avg
        else:
            # No data - use conservative default (5 bps)
            blended = 5.0

        logger.debug(
            f"[SlippageEstimator] {symbol} {side}: "
            f"orderbook={orderbook_est:.2f if orderbook_est else 'N/A'} bps, "
            f"historical={historical_avg:.2f if historical_avg else 'N/A'} bps, "
            f"blended={blended:.2f} bps"
        )

        return blended

    def _estimate_from_orderbook(
        self,
        side: str,
        size: float,
        orderbook_depth: list[tuple[float, float]],
    ) -> float | None:
        """Estimate slippage from orderbook depth.

        Walks through the orderbook to see how much price impact
        the order would have.

        Args:
            side: "buy" or "sell"
            size: Order size
            orderbook_depth: List of (price, qty) levels

        Returns:
            Estimated slippage in bps, or None if insufficient data
        """
        if not orderbook_depth:
            return None

        # Sort orderbook (ascending for buy, descending for sell)
        if side.lower() == "buy":
            # Walk up the ask side
            levels = sorted(orderbook_depth, key=lambda x: x[0])
        else:
            # Walk down the bid side
            levels = sorted(orderbook_depth, key=lambda x: x[0], reverse=True)

        if not levels:
            return None

        # Best price (reference)
        best_price = levels[0][0]

        # Walk through levels until size is filled
        remaining_size = size
        total_cost = 0.0

        for price, qty in levels:
            if remaining_size <= 0:
                break

            fill_qty = min(remaining_size, qty)
            total_cost += fill_qty * price
            remaining_size -= fill_qty

        if remaining_size > 0:
            # Not enough liquidity - assume high slippage
            return 20.0

        # Calculate average fill price
        avg_fill_price = total_cost / size

        # Slippage in bps
        slippage_bps = abs((avg_fill_price - best_price) / best_price) * 10000

        return slippage_bps

    def _get_historical_average(self, symbol: str, side: str) -> float | None:
        """Get average historical slippage for symbol and side.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"

        Returns:
            Average slippage in bps, or None if no history
        """
        history = self._slippage_history.get(symbol)
        if not history:
            return None

        # Filter by side
        side_history = [slip for s, slip in history if s.lower() == side.lower()]

        if not side_history:
            return None

        return sum(side_history) / len(side_history)

    def record_actual_slippage(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
    ) -> None:
        """Record actual slippage for learning.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            expected_price: Expected fill price (e.g., mid price)
            actual_price: Actual fill price
        """
        if expected_price == 0:
            return

        # Calculate slippage in bps
        slippage_bps = abs((actual_price - expected_price) / expected_price) * 10000

        # Initialize history for symbol
        if symbol not in self._slippage_history:
            self._slippage_history[symbol] = deque(maxlen=self._history_window)

        # Append (side, slippage_bps)
        self._slippage_history[symbol].append((side, slippage_bps))

        logger.debug(
            f"[SlippageEstimator] Recorded {symbol} {side}: "
            f"expected={expected_price:.2f}, actual={actual_price:.2f}, "
            f"slippage={slippage_bps:.2f} bps"
        )

    def get_stats(self, symbol: str) -> dict:
        """Get slippage statistics for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with buy/sell average slippage and sample counts
        """
        history = self._slippage_history.get(symbol)
        if not history:
            return {
                "buy_avg_bps": None,
                "sell_avg_bps": None,
                "buy_count": 0,
                "sell_count": 0,
            }

        buy_slips = [slip for side, slip in history if side.lower() == "buy"]
        sell_slips = [slip for side, slip in history if side.lower() == "sell"]

        return {
            "buy_avg_bps": sum(buy_slips) / len(buy_slips) if buy_slips else None,
            "sell_avg_bps": sum(sell_slips) / len(sell_slips) if sell_slips else None,
            "buy_count": len(buy_slips),
            "sell_count": len(sell_slips),
        }
