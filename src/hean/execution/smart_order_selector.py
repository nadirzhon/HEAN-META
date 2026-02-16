"""Smart Order Type Selector.

Decides between limit (maker rebate) vs market (taker fee) vs skip based on:
- Edge after fees
- Signal urgency
- Slippage estimates

Bybit fee model:
- Taker: 0.055% (5.5 bps)
- Maker: -0.01% (rebate, -1 bp)
"""

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.execution.slippage_estimator import SlippageEstimator
from hean.logging import get_logger

logger = get_logger(__name__)


class SmartOrderSelector:
    """Select optimal order type based on fees, slippage, and urgency."""

    # Bybit fee structure (in bps)
    TAKER_FEE_BPS = 5.5
    MAKER_REBATE_BPS = -1.0  # Negative = rebate

    def __init__(
        self,
        bus: EventBus,
        slippage_estimator: SlippageEstimator,
        enabled: bool = True,
    ) -> None:
        """Initialize smart order selector.

        Args:
            bus: Event bus for subscribing to TICK events
            slippage_estimator: Slippage estimator instance
            enabled: Whether smart selection is enabled
        """
        self._bus = bus
        self._slippage_estimator = slippage_estimator
        self._enabled = enabled
        self._running = False

        # Bid/ask cache per symbol
        self._bid_ask_cache: dict[str, tuple[float, float]] = {}

        # Stats
        self._stats = {
            "decisions_made": 0,
            "limit_selected": 0,
            "market_selected": 0,
            "skip_selected": 0,
        }

    async def start(self) -> None:
        """Start the smart order selector."""
        if not self._enabled:
            logger.info("SmartOrderSelector disabled (SMART_ORDER_SELECTION_ENABLED=false)")
            return

        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info("SmartOrderSelector started")

    async def stop(self) -> None:
        """Stop the smart order selector."""
        if not self._enabled:
            return

        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("SmartOrderSelector stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Update bid/ask cache from TICK events."""
        if not self._running:
            return

        tick = event.data.get("tick")
        if not tick:
            return

        symbol = tick.symbol
        bid = tick.bid
        ask = tick.ask

        if bid and ask:
            self._bid_ask_cache[symbol] = (bid, ask)

    def select_order_type(
        self,
        signal_side: str,
        symbol: str,
        size: float,
        current_bid: float | None = None,
        current_ask: float | None = None,
        urgency: float = 0.5,
        orderbook_depth: list[tuple[float, float]] | None = None,
    ) -> dict:
        """Select order type and parameters.

        Args:
            signal_side: "buy" or "sell"
            symbol: Trading symbol
            size: Order size
            current_bid: Current best bid (if None, uses cached)
            current_ask: Current best ask (if None, uses cached)
            urgency: Signal urgency [0, 1] - higher = more urgent
            orderbook_depth: Orderbook levels for slippage estimation

        Returns:
            Dict with:
                - order_type: "limit", "market", or "skip"
                - price: Limit price (if order_type == "limit")
                - edge_bps: Edge in bps after costs
                - reason: Human-readable reason for decision
        """
        if not self._enabled:
            return {
                "order_type": "market",
                "price": None,
                "edge_bps": 0.0,
                "reason": "smart_selection_disabled",
            }

        self._stats["decisions_made"] += 1

        # Get bid/ask
        if current_bid is None or current_ask is None:
            cached = self._bid_ask_cache.get(symbol)
            if cached:
                current_bid, current_ask = cached
            else:
                # No price data - default to market
                logger.warning(
                    f"[SmartOrderSelector] No bid/ask for {symbol}, defaulting to market"
                )
                self._stats["market_selected"] += 1
                return {
                    "order_type": "market",
                    "price": None,
                    "edge_bps": 0.0,
                    "reason": "no_price_data",
                }

        # Calculate mid price
        mid = (current_bid + current_ask) / 2

        # Estimate slippage
        slippage_bps = self._slippage_estimator.estimate_slippage(
            symbol, signal_side, size, orderbook_depth
        )

        # Calculate edge for each option
        if signal_side.lower() == "buy":
            # Buy limit at bid (maker)
            limit_price = current_bid
            limit_edge_bps = (mid - limit_price) / mid * 10000 + abs(self.MAKER_REBATE_BPS)

            # Buy market at ask (taker)
            market_price = current_ask
            market_edge_bps = (mid - market_price) / mid * 10000 - self.TAKER_FEE_BPS - slippage_bps
        else:
            # Sell limit at ask (maker)
            limit_price = current_ask
            limit_edge_bps = (limit_price - mid) / mid * 10000 + abs(self.MAKER_REBATE_BPS)

            # Sell market at bid (taker)
            market_price = current_bid
            market_edge_bps = (market_price - mid) / mid * 10000 - self.TAKER_FEE_BPS - slippage_bps

        # Decision logic
        # 1. If urgency is high (>= 0.7), prefer market execution
        if urgency >= 0.7:
            if market_edge_bps > 0.5:
                self._stats["market_selected"] += 1
                return {
                    "order_type": "market",
                    "price": None,
                    "edge_bps": market_edge_bps,
                    "reason": f"high_urgency_{urgency:.2f}",
                }
            else:
                # High urgency but negative edge - skip
                self._stats["skip_selected"] += 1
                return {
                    "order_type": "skip",
                    "price": None,
                    "edge_bps": market_edge_bps,
                    "reason": f"high_urgency_but_negative_edge_{market_edge_bps:.2f}bps",
                }

        # 2. If limit edge is good (> 2 bps), use limit
        if limit_edge_bps > 2.0:
            self._stats["limit_selected"] += 1
            return {
                "order_type": "limit",
                "price": limit_price,
                "edge_bps": limit_edge_bps,
                "reason": f"maker_edge_{limit_edge_bps:.2f}bps",
            }

        # 3. If market edge is positive (> 0.5 bps), use market
        if market_edge_bps > 0.5:
            self._stats["market_selected"] += 1
            return {
                "order_type": "market",
                "price": None,
                "edge_bps": market_edge_bps,
                "reason": f"taker_edge_{market_edge_bps:.2f}bps",
            }

        # 4. Insufficient edge - skip
        self._stats["skip_selected"] += 1
        logger.info(
            f"[SmartOrderSelector] {symbol} {signal_side}: Insufficient edge. "
            f"limit={limit_edge_bps:.2f}bps, market={market_edge_bps:.2f}bps - SKIP"
        )
        return {
            "order_type": "skip",
            "price": None,
            "edge_bps": max(limit_edge_bps, market_edge_bps),
            "reason": f"insufficient_edge_limit{limit_edge_bps:.2f}_market{market_edge_bps:.2f}",
        }

    def get_stats(self) -> dict:
        """Get selection statistics."""
        total = self._stats["decisions_made"]
        if total == 0:
            return {**self._stats, "limit_pct": 0, "market_pct": 0, "skip_pct": 0}

        return {
            **self._stats,
            "limit_pct": self._stats["limit_selected"] / total * 100,
            "market_pct": self._stats["market_selected"] / total * 100,
            "skip_pct": self._stats["skip_selected"] / total * 100,
        }
