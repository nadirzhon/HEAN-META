"""Trade gating logic for small capital mode.

Combines cost estimation, edge calculation, and market filters
to make go/no-go decisions on trades. Every decision is logged
and emitted as an ORDER_DECISION event.
"""

from datetime import datetime, timezone
from typing import Any, Literal

from hean.config import settings
from hean.core.regime import Regime
from hean.core.types import Signal, Tick
from hean.execution.cost_engine import CostEngine
from hean.execution.edge_estimator import ExecutionEdgeEstimator
from hean.execution.market_filters import MarketFilters
from hean.logging import get_logger

logger = get_logger(__name__)


class TradeGating:
    """Trade gating and decision tracking for small capital mode."""

    def __init__(
        self,
        cost_engine: CostEngine | None = None,
        edge_estimator: ExecutionEdgeEstimator | None = None,
        market_filters: MarketFilters | None = None,
    ) -> None:
        """Initialize trade gating.

        Args:
            cost_engine: Cost estimation engine (creates if None)
            edge_estimator: Edge estimator (creates if None)
            market_filters: Market filters (creates if None)
        """
        self._cost_engine = cost_engine or CostEngine()
        self._edge_estimator = edge_estimator or ExecutionEdgeEstimator()
        self._market_filters = market_filters or MarketFilters()

        # Decision tracking (rolling 5min window)
        self._decisions_create = 0
        self._decisions_skip = 0
        self._decisions_block = 0

        # Reason code tracking
        self._block_reasons: dict[str, int] = {}

    def evaluate_trade_decision(
        self,
        signal: Signal,
        tick: Tick,
        regime: Regime,
        order_type: Literal["maker", "taker"] = "maker",
        volatility_proxy: float = 0.0,
    ) -> dict[str, Any]:
        """Evaluate whether to allow a trade.

        Returns ORDER_DECISION payload with:
        - decision: "CREATE" | "SKIP" | "BLOCK"
        - reason_codes: List of reasons for SKIP/BLOCK
        - expected_edge_bps: Estimated edge
        - cost_total_bps: Estimated total cost
        - cost_breakdown: Cost component breakdown
        - edge_cost_ratio: Edge / Cost ratio
        - maker_or_taker: Planned execution type
        """
        decision_time = datetime.now(timezone.utc)
        reason_codes = []

        # If not in small capital mode, use legacy edge estimator only
        if not settings.small_capital_mode:
            legacy_pass = self._edge_estimator.should_emit_signal(signal, tick, regime)
            if legacy_pass:
                self._decisions_create += 1
                return {
                    "decision": "CREATE",
                    "symbol": signal.symbol,
                    "strategy_id": signal.strategy_id,
                    "expected_edge_bps": self._edge_estimator.estimate_edge(signal, tick, regime),
                    "cost_total_bps": 0.0,
                    "cost_breakdown": {},
                    "edge_cost_ratio": 0.0,
                    "maker_or_taker": order_type,
                    "reason_codes": [],
                    "regime": regime.value,
                    "timestamp": decision_time.isoformat(),
                }
            else:
                self._decisions_skip += 1
                return {
                    "decision": "SKIP",
                    "symbol": signal.symbol,
                    "strategy_id": signal.strategy_id,
                    "expected_edge_bps": 0.0,
                    "cost_total_bps": 0.0,
                    "cost_breakdown": {},
                    "edge_cost_ratio": 0.0,
                    "maker_or_taker": order_type,
                    "reason_codes": ["LEGACY_EDGE_FILTER"],
                    "regime": regime.value,
                    "timestamp": decision_time.isoformat(),
                }

        # SMALL CAPITAL MODE ACTIVE
        # Step 1: Check market filters (stale, liquidity)
        market_blocked, market_reasons = self._market_filters.check_all_filters(
            symbol=signal.symbol,
            tick=tick,
            current_time=decision_time,
        )

        if market_blocked:
            for reason in market_reasons:
                self._increment_block_reason(reason)
            reason_codes.extend(market_reasons)
            self._decisions_block += 1

            return {
                "decision": "BLOCK",
                "symbol": signal.symbol,
                "strategy_id": signal.strategy_id,
                "expected_edge_bps": 0.0,
                "cost_total_bps": 0.0,
                "cost_breakdown": {},
                "edge_cost_ratio": 0.0,
                "maker_or_taker": order_type,
                "reason_codes": reason_codes,
                "regime": regime.value,
                "timestamp": decision_time.isoformat(),
            }

        # Step 2: Estimate edge
        expected_edge_bps = self._edge_estimator.estimate_edge(signal, tick, regime)

        # Step 3: Estimate cost
        cost_result = self._cost_engine.estimate_total_cost_bps(
            symbol=signal.symbol,
            side=signal.side,
            qty=signal.quantity,
            order_type=order_type,
            tick=tick,
            volatility_proxy=volatility_proxy,
        )

        cost_total_bps = cost_result["total_cost_bps"]
        cost_breakdown = cost_result

        # Step 4: Check minimum notional
        notional_blocked, notional_reason = self._cost_engine.should_block_by_min_notional(
            qty=signal.quantity,
            price=signal.entry_price,
        )
        if notional_blocked:
            reason_codes.append(notional_reason)
            self._increment_block_reason(notional_reason)

        # Step 5: Cost vs Edge gating
        required_edge_multiplier = settings.cost_edge_multiplier
        if order_type == "taker" and settings.allow_taker_if_edge_strong:
            required_edge_multiplier = settings.taker_edge_multiplier

        required_edge_bps = cost_total_bps * required_edge_multiplier
        edge_cost_ratio = expected_edge_bps / cost_total_bps if cost_total_bps > 0 else 0.0

        if expected_edge_bps < required_edge_bps:
            reason_codes.append("EDGE_TOO_LOW_FOR_COST")
            self._increment_block_reason("EDGE_TOO_LOW_FOR_COST")

        # Step 6: Maker-only enforcement
        if settings.maker_only_default and order_type == "taker":
            if not settings.allow_taker_if_edge_strong:
                reason_codes.append("MAKER_ONLY_ENFORCED")
                self._increment_block_reason("MAKER_ONLY_ENFORCED")
            elif expected_edge_bps < cost_total_bps * settings.taker_edge_multiplier:
                reason_codes.append("MAKER_ONLY_ENFORCED")
                self._increment_block_reason("MAKER_ONLY_ENFORCED")

        # Final decision
        if len(reason_codes) > 0:
            self._decisions_block += 1
            decision = "BLOCK"
        else:
            self._decisions_create += 1
            decision = "CREATE"

        result = {
            "decision": decision,
            "symbol": signal.symbol,
            "strategy_id": signal.strategy_id,
            "expected_edge_bps": expected_edge_bps,
            "cost_total_bps": cost_total_bps,
            "cost_breakdown": cost_breakdown,
            "edge_cost_ratio": edge_cost_ratio,
            "maker_or_taker": order_type,
            "reason_codes": reason_codes,
            "regime": regime.value,
            "timestamp": decision_time.isoformat(),
            "required_edge_bps": required_edge_bps,
            "required_multiplier": required_edge_multiplier,
        }

        logger.debug(
            f"Trade decision for {signal.symbol}: {decision} "
            f"(edge={expected_edge_bps:.1f} bps, cost={cost_total_bps:.1f} bps, "
            f"ratio={edge_cost_ratio:.2f}x, reasons={reason_codes})"
        )

        return result

    def _increment_block_reason(self, reason: str) -> None:
        """Track block reason counts."""
        self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1

    def get_decision_counts(self) -> dict[str, int]:
        """Get decision counts (CREATE/SKIP/BLOCK)."""
        return {
            "create": self._decisions_create,
            "skip": self._decisions_skip,
            "block": self._decisions_block,
            "total": self._decisions_create + self._decisions_skip + self._decisions_block,
        }

    def get_top_block_reasons(self, top_n: int = 5) -> list[dict[str, any]]:
        """Get top N block reasons.

        Returns:
            List of {reason, count} sorted by count descending
        """
        items = [
            {"reason": reason, "count": count}
            for reason, count in self._block_reasons.items()
        ]
        items.sort(key=lambda x: x["count"], reverse=True)
        return items[:top_n]

    def get_metrics(self) -> dict[str, Any]:
        """Get trade gating metrics."""
        decision_counts = self.get_decision_counts()
        return {
            "decisions": decision_counts,
            "top_block_reasons": self.get_top_block_reasons(top_n=5),
            "cost_metrics": self._cost_engine.get_metrics(),
            "edge_metrics": self._edge_estimator.get_metrics(),
            "market_filter_metrics": self._market_filters.get_metrics(),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self._decisions_create = 0
        self._decisions_skip = 0
        self._decisions_block = 0
        self._block_reasons.clear()
        self._cost_engine.reset_metrics()
        self._edge_estimator.reset_metrics()
        self._market_filters.reset_metrics()
