"""Main entrypoint for HEAN system."""

import argparse
import asyncio
import random
import signal
import sys
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from hean.agent_generation.capital_optimizer import CapitalOptimizer
from hean.agent_generation.catalyst import ImprovementCatalyst
from hean.agent_generation.report_generator import ReportGenerator
from hean.backtest.event_sim import EventSimulator
from hean.backtest.metrics import BacktestMetrics
from hean.config import settings
from hean.core.arb.triangular_scanner import TriangularScanner
from hean.core.bus import EventBus
from hean.core.clock import Clock
from hean.core.intelligence.causal_inference_engine import CausalInferenceEngine
from hean.core.intelligence.correlation_engine import CorrelationEngine
from hean.core.intelligence.meta_learning_engine import MetaLearningEngine
from hean.core.intelligence.multimodal_swarm import MultimodalSwarm
from hean.core.multi_symbol_scanner import MultiSymbolScanner
from hean.core.regime import Regime, RegimeDetector
from hean.core.timeframes import CandleAggregator
from hean.core.trade_density import trade_density
from hean.core.types import (
    Event,
    EventType,
    Order,
    OrderRequest,
    OrderStatus,
    Position,
    Signal,
    Tick,
)
from hean.evaluation.truth_layer import analyze_truth, print_truth
from hean.exchange.models import PriceFeed
from hean.execution.order_manager import OrderManager
from hean.execution.position_monitor import PositionMonitor
from hean.execution.position_reconciliation import PositionReconciler
from hean.execution.router import ExecutionRouter
from hean.income.streams import (
    BasisHedgeStream,
    FundingHarvesterStream,
    MakerRebateStream,
    VolatilityHarvestStream,
)
from hean.logging import get_logger, setup_logging
from hean.observability.health import HealthCheck
from hean.observability.metrics import metrics
from hean.observability.monitoring.self_healing import SelfHealingMiddleware
from hean.observability.no_trade_report import no_trade_report
from hean.paper_trade_assist import (
    is_paper_assist_enabled,
    log_allow_reason,
    log_block_reason,
)
from hean.portfolio.accounting import PortfolioAccounting
from hean.portfolio.allocator import CapitalAllocator
from hean.portfolio.decision_memory import DecisionMemory
from hean.portfolio.profit_capture import ProfitCapture
from hean.portfolio.profit_target_tracker import ProfitTargetTracker
from hean.portfolio.smart_reinvestor import SmartReinvestor
from hean.risk.capital_preservation import CapitalPreservationMode
from hean.risk.deposit_protector import DepositProtector
from hean.risk.killswitch import KillSwitch
from hean.risk.limits import RiskLimits
from hean.risk.multi_level_protection import MultiLevelProtection
from hean.risk.position_sizer import PositionSizer
from hean.risk.tail_risk import GlobalSafetyNet
from hean.strategies.base import BaseStrategy
from hean.strategies.basis_arbitrage import BasisArbitrage
from hean.strategies.correlation_arb import CorrelationArbitrage
from hean.strategies.enhanced_grid import EnhancedGridStrategy
from hean.strategies.funding_harvester import FundingHarvester
from hean.strategies.hf_scalping import HFScalpingStrategy
from hean.strategies.impulse_engine import ImpulseEngine
from hean.strategies.inventory_neutral_mm import InventoryNeutralMM
from hean.strategies.liquidity_sweep import LiquiditySweepDetector
from hean.strategies.momentum_trader import MomentumTrader
from hean.strategies.rebate_farmer import RebateFarmer
from hean.strategies.sentiment_strategy import SentimentStrategy

logger = get_logger(__name__)


class TradingSystem:
    """Main trading system orchestrator."""

    def __init__(
        self,
        mode: Literal["run", "evaluate"] = "run",
        bus: EventBus | None = None,
    ) -> None:
        """Initialize the trading system.

        Args:
            mode: Operation mode. "run" for live/paper trading, "evaluate" for evaluation.
                  In evaluate mode, HealthCheck is disabled and periodic status is skipped.
            bus: Optional shared EventBus instance for external subscribers.
        """
        self._mode = mode
        self._bus = bus or EventBus()
        self._clock = Clock()

        # Use backtest capital in evaluate mode
        initial_capital = (
            settings.backtest_initial_capital if mode == "evaluate" else settings.initial_capital
        )
        self._accounting = PortfolioAccounting(initial_capital)

        # Deposit protection and profit tracking
        self._deposit_protector = DepositProtector(self._bus, initial_capital)
        self._profit_tracker = ProfitTargetTracker()
        self._profit_capture = ProfitCapture(self._bus, initial_capital)

        self._allocator = CapitalAllocator()
        self._decision_memory = DecisionMemory()
        self._risk_limits = RiskLimits()
        self._position_sizer = PositionSizer()
        self._killswitch = KillSwitch(self._bus)
        self._order_manager = OrderManager()
        self._regime_detector = RegimeDetector(self._bus)
        self._execution_router = ExecutionRouter(
            self._bus, self._order_manager, self._regime_detector
        )
        # Position Monitor - force-closes stale positions
        self._position_monitor = PositionMonitor(self._bus, self._accounting)
        # Position Reconciler - ensures local state matches exchange (initialized in run())
        self._position_reconciler: PositionReconciler | None = None
        self._multi_symbol_scanner = MultiSymbolScanner(self._bus)
        self._price_feed: PriceFeed | None = None
        self._strategies: list = []
        self._income_streams: list = []
        self._candle_aggregator: CandleAggregator | None = None

        # Context aggregator (fuses Brain+Physics+TCN+OFI+Causal → CONTEXT_READY)
        self._context_aggregator = None

        # Auto-improvement systems
        self._improvement_catalyst: ImprovementCatalyst | None = None
        self._ai_factory = None
        self._capital_optimizer = CapitalOptimizer()
        self._report_generator = ReportGenerator()

        # AI Council (multi-model periodic system review)
        self._council = None

        # Phase 5: Statistical Arbitrage & Anti-Fragile Architecture
        self._correlation_engine: CorrelationEngine | None = None
        self._safety_net: GlobalSafetyNet | None = None
        self._self_healing: SelfHealingMiddleware | None = None

        # Absolute+: Post-Singularity Systems (Market-Architecting Entity)
        self._meta_learning_engine: MetaLearningEngine | None = None
        self._causal_inference_engine: CausalInferenceEngine | None = None
        self._multimodal_swarm: MultimodalSwarm | None = None

        # Skip HealthCheck in evaluate mode
        self._health_check = HealthCheck() if mode == "run" else None

        self._running = False
        self._stop_trading = False
        self._current_regime: dict[str, Regime] = {}

        # Paper trade assist: fallback micro-trade tracking
        self._last_micro_trade_time: dict[str, datetime] = {}
        self._micro_trade_task: asyncio.Task[None] | None = None
        self._impulse_engine: ImpulseEngine | None = None
        self._triangular_scanner: TriangularScanner | None = None

        # Physics engine components
        self._physics_engine = None
        self._participant_classifier = None
        self._anomaly_detector = None
        self._temporal_stack = None
        self._cross_market = None

        # Debug metrics for forced order flow
        self._signals_generated = 0
        self._signals_after_filters = 0
        self._orders_sent = 0
        self._orders_filled = 0
        self._order_decision_history: list[dict[str, Any]] = []
        self._order_exit_decision_history: list[dict[str, Any]] = []

        # Exit management / heartbeat tracking
        self._last_exit_decision_ts: dict[str, datetime] = {}
        self._last_tick_at: dict[str, datetime] = {}

    async def _emit_order_decision(
        self,
        signal: Signal,
        decision: str,
        reason_code: str,
        computed_qty: float | None,
        context: dict[str, Any],
    ) -> None:
        """Emit structured order decision telemetry for observability.

        Includes gating flags, reason_codes array, market_regime, and advisory fields per AFO-Director spec.
        """
        # Collect gating flags for diagnostics
        equity = self._accounting.get_equity()
        drawdown_amount, drawdown_pct = self._accounting.get_drawdown(equity)
        open_positions = len(self._accounting.get_positions())
        open_orders = len(self._order_manager.get_open_orders())

        # Build reason_codes array (reason_code + any additional reasons from context)
        reason_codes = [reason_code]
        if "additional_reasons" in context:
            reason_codes.extend(context["additional_reasons"])

        # Get current regime for symbol
        market_regime = None
        market_metrics_short = {}
        if hasattr(self, "_regime_detector"):
            current_regime = self._regime_detector.get_regime(signal.symbol) if hasattr(self._regime_detector, "get_regime") else None
            if current_regime:
                # Map existing regime to spec regime types
                regime_map = {
                    "impulse": "TREND",
                    "normal": "RANGE",
                    "range": "RANGE",
                }
                market_regime = regime_map.get(current_regime.value if hasattr(current_regime, "value") else str(current_regime), "RANGE")

        # Check for stale data
        last_tick_age_sec = None
        if hasattr(self, "_last_tick_at") and signal.symbol in self._last_tick_at:
            last_tick_age_sec = (datetime.utcnow() - self._last_tick_at[signal.symbol]).total_seconds()
            if last_tick_age_sec > 30:
                market_regime = "STALE_DATA"
                reason_codes.append("STALE_MARKET_DATA")

        # Check liquidity (simple heuristic: if spread is too wide, mark as LOW_LIQ)
        if hasattr(self, "_execution_router") and hasattr(self._execution_router, "_current_bids") and hasattr(self._execution_router, "_current_asks"):
            bid = self._execution_router._current_bids.get(signal.symbol)
            ask = self._execution_router._current_asks.get(signal.symbol)
            if bid and ask and bid > 0:
                spread_pct = ((ask - bid) / bid) * 100
                market_metrics_short["spread_pct"] = spread_pct
                if spread_pct > 0.5:  # 0.5% spread threshold
                    if market_regime != "STALE_DATA":
                        market_regime = "LOW_LIQ"
                    reason_codes.append("LOW_LIQUIDITY")

        # Build comprehensive gating flags
        gating_flags = {
            "risk_ok": not self._stop_trading and not (self._killswitch.is_triggered() if hasattr(self, "_killswitch") else False),
            "data_fresh_ok": last_tick_age_sec is None or last_tick_age_sec < 30,
            "profit_lock_ok": True,  # Will be set by profit capture in B2
            "engine_running_ok": not self._stop_trading,
            "symbol_enabled_ok": signal.symbol in (settings.trading_symbols if hasattr(settings, "trading_symbols") else []),
            "liquidity_ok": market_regime != "LOW_LIQ",
            "execution_ok": True,  # Will be enhanced by execution quality checks
            "stop_trading": self._stop_trading,
            "killswitch_triggered": self._killswitch.is_triggered() if hasattr(self, "_killswitch") else False,
            "max_positions": open_positions >= settings.max_open_positions,
            "max_orders": open_orders >= settings.max_open_orders,
            "drawdown_pct": drawdown_pct,
            "equity": equity,
        }

        # Add killswitch reason if triggered
        if gating_flags["killswitch_triggered"] and hasattr(self, "_killswitch"):
            gating_flags["killswitch_reason"] = self._killswitch.get_reason()

        # Build advisory (how to continue)
        advisory = None
        if decision in ("SKIP", "BLOCK"):
            hints = []
            how_to_continue = None

            if not gating_flags["risk_ok"]:
                if gating_flags["killswitch_triggered"]:
                    how_to_continue = "check_killswitch"
                    hints.append("Review killswitch reasons and thresholds")
                elif self._stop_trading:
                    how_to_continue = "resume"
                    hints.append("Resume trading if conditions are safe")
            elif not gating_flags["data_fresh_ok"]:
                how_to_continue = "check_data_feed"
                hints.append("Verify market data connection")
            elif not gating_flags["liquidity_ok"]:
                how_to_continue = "wait_for_liquidity"
                hints.append("Wait for better liquidity conditions")
            elif gating_flags["max_positions"] or gating_flags["max_orders"]:
                how_to_continue = "reduce_positions"
                hints.append("Close some positions or orders to free capacity")
            elif drawdown_pct > 10:
                how_to_continue = "reduce_risk"
                hints.append("Reduce risk exposure due to drawdown")
            else:
                how_to_continue = "resume"
                hints.append("System should resume automatically")

            advisory = {
                "how_to_continue": how_to_continue,
                "hints": hints,
            }

        # Get score/confidence from context if available
        score = context.get("score") or context.get("confidence")

        payload = {
            "type": "ORDER_DECISION",
            "decision": decision,  # CREATE|SKIP|BLOCK
            "reason_codes": reason_codes,  # Array of reason codes
            "reason_code": reason_code,  # Keep for backward compatibility
            "signal_id": signal.strategy_id + ":" + signal.symbol + ":" + signal.side,
            "strategy_id": signal.strategy_id,
            "symbol": signal.symbol,
            "side": signal.side,
            "timestamp": datetime.utcnow().isoformat(),
            "computed_qty": computed_qty,
            "gating_flags": gating_flags,
            "score": score,  # nullable
            "confidence": score,  # nullable, alias
            "market_regime": market_regime,  # TREND|RANGE|LOW_LIQ|STALE_DATA
            "market_metrics_short": market_metrics_short,  # spread_pct, etc.
            "advisory": advisory,  # nullable
            **{k: v for k, v in context.items() if k not in ("additional_reasons", "score", "confidence")},
        }
        # Keep short history for diagnostics
        self._order_decision_history.append(payload)
        self._order_decision_history = self._order_decision_history[-100:]
        logger.info(f"[ORDER_DECISION] {payload}")
        try:
            await self._bus.publish(
                Event(event_type=EventType.ORDER_DECISION, data=payload)
            )
        except Exception as e:
            logger.debug(f"Failed to publish ORDER_DECISION event: {e}")

    async def _emit_order_exit_decision(
        self,
        position: Position,
        decision: str,
        reason_code: str,
        tick_price: float | None,
        thresholds: dict[str, Any],
        hold_seconds: float | None = None,
        note: str | None = None,
    ) -> None:
        """Emit structured exit decision telemetry (TP/SL/TTL/HOLD)."""
        payload = {
            "type": "ORDER_EXIT_DECISION",
            "decision": decision,
            "reason_code": reason_code,
            "position_id": position.position_id,
            "order_id": position.metadata.get("entry_order_id") if position.metadata else None,
            "symbol": position.symbol,
            "side": position.side,
            "timestamp": datetime.utcnow().isoformat(),
            "last_price": tick_price,
            "entry_price": position.entry_price,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
            "qty": position.size,
            "strategy_id": position.strategy_id,
            "hold_seconds": hold_seconds,
            "thresholds": thresholds,
            "metadata": position.metadata or {},
        }
        if note:
            payload["note"] = note

        self._order_exit_decision_history.append(payload)
        self._order_exit_decision_history = self._order_exit_decision_history[-200:]
        self._last_exit_decision_ts[position.position_id] = datetime.utcnow()
        logger.info(f"[ORDER_EXIT_DECISION] {payload}")
        try:
            await self._bus.publish(
                Event(event_type=EventType.ORDER_EXIT_DECISION, data=payload)
            )
        except Exception as e:
            logger.debug(f"Failed to publish ORDER_EXIT_DECISION event: {e}")

    async def start(self, price_feed=None) -> None:
        """Start the trading system.

        Args:
            price_feed: Optional PriceFeed instance to inject. If provided, uses this
                       instead of creating BybitPriceFeed. Used for evaluation mode
                       with EventSimulator.
        """
        logger.info("Starting HEAN trading system...")
        logger.info(f"Trading mode: {settings.trading_mode}")
        logger.info(f"DRY_RUN: {settings.dry_run}")
        logger.info(f"bybit_testnet: {settings.bybit_testnet}")
        logger.info(f"PAPER_TRADE_ASSIST: {settings.paper_trade_assist}")
        logger.info(f"LIVE_CONFIRM: {settings.live_confirm}")

        # Use initial capital from config (balance sync disabled to prevent hangs)
        initial_capital = (
            settings.backtest_initial_capital
            if self._mode == "evaluate"
            else settings.initial_capital
        )
        logger.info(f"Using configured capital: ${initial_capital:.2f}")

        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Profit target: ${settings.profit_target_daily_usd:.2f}/day")
        logger.info(
            f"Deposit protection: {'ENABLED' if settings.deposit_protection_active else 'DISABLED'}"
        )
        logger.info(f"DRY_RUN: {settings.dry_run}")
        logger.info(f"PROCESS_FACTORY_ENABLED: {settings.process_factory_enabled}")
        logger.info(f"PROCESS_FACTORY_ALLOW_ACTIONS: {settings.process_factory_allow_actions}")
        logger.info(f"EXECUTION_SMOKE_TEST_ENABLED: {settings.execution_smoke_test_enabled}")

        # Start core components
        await self._bus.start()
        await self._clock.start()
        await self._execution_router.start()
        # Start multi-timeframe candle aggregation
        self._candle_aggregator = CandleAggregator(self._bus)
        await self._candle_aggregator.start()

        # Start Position Monitor - force-closes stale positions
        await self._position_monitor.start()
        logger.info(f"Position Monitor started (max_hold={settings.max_hold_seconds}s)")

        # Start Position Reconciler - ensures local state matches exchange
        # Only in run mode (not evaluate mode) and if execution router has bybit_http
        if self._mode == "run" and hasattr(self._execution_router, "_bybit_http"):
            try:
                self._position_reconciler = PositionReconciler(
                    bus=self._bus,
                    bybit_http=self._execution_router._bybit_http,
                    accounting=self._accounting,
                )
                await self._position_reconciler.start()
                logger.info("Position Reconciler started (30s interval)")
            except Exception as e:
                logger.warning(f"Position Reconciler failed to start: {e}")

        # Only start HealthCheck in run mode
        if self._health_check:
            await self._health_check.start()

        # Start regime detector
        await self._regime_detector.start()
        self._bus.subscribe(EventType.REGIME_UPDATE, self._handle_regime_update)

        # Start multi-symbol scanner
        await self._multi_symbol_scanner.start()

        # Phase: Oracle Engine Integration (Algorithmic Fingerprinting + TCN)
        if self._mode == "run" and getattr(settings, 'oracle_engine_enabled', True):
            try:
                from hean.core.intelligence.oracle_integration import OracleIntegration
                self._oracle_integration = OracleIntegration(self._bus, symbols=settings.trading_symbols)
                await self._oracle_integration.start()
                logger.info("Oracle Engine Integration started (Fingerprinting + TCN)")
            except ModuleNotFoundError as e:
                logger.warning(f"Oracle Engine disabled (missing dependency): {e}")
            except Exception as e:
                logger.warning(f"Oracle Engine failed to start: {e}")

        # Phase 5: Initialize Statistical Arbitrage & Anti-Fragile Architecture
        if self._mode == "run" and settings.phase5_correlation_engine_enabled:
            # 1. Correlation Engine for pair trading
            self._correlation_engine = CorrelationEngine(self._bus, symbols=settings.trading_symbols)
            await self._correlation_engine.start()
            logger.info("Phase 5: Correlation Engine started")

        if self._mode == "run" and settings.phase5_safety_net_enabled:
            # 2. Global Safety Net (Black Swan Protection)
            self._safety_net = GlobalSafetyNet(
                bus=self._bus,
                regime_detector=self._regime_detector,
                accounting=self._accounting,
                position_sizer=self._position_sizer
            )
            await self._safety_net.start()
            logger.info("Phase 5: Global Safety Net activated")

        if self._mode == "run" and settings.phase5_self_healing_enabled:
            # 3. Self-Healing Middleware
            self._self_healing = SelfHealingMiddleware(
                bus=self._bus,
                order_manager=self._order_manager
            )
            await self._self_healing.start()
            logger.info("Phase 5: Self-Healing Middleware started")

        if self._mode == "run" and settings.phase5_kelly_criterion_enabled:
            # 4. Enable Kelly Criterion for Position Sizer
            try:
                if hasattr(self._position_sizer, 'enable_kelly_criterion'):
                    self._position_sizer.enable_kelly_criterion(
                        self._accounting,
                        fractional_kelly=settings.phase5_kelly_fractional
                    )
                    logger.info(f"Phase 5: Kelly Criterion enabled with fractional_kelly={settings.phase5_kelly_fractional}")
            except Exception as e:
                logger.warning(f"Could not enable Kelly Criterion: {e}")

        # Absolute+: Initialize Post-Singularity Systems
        if self._mode == "run" and getattr(settings, 'absolute_plus_enabled', True):
            # DISABLED: Dead code - MetaLearningEngine publishes META_LEARNING_PATCH but no handler exists
            # 1. Meta-Learning Engine (Recursive Intelligence Core)
            self._meta_learning_engine = None
            logger.info("MetaLearningEngine DISABLED — not connected to trading flow")

            # DISABLED: Dead code - CausalInferenceEngine output stored in ctx.causal but never used by strategies
            # 2. Causal Inference Engine
            self._causal_inference_engine = None
            logger.info("CausalInferenceEngine DISABLED — output not consumed by any strategy")

            # DISABLED: Dead code - MultimodalSwarm output not consumed by any strategy
            # 3. Multimodal Swarm (Unified Tensor Processing)
            self._multimodal_swarm = None
            logger.info("MultimodalSwarm DISABLED — no strategy consumes tensor output")

        # Physics Engine: Market thermodynamics
        if self._mode == "run":
            try:
                from hean.physics import (
                    PhysicsEngine,
                    ParticipantClassifier,
                    MarketAnomalyDetector,
                    TemporalStack,
                    CrossMarketImpulse,
                )

                self._physics_engine = PhysicsEngine(bus=self._bus)
                await self._physics_engine.start()

                self._participant_classifier = ParticipantClassifier(bus=self._bus)
                await self._participant_classifier.start()

                self._anomaly_detector = MarketAnomalyDetector()

                self._temporal_stack = TemporalStack(symbols=settings.trading_symbols)

                self._cross_market = CrossMarketImpulse(
                    leader_symbols=["BTCUSDT"],
                    follower_symbols=["ETHUSDT", "SOLUSDT"],
                )

                logger.info("Physics Engine started (Temperature/Entropy/Phase/Participants/Anomalies/TemporalStack)")
            except Exception as e:
                logger.warning(f"Physics Engine failed to start: {e}")

        # Brain: Claude AI market analysis
        self._brain_client = None
        if self._mode == "run" and getattr(settings, 'brain_enabled', True):
            try:
                from hean.brain.claude_client import ClaudeBrainClient

                self._brain_client = ClaudeBrainClient(
                    bus=self._bus,
                    api_key=getattr(settings, 'anthropic_api_key', ''),
                    analysis_interval=getattr(settings, 'brain_analysis_interval', 60),
                    openrouter_api_key=getattr(settings, 'openrouter_api_key', ''),
                )
                await self._brain_client.start()
                logger.info("Brain Client started")
            except Exception as e:
                logger.warning(f"Brain Client failed to start: {e}")

        # ContextAggregator: fuses Brain+Physics+TCN+OFI+Causal → CONTEXT_READY
        if self._mode == "run":
            try:
                from hean.core.context_aggregator import ContextAggregator

                symbols_for_context = settings.symbols if settings.multi_symbol_enabled else settings.trading_symbols
                self._context_aggregator = ContextAggregator(self._bus, symbols_for_context)
                await self._context_aggregator.start()
                logger.info("ContextAggregator started")
            except Exception as e:
                logger.warning(f"ContextAggregator failed to start: {e}")

        # AI Factory: Shadow → Canary → Production pipeline
        if self._mode == "run":
            try:
                from hean.ai.factory import AIFactory

                self._ai_factory = AIFactory(self._bus)
                logger.info("AI Factory initialized")
            except Exception as e:
                logger.warning(f"AI Factory failed to initialize: {e}")

        # DuckDB Storage (optional)
        self._duckdb_store = None
        if self._mode == "run":
            try:
                from hean.storage.duckdb_store import DuckDBStore

                self._duckdb_store = DuckDBStore(bus=self._bus)
                await self._duckdb_store.start()
                logger.info("DuckDB Storage started")
            except ImportError:
                logger.info("DuckDB not installed, storage disabled")
            except Exception as e:
                logger.warning(f"DuckDB Storage failed to start: {e}")

        # Start price feed
        # Use multi-symbol list if MULTI_SYMBOL_ENABLED, otherwise use trading_symbols
        if settings.multi_symbol_enabled:
            symbols = settings.symbols
            logger.info(f"Multi-Symbol Mode ENABLED: trading {len(symbols)} symbols: {symbols}")
        else:
            symbols = settings.trading_symbols
            logger.info(f"Single/Limited Symbol Mode: trading {len(symbols)} symbols: {symbols}")
        if price_feed is not None:
            # Use injected price feed (e.g., EventSimulator for evaluation)
            self._price_feed = price_feed
            # EventSimulator needs bus injected via start()
            if isinstance(price_feed, EventSimulator):
                await price_feed.start(bus=self._bus)
            else:
                await price_feed.start()
        elif settings.is_live or settings.paper_use_live_feed:
            # Use Bybit price feed for live or paper mode (public data)
            try:
                from hean.exchange.bybit.price_feed import BybitPriceFeed

                self._price_feed = BybitPriceFeed(self._bus, symbols)
                await self._price_feed.start()
                mode_label = "live" if settings.is_live else "paper"
                logger.info(f"MARKET_STREAM_STARTED: Bybit price feed ({mode_label}) for symbols {symbols}")

                # Auto-detect balance from exchange in live mode
                # System works with any amount - no need to specify INITIAL_CAPITAL
                try:
                    from hean.exchange.bybit.http import BybitHTTPClient
                    http_client = BybitHTTPClient()
                    await http_client.connect()
                    account_info = await http_client.get_account_info()
                    await http_client.disconnect()

                    # Parse balance from Bybit API response
                    # _request() returns result.get("result", {}) — already unwrapped
                    balance = 0.0
                    account_list = account_info.get("list", [])
                    if account_list:
                        coins = account_list[0].get("coin", [])
                        for coin in coins:
                            if coin.get("coin") == "USDT":
                                balance = float(coin.get("walletBalance", 0))
                                break

                    if balance > 0:
                        # Update accounting with real exchange balance
                        self._accounting.set_balance_from_exchange(balance)
                        # Update deposit protector with real balance
                        self._deposit_protector.update_initial_capital(balance)
                        # Update killswitch with real balance
                        self._killswitch.set_initial_capital(balance)
                        # Update profit capture so it doesn't see config vs exchange diff as gain
                        if self._profit_capture:
                            self._profit_capture.sync_start_equity(balance)
                        logger.info(f"✅ Synced with exchange balance: ${balance:,.2f} USDT")
                        logger.info("📊 System ready to trade with any capital amount")
                    else:
                        logger.warning(f"⚠️ Could not get balance from exchange, using config: ${initial_capital:,.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not sync with exchange balance: {e}")
                    logger.info(f"Using configured initial capital: ${initial_capital:,.2f}")
            except Exception as e:
                logger.critical(f"FATAL: Bybit price feed failed. Cannot start without real market data. Error: {e}", exc_info=True)
                self._stop_trading = True
                await self._bus.publish(
                    Event(
                        event_type=EventType.KILLSWITCH_TRIGGERED,
                        data={
                            "reason": "bybit_price_feed_failure",
                            "error": str(e),
                        },
                    )
                )
                raise RuntimeError(
                    f"Cannot start trading system: Bybit price feed failed. "
                    f"No synthetic/simulated data fallback allowed. Error: {e}"
                ) from e
        else:
            # No live feed configured — require Bybit connection
            raise RuntimeError(
                "Cannot start trading system without Bybit price feed. "
                "Set BYBIT_API_KEY/BYBIT_API_SECRET and ensure BYBIT_TESTNET=true. "
                "No synthetic/simulated data allowed in production."
            )

        # Start triangular arbitrage scanner (paper + live)
        if self._mode == "run" and settings.triangular_arb_enabled:
            try:
                self._triangular_scanner = TriangularScanner(
                    self._bus,
                    fee_buffer=settings.triangular_fee_buffer,
                    min_profit_bps=settings.triangular_min_profit_bps,
                )
                await self._triangular_scanner.start()
                logger.info("Triangular Arbitrage Scanner started")
            except Exception as e:
                logger.warning(f"Failed to start Triangular Arbitrage Scanner: {e}")

        # Killswitch auto-reset is enabled by default; keep testnet-friendly settings
        if settings.bybit_testnet and self._killswitch:
            self._killswitch.enable_auto_reset(True)
            logger.info("Killswitch configured for testnet: auto-reset enabled (15min cooldown, 10% recovery, max 10/day)")

        # Start strategies
        if settings.funding_harvester_enabled:
            strategy = FundingHarvester(self._bus, symbols)
            await strategy.start()
            self._strategies.append(strategy)

        if settings.basis_arbitrage_enabled:
            strategy = BasisArbitrage(self._bus, symbols)
            await strategy.start()
            self._strategies.append(strategy)

        if settings.impulse_engine_enabled:
            # Phase 1 Profit Doubling: Pass Oracle Engine and OFI Monitor to impulse_engine
            oracle_ref = getattr(self, '_oracle_integration', None)
            oracle_engine_obj = oracle_ref._oracle if oracle_ref else None

            # Initialize OFI Monitor if not already created
            if not hasattr(self, '_ofi_monitor'):
                from hean.core.ofi import OrderFlowImbalance
                self._ofi_monitor = OrderFlowImbalance(
                    self._bus,
                    lookback_window=20,
                    use_ml_prediction=True
                )
                await self._ofi_monitor.start()
                logger.info("OFI Monitor initialized for impulse_engine")

            strategy = ImpulseEngine(
                self._bus, symbols,
                oracle_engine=oracle_engine_obj,
                ofi_monitor=self._ofi_monitor
            )
            await strategy.start()
            self._strategies.append(strategy)
            self._impulse_engine = strategy  # Store reference for metrics
        else:
            self._impulse_engine = None

        # Phase 5: Register dormant strategies (AFO-Director dormant strategies)
        # HF Scalping - high frequency scalping for range/normal regimes
        if getattr(settings, 'hf_scalping_enabled', False):
            strategy = HFScalpingStrategy(self._bus, symbols)
            await strategy.start()
            self._strategies.append(strategy)
            logger.info("HF Scalping Strategy registered and started")

        # Enhanced Grid - grid trading for range-bound markets
        if getattr(settings, 'enhanced_grid_enabled', False):
            strategy = EnhancedGridStrategy(self._bus, symbols)
            await strategy.start()
            self._strategies.append(strategy)
            logger.info("Enhanced Grid Strategy registered and started")

        # Momentum Trader - momentum following strategy
        if getattr(settings, 'momentum_trader_enabled', False):
            strategy = MomentumTrader(self._bus, symbols)
            await strategy.start()
            self._strategies.append(strategy)
            logger.info("Momentum Trader Strategy registered and started")

        # Inventory Neutral Market Making
        if getattr(settings, 'inventory_neutral_mm_enabled', False):
            strategy = InventoryNeutralMM(
                self._bus,
                ofi_monitor=getattr(self, '_ofi_monitor', None),
                symbols=symbols,
            )
            await strategy.start()
            self._strategies.append(strategy)
            logger.info("Inventory Neutral MM Strategy registered and started")

        # Correlation Arbitrage
        if getattr(settings, 'correlation_arb_enabled', False):
            strategy = CorrelationArbitrage(self._bus)
            await strategy.start()
            self._strategies.append(strategy)
            logger.info("Correlation Arbitrage Strategy registered and started")

        # Rebate Farmer
        if getattr(settings, 'rebate_farmer_enabled', False):
            strategy = RebateFarmer(self._bus, symbols=symbols)
            await strategy.start()
            self._strategies.append(strategy)
            logger.info("Rebate Farmer Strategy registered and started")

        # Liquidity Sweep Detector
        if getattr(settings, 'liquidity_sweep_enabled', False):
            strategy = LiquiditySweepDetector(
                self._bus,
                symbols=symbols,
                ofi_monitor=getattr(self, '_ofi_monitor', None),
            )
            await strategy.start()
            self._strategies.append(strategy)
            logger.info("Liquidity Sweep Detector registered and started")

        # Sentiment Strategy
        if getattr(settings, 'sentiment_strategy_enabled', False):
            strategy = SentimentStrategy(self._bus, symbols=symbols)
            await strategy.start()
            self._strategies.append(strategy)
            logger.info("Sentiment Strategy registered and started")

        # Start income streams (infrastructure-level, paper-safe by default)
        if settings.stream_funding_enabled:
            stream = FundingHarvesterStream(self._bus, symbols)
            await stream.start()
            self._income_streams.append(stream)

        if settings.stream_maker_rebate_enabled:
            stream = MakerRebateStream(self._bus, symbols)
            await stream.start()
            self._income_streams.append(stream)

        if settings.stream_basis_enabled:
            stream = BasisHedgeStream(self._bus, symbols)
            await stream.start()
            self._income_streams.append(stream)

        if settings.stream_volatility_enabled:
            stream = VolatilityHarvestStream(self._bus, symbols)
            await stream.start()
            self._income_streams.append(stream)

        # Subscribe to events
        self._bus.subscribe(EventType.SIGNAL, self._handle_signal)
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.subscribe(EventType.STOP_TRADING, self._handle_stop_trading)
        self._bus.subscribe(EventType.KILLSWITCH_TRIGGERED, self._handle_killswitch)
        # Subscribe to ticks for TP/SL checks, TTL, and mark-to-market updates
        self._bus.subscribe(EventType.TICK, self._handle_tick_forced_exit)
        # Subscribe to POSITION_CLOSE_REQUEST from Oracle/other modules
        self._bus.subscribe(EventType.POSITION_CLOSE_REQUEST, self._handle_position_close_request)
        # DISABLED: META_LEARNING_PATCH subscription removed — engine disabled, handler never existed
        # self._bus.subscribe(EventType.META_LEARNING_PATCH, self._handle_meta_learning_patch)

        # Schedule periodic status updates (skip in evaluate mode)
        if self._mode == "run":
            self._clock.schedule_periodic(self._print_status, timedelta(seconds=10))
            # Anti-stuck: periodic TTL/NO_TICKS checks even if feed stalls
            self._clock.schedule_periodic(self._check_position_timeouts, timedelta(seconds=5))

            # Start fallback micro-trade task if paper assist enabled
            if is_paper_assist_enabled():
                self._micro_trade_task = asyncio.create_task(self._micro_trade_fallback_loop())
                logger.info("Paper Trade Assist: Fallback micro-trade loop started")

        # Start improvement catalyst (only in run mode)
        if self._mode == "run":
            try:
                # Create strategy dict for catalyst
                strategy_dict = {strategy.strategy_id: strategy for strategy in self._strategies}
                self._improvement_catalyst = ImprovementCatalyst(
                    bus=self._bus,
                    accounting=self._accounting,
                    strategies=strategy_dict,
                    ai_factory=self._ai_factory,
                    check_interval_minutes=30,
                    min_trades_for_analysis=10,
                )
                await self._improvement_catalyst.start()
                logger.info("Improvement Catalyst started")
            except Exception as e:
                logger.warning(f"Could not start Improvement Catalyst: {e}")

        # Start AI Council (multi-model periodic system review)
        if self._mode == "run" and settings.council_enabled:
            try:
                from hean.council.council import AICouncil

                strategy_dict = {s.strategy_id: s for s in self._strategies}
                self._council = AICouncil(
                    bus=self._bus,
                    accounting=self._accounting,
                    strategies=strategy_dict,
                    killswitch=self._killswitch,
                    ai_factory=self._ai_factory,
                    improvement_catalyst=self._improvement_catalyst,
                    review_interval=settings.council_review_interval,
                    auto_apply_safe=settings.council_auto_apply_safe,
                )
                await self._council.start()
                logger.info("AI Council started")
            except Exception as e:
                logger.warning(f"Could not start AI Council: {e}")

        self._running = True
        logger.info("Trading system started")

    async def stop(self) -> None:
        """Stop the trading system."""
        logger.info("Stopping trading system...")
        self._running = False

        # Stop AI Council
        if self._council:
            await self._council.stop()

        # Stop improvement catalyst
        if self._improvement_catalyst:
            await self._improvement_catalyst.stop()

        # Stop strategies
        for strategy in self._strategies:
            await strategy.stop()

        # Stop income streams
        for stream in self._income_streams:
            await stream.stop()

        # Stop price feed
        if self._price_feed:
            await self._price_feed.stop()
        if hasattr(self, "_bybit_ws_public") and self._bybit_ws_public:
            await self._bybit_ws_public.disconnect()

        # Stop triangular arbitrage scanner
        if self._triangular_scanner:
            await self._triangular_scanner.stop()

        # Stop candle aggregation
        if self._candle_aggregator:
            await self._candle_aggregator.stop()

        # Stop Position Monitor
        await self._position_monitor.stop()

        # Stop Position Reconciler
        if self._position_reconciler:
            await self._position_reconciler.stop()

        # Stop ContextAggregator
        if self._context_aggregator:
            await self._context_aggregator.stop()

        # Stop Brain Client
        if hasattr(self, '_brain_client') and self._brain_client:
            await self._brain_client.stop()

        # Stop DuckDB Storage
        if hasattr(self, '_duckdb_store') and self._duckdb_store:
            await self._duckdb_store.stop()

        # Stop Physics Engine components
        if self._physics_engine:
            await self._physics_engine.stop()
        if self._participant_classifier:
            await self._participant_classifier.stop()

        # Phase 5: Stop Statistical Arbitrage & Anti-Fragile Architecture
        if self._self_healing:
            await self._self_healing.stop()
        if self._safety_net:
            await self._safety_net.stop()
        if self._correlation_engine:
            await self._correlation_engine.stop()

        # Stop regime detector
        self._bus.unsubscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        await self._regime_detector.stop()

        # Stop micro-trade task
        if self._micro_trade_task:
            self._micro_trade_task.cancel()
            try:
                await self._micro_trade_task
            except asyncio.CancelledError:
                pass

        # Stop core components
        await self._execution_router.stop()
        if self._health_check:
            await self._health_check.stop()
        await self._clock.stop()
        await self._bus.stop()

        logger.info("Trading system stopped")

    async def _handle_signal(self, event: Event) -> None:
        """Handle trading signal from strategy."""
        if self._stop_trading:
            logger.debug("Trading stopped, ignoring signal")
            signal: Signal = event.data["signal"]
            await self._emit_order_decision(
                signal=signal,
                decision="SKIP",
                reason_code="TRADING_DISABLED",
                computed_qty=None,
                context={
                    "note": "stop_trading flag set",
                    "open_orders": len(self._order_manager.get_open_orders()),
                    "open_positions": len(self._accounting.get_positions()),
                },
            )
            return

        signal: Signal = event.data["signal"]
        logger.debug(f"Signal received: {signal.strategy_id} {signal.symbol} {signal.side}")

        # CRITICAL: Mandatory stop_loss validation
        # Signals without stop_loss cannot be properly sized and represent unbounded risk
        if signal.stop_loss is None:
            logger.warning(
                f"Signal REJECTED: missing stop_loss (strategy={signal.strategy_id}, "
                f"symbol={signal.symbol}, side={signal.side})"
            )
            no_trade_report.increment("missing_stop_loss", signal.symbol, signal.strategy_id)
            await self._emit_order_decision(
                signal=signal,
                decision="REJECT",
                reason_code="MISSING_STOP_LOSS",
                computed_qty=None,
                context={
                    "error": "Signals without stop_loss represent unbounded risk",
                    "fix": "Add stop_loss to signal before publishing",
                },
            )
            return

        # Track signal attempt for diagnostics

        # Hard guards against runaway state (anti-stuck)
        open_positions = len(self._accounting.get_positions())
        open_orders = len(self._order_manager.get_open_orders())
        if open_positions >= settings.max_open_positions or open_orders >= settings.max_open_orders:
            await self._emit_order_decision(
                signal=signal,
                decision="SKIP",
                reason_code="LIMIT_REACHED",
                computed_qty=None,
                context={
                    "note": "risk guard: max positions/orders reached",
                    "open_positions": open_positions,
                    "open_orders": open_orders,
                    "max_open_positions": settings.max_open_positions,
                    "max_open_orders": settings.max_open_orders,
                },
            )
            log_block_reason(
                reason_code="max_positions_orders",
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                reasons=["max_open_positions/orders reached"],
                suggested_fix=[
                    f"Close positions below {settings.max_open_positions}",
                    f"Reduce open orders below {settings.max_open_orders}",
                ],
            )
            return

        # Total exposure guard: block if total notional > max_exposure_multiplier * equity
        equity = self._accounting.get_equity()
        if equity > 0:
            total_notional = 0.0
            for pos in self._accounting.get_positions():
                pos_price = pos.current_price or pos.entry_price
                if pos_price:
                    total_notional += abs(pos.size) * pos_price
            max_notional = equity * settings.max_exposure_multiplier
            if total_notional >= max_notional:
                logger.warning(
                    f"Signal BLOCKED by exposure guard: total_notional=${total_notional:.2f} >= "
                    f"max=${max_notional:.2f} ({settings.max_exposure_multiplier}x equity=${equity:.2f})"
                )
                no_trade_report.increment("exposure_guard_block", signal.symbol, signal.strategy_id)
                await self._emit_order_decision(
                    signal=signal,
                    decision="REJECT",
                    reason_code="EXPOSURE_LIMIT",
                    computed_qty=None,
                    context={
                        "total_notional": total_notional,
                        "max_notional": max_notional,
                        "equity": equity,
                        "exposure_ratio": total_notional / equity,
                    },
                )
                return

        # WHY NOT TRADING: Check basic execution gates first
        if settings.process_factory_enabled:
            from hean.process_factory.integrations.trade_diagnostics import (
                check_dry_run,
                check_live_enabled,
                check_process_factory_actions,
                log_trade_blocked,
            )

            blocked_reasons = []
            suggested_fixes = []

            # Check live enabled
            live_ok, live_reasons, live_fixes = check_live_enabled()
            if not live_ok:
                blocked_reasons.extend(live_reasons)
                suggested_fixes.extend(live_fixes)

            # Check dry run
            dry_run_ok, dry_reasons, dry_fixes = check_dry_run()
            if not dry_run_ok:
                blocked_reasons.extend(dry_reasons)
                suggested_fixes.extend(dry_fixes)

            # Check process factory actions
            actions_ok, actions_reasons, actions_fixes = check_process_factory_actions()
            if not actions_ok:
                blocked_reasons.extend(actions_reasons)
                suggested_fixes.extend(actions_fixes)

            if blocked_reasons:
                log_trade_blocked(
                    symbol=signal.symbol,
                    strategy_id=signal.strategy_id,
                    reasons=blocked_reasons,
                    suggested_fix=suggested_fixes,
                )
                # Don't return here - continue with other checks for diagnostics

        # CRITICAL: Check deposit protection FIRST (before all other checks)
        if settings.deposit_protection_active:
            equity = self._accounting.get_equity()
            is_safe, reason = self._deposit_protector.check_equity(equity)
            if not is_safe:
                logger.warning(f"Signal blocked by deposit protector: {reason}")
                metrics.increment("signals_blocked_deposit_protection")
                no_trade_report.increment(
                    "deposit_protection_block", signal.symbol, signal.strategy_id
                )
                await self._emit_order_decision(
                    signal=signal,
                    decision="REJECT",
                    reason_code="DEPOSIT_PROTECTION",
                    computed_qty=None,
                    context={
                        "reason": reason,
                        "equity": equity,
                        "available_cash": self._accounting.get_cash_balance(),
                        "open_orders": len(self._order_manager.get_open_orders()),
                        "open_positions": len(self._accounting.get_positions()),
                    },
                )
                if settings.process_factory_enabled:
                    log_trade_blocked(
                        symbol=signal.symbol,
                        strategy_id=signal.strategy_id,
                        reasons=[f"deposit_protection: {reason}"],
                        suggested_fix=["Check account equity vs initial capital"],
                    )
                return

        # DEBUG: Track signal generation
        self._signals_generated += 1
        metrics.increment("signals_generated_total")

        # МНОЖЕСТВЕННАЯ ЗАЩИТА (always active — never bypass risk controls)
        protection_size_multiplier = 1.0
        equity = self._accounting.get_equity()
        initial_capital = self._accounting.initial_capital
        protection = MultiLevelProtection(initial_capital=initial_capital)

        allowed, reason = protection.check_all_protections(
            strategy_id=signal.strategy_id, equity=equity, initial_capital=initial_capital
        )

        if not allowed:
            logger.warning(f"Signal blocked by protection: {reason}")
            metrics.increment("signals_blocked_protection")
            no_trade_report.increment("protection_block", signal.symbol, signal.strategy_id)
            no_trade_report.increment_pipeline("signals_blocked_protection", signal.strategy_id)
            await self._emit_order_decision(
                signal=signal,
                decision="REJECT",
                reason_code="PROTECTION_BLOCK",
                computed_qty=None,
                context={
                    "reason": reason,
                    "equity": equity,
                    "initial_capital": initial_capital,
                    "open_orders": len(self._order_manager.get_open_orders()),
                    "open_positions": len(self._accounting.get_positions()),
                },
            )
            return

        # Get equity for later use
        equity = self._accounting.get_equity()

        # Capital Preservation Mode (always active)
        preservation_mode = CapitalPreservationMode()
        drawdown_pct = self._accounting.get_drawdown(equity)[1]
        strategy_metrics = self._accounting.get_strategy_metrics()
        rolling_pf = (
            strategy_metrics.get(signal.strategy_id, {}).get("profit_factor", 1.0)
            if strategy_metrics
            else 1.0
        )
        consecutive_losses = self._risk_limits.get_consecutive_losses(signal.strategy_id)

        if preservation_mode.should_activate(drawdown_pct, rolling_pf, consecutive_losses):
            if signal.metadata is None:
                signal.metadata = {}
            signal.metadata["preservation_mode"] = True
            signal.metadata["risk_multiplier"] = preservation_mode.get_risk_pct(1.0)
            logger.warning(
                f"Capital Preservation Mode active: drawdown={drawdown_pct:.1f}%, "
                f"PF={rolling_pf:.2f}, consecutive_losses={consecutive_losses}"
            )

        # Calculate position size BEFORE creating OrderRequest
        equity = self._accounting.get_equity()

        # Get current price for size calculation
        current_price = signal.entry_price or self._regime_detector.get_price(signal.symbol)
        if not current_price or current_price <= 0:
            logger.warning(f"No valid price for {signal.symbol}, skipping signal")
            await self._emit_order_decision(
                signal=signal,
                decision="SKIP",
                reason_code="VALIDATION_FAIL",
                computed_qty=None,
                context={
                    "reason": "missing_price",
                    "open_orders": len(self._order_manager.get_open_orders()),
                    "open_positions": len(self._accounting.get_positions()),
                },
            )
            return

        # Get regime first (needed for size calculation)
        regime = self._current_regime.get(signal.symbol, Regime.NORMAL)

        # Get metrics for dynamic risk scaling (needed for size calculation)
        rolling_pf = None
        recent_drawdown = None
        volatility_percentile = None
        edge_bps = None

        # Get rolling profit factor from strategy metrics
        strategy_metrics = self._accounting.get_strategy_metrics()
        if strategy_metrics:
            signal_metrics = strategy_metrics.get(signal.strategy_id)
            if signal_metrics:
                rolling_pf = signal_metrics.get("profit_factor", 1.0)
            else:
                # Fallback: aggregate PF across all strategies
                total_wins = 0.0
                total_losses = 0.0
                for m in strategy_metrics.values():
                    total_wins += m.get("wins", 0)
                    total_losses += m.get("losses", 0)
                if total_losses > 0:
                    rolling_pf = total_wins / total_losses
                elif total_wins > 0:
                    rolling_pf = total_wins
                else:
                    rolling_pf = 1.0
        else:
            rolling_pf = 1.0

        # Get recent drawdown
        _, drawdown_pct = self._accounting.get_drawdown(equity)
        recent_drawdown = drawdown_pct

        # Get volatility and calculate percentile
        current_volatility = self._regime_detector.get_volatility(signal.symbol)
        if current_volatility > 0:
            # Update volatility history in position sizer
            self._position_sizer.update_volatility(current_volatility)
            # Calculate percentile
            dynamic_risk = self._position_sizer.get_dynamic_risk_manager()
            volatility_percentile = dynamic_risk.calculate_volatility_percentile(current_volatility)

        # Get edge_bps from signal metadata if available
        if signal.metadata:
            edge_bps = signal.metadata.get("edge_bps")

        # Calculate base size
        base_size = signal.size or self._position_sizer.calculate_size(
            signal,
            equity,
            current_price,
            regime,
            rolling_pf=rolling_pf,
            recent_drawdown=recent_drawdown,
            volatility_percentile=volatility_percentile,
            edge_bps=edge_bps,
        )

        # Ensure minimum size
        if base_size <= 0:
            min_size_value = (equity * 0.001) / current_price  # 0.1% of equity
            absolute_min = 0.001
            base_size = max(min_size_value, absolute_min)
            logger.warning(f"Base size was 0, using minimum {base_size:.6f}")

        # Apply protection multiplier
        if protection_size_multiplier < 1.0:
            base_size *= protection_size_multiplier
            if base_size <= 0:
                min_size_value = (equity * 0.001) / current_price
                absolute_min = 0.001
                base_size = max(min_size_value, absolute_min)

        # Check risk limits with calculated size (always active)
        allowed, reason = self._risk_limits.check_order_request(
            OrderRequest(
                signal_id=str(uuid.uuid4()),
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                side=signal.side,
                size=base_size,
                price=signal.entry_price,
            ),
            equity,
        )

        if not allowed:
            logger.debug(f"Signal rejected by risk limits: {reason}")
            reason_code = "RISK_BLOCKED"
            if "Position already exists" in reason:
                reason_code = "POSITION_EXISTS"
            elif "Max open positions" in reason:
                reason_code = "RISK_BLOCKED"
            metrics.increment("signals_rejected")
            no_trade_report.increment("risk_limits_reject", signal.symbol, signal.strategy_id)
            no_trade_report.increment_pipeline("signals_rejected_risk", signal.strategy_id)
            log_block_reason(
                "risk_limits_reject",
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                agent_name=signal.strategy_id,
                threshold=None,
                measured_value=None,
            )
            log_allow_reason("BLOCK", symbol=signal.symbol, strategy_id=signal.strategy_id, note=f"risk_limits: {reason}")
            await self._emit_order_decision(
                signal=signal,
                decision="REJECT",
                reason_code=reason_code,
                computed_qty=base_size,
                context={
                    "equity": equity,
                    "price": current_price,
                    "reason": reason,
                    "open_orders": len(self._order_manager.get_open_orders()),
                    "open_positions": len(self._accounting.get_positions()),
                },
            )
            return

        # Check daily attempts and cooldown (always active)
        regime = self._current_regime.get(signal.symbol, Regime.NORMAL)
        allowed, reason = self._risk_limits.check_daily_attempts(signal.strategy_id, regime)
        if not allowed:
            logger.debug(f"Signal rejected: {reason}")
            metrics.increment("signals_rejected")
            no_trade_report.increment(
                "daily_attempts_reject", signal.symbol, signal.strategy_id
            )
            no_trade_report.increment_pipeline(
                "signals_rejected_daily_attempts", signal.strategy_id
            )
            log_block_reason(
                "daily_attempts_reject",
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                agent_name=signal.strategy_id,
            )
            log_allow_reason("BLOCK", symbol=signal.symbol, strategy_id=signal.strategy_id, note=f"daily_attempts: {reason}")
            await self._emit_order_decision(
                signal=signal,
                decision="SKIP",
                reason_code="DAILY_LIMIT",
                computed_qty=base_size,
                context={
                    "reason": reason,
                    "attempts": self._risk_limits._daily_attempts.get(signal.strategy_id, 0),
                    "regime": regime.value if hasattr(regime, "value") else str(regime),
                    "open_orders": len(self._order_manager.get_open_orders()),
                    "open_positions": len(self._accounting.get_positions()),
                },
            )
            return

        # Check cooldown (always active)
        allowed, reason = self._risk_limits.check_cooldown(signal.strategy_id)
        if not allowed:
            logger.debug(f"Signal rejected: {reason}")
            metrics.increment("signals_rejected")
            no_trade_report.increment("cooldown_reject", signal.symbol, signal.strategy_id)
            no_trade_report.increment_pipeline("signals_rejected_cooldown", signal.strategy_id)
            log_block_reason(
                "cooldown_reject",
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                agent_name=signal.strategy_id,
            )
            log_allow_reason("BLOCK", symbol=signal.symbol, strategy_id=signal.strategy_id, note=f"cooldown: {reason}")
            await self._emit_order_decision(
                signal=signal,
                decision="SKIP",
                reason_code="COOLDOWN",
                computed_qty=base_size,
                context={
                    "reason": reason,
                    "consecutive_losses": self._risk_limits.get_consecutive_losses(signal.strategy_id),
                    "open_orders": len(self._order_manager.get_open_orders()),
                    "open_positions": len(self._accounting.get_positions()),
                },
            )
            return

        # ------------------------------------------------------------------
        # Decision Memory: context-aware gating and penalty
        # ------------------------------------------------------------------
        # Get current regime first (needed for context building)
        regime = self._current_regime.get(signal.symbol, Regime.NORMAL)

        # Build context from current regime, approximate spread/vol, and time
        spread_bps = None
        if getattr(signal, "metadata", None):
            # Strategies may pass finer-grained context
            spread_bps = signal.metadata.get("spread_bps")
        volatility = self._regime_detector.get_volatility(signal.symbol)
        context = self._decision_memory.build_context(
            regime=regime,
            spread_bps=spread_bps,
            volatility=volatility,
            timestamp=event.timestamp,
        )

        # Check if context is blocked (always active)
        has_dm_history = self._decision_memory.has_history(signal.strategy_id, context)
        if has_dm_history and self._decision_memory.blocked(signal.strategy_id, context):
            logger.debug(
                "Signal rejected by decision memory: strategy=%s context=%s",
                signal.strategy_id,
                context,
            )
            metrics.increment("signals_rejected")
            no_trade_report.increment(
                "decision_memory_block", signal.symbol, signal.strategy_id
            )
            no_trade_report.increment_pipeline(
                "signals_blocked_decision_memory", signal.strategy_id
            )
            log_block_reason(
                "decision_memory_block",
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                agent_name=signal.strategy_id,
            )
            log_allow_reason(
                "BLOCK",
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                note="decision_memory",
            )
            await self._emit_order_decision(
                signal=signal,
                decision="SKIP",
                reason_code="DECISION_MEMORY",
                computed_qty=base_size,
                context={
                    "reason": "decision_memory_block",
                    "context": context,
                    "has_history": has_dm_history,
                    "open_orders": len(self._order_manager.get_open_orders()),
                    "open_positions": len(self._accounting.get_positions()),
                },
            )
            return

        # Position size already calculated above, reuse it
        # (variables regime, rolling_pf, recent_drawdown, volatility_percentile, edge_bps already defined)
        logger.info(f"Base size calculated: {base_size:.6f} (signal.size={signal.size}, equity=${equity:.2f}, price=${current_price:.2f})")

        # CRITICAL: If base_size is 0 or negative, use minimum size immediately
        original_base_size = base_size
        if base_size <= 0:
            min_size_value = (equity * 0.001) / current_price  # 0.1% of equity in units
            absolute_min = 0.001
            base_size = max(min_size_value, absolute_min)
            logger.warning(
                f"Base size was {original_base_size:.6f}, using minimum size {base_size:.6f} "
                f"to ensure trading can occur"
            )

        # Apply protection size multiplier
        if protection_size_multiplier < 1.0:
            base_size *= protection_size_multiplier
            logger.debug(f"Position size reduced by protection: {protection_size_multiplier:.2f}x")
            # Ensure base_size doesn't become 0 after protection multiplier
            if base_size <= 0:
                min_size_value = (equity * 0.001) / current_price  # 0.1% of equity in units
                absolute_min = 0.001
                base_size = max(min_size_value, absolute_min)
                logger.warning(
                    f"Base size became 0 after protection multiplier, using minimum {base_size:.6f}"
                )

        # Decision memory penalty
        # Decision memory penalty (always active)
        penalty_multiplier = self._decision_memory.penalty(signal.strategy_id, context)

        size_multiplier = penalty_multiplier
        if getattr(signal, "metadata", None) and "size_multiplier" in signal.metadata:
            try:
                signal_mult = float(signal.metadata["size_multiplier"])
            except (TypeError, ValueError):
                signal_mult = 1.0
            # Combined multiplier, but still never > 1.0
            size_multiplier *= min(signal_mult, 1.0)
        logger.debug(f"Size multiplier after signal metadata: {size_multiplier}")

        if size_multiplier > 1.0:
            size_multiplier = 1.0  # Safety: do not increase risk above base sizing

        size = base_size * size_multiplier
        logger.debug(f"Final size: {size} (base={base_size}, mult={size_multiplier})")

        # CRITICAL FIX: Always ensure minimum size, never reject due to size
        # Calculate minimum size: at least 0.1% of equity worth
        min_size_value = (equity * 0.001) / current_price  # 0.1% of equity in units
        # Also ensure absolute minimum (e.g., 0.001 BTC or equivalent)
        absolute_min = 0.001
        min_size = max(min_size_value, absolute_min)

        if size < min_size:
            logger.warning(
                f"Position size {size:.6f} below minimum {min_size:.6f}, "
                f"using minimum size to allow trading"
            )
            size = min_size
            logger.info(
                f"Using minimum position size {size:.6f} to ensure trades can execute "
                f"(equity=${equity:.2f}, price=${current_price:.2f})"
            )

        if signal.metadata is None:
            signal.metadata = {}

        sizing_details = self._position_sizer.get_last_calculation() or {}
        applied_leverage = max(1.0, float(sizing_details.get("leverage", 1.0)))
        exit_plan = {
            "tp": signal.take_profit,
            "sl": signal.stop_loss,
            "trailing": signal.metadata.get("trailing"),
            "time_stop_seconds": signal.metadata.get("time_stop_seconds")
            or signal.metadata.get("max_time_sec"),
            "invalidation_rule": signal.metadata.get("invalidation_rule"),
        }
        expected_pnl = self._calculate_expected_pnl(
            side=signal.side,
            entry_price=signal.entry_price,
            qty=size,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
        )
        rationale = {
            "strategy_name": signal.strategy_id,
            "entry_reason": signal.metadata.get("entry_reason")
            or signal.metadata.get("reason")
            or "Strategy signal",
            "confidence": signal.metadata.get("confidence")
            or signal.metadata.get("edge_bps")
            or sizing_details.get("edge_bps"),
        }

        order_metadata = {
            **(signal.metadata or {}),
            "exit_plan": exit_plan,
            "expected_pnl": expected_pnl,
            "rationale": rationale,
            "applied_leverage": applied_leverage,
        }

        # Create order request
        signal_id = str(uuid.uuid4())
        order_request = OrderRequest(
            signal_id=signal_id,
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            side=signal.side,
            size=size,
            price=signal.entry_price,
            order_type="market",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata=order_metadata,
        )
        order_request.metadata["signal_id"] = signal_id

        # Track order creation
        no_trade_report.increment_pipeline("orders_created", signal.strategy_id)

        # Record attempt
        self._risk_limits.record_attempt(signal.strategy_id)

        # DEBUG: Track signals after filters and order creation
        self._signals_after_filters += 1
        self._orders_sent += 1
        metrics.increment("signals_after_filters_total")
        metrics.increment("orders_sent_total")

        # Log order creation details
        penalties = []
        if size_multiplier < 1.0:
            penalties.append(f"size_mult={size_multiplier:.2f}")
        if protection_size_multiplier < 1.0:
            penalties.append(f"protection_mult={protection_size_multiplier:.2f}")
        if penalty_multiplier < 1.0:
            penalties.append(f"penalty_mult={penalty_multiplier:.2f}")
        penalty_str = ", ".join(penalties) if penalties else "none"
        logger.debug(
            f"Order created: signal_id={order_request.signal_id} strategy={signal.strategy_id} "
            f"symbol={signal.symbol} side={signal.side} size={size:.6f} penalties=[{penalty_str}]"
        )

        # Emit ORDER_DECISION with full sizing context before publishing
        await self._emit_order_decision(
            signal=signal,
            decision="CREATE",
            reason_code="ACCEPTED",
            computed_qty=size,
            context={
                "price": current_price,
                "notional": size * current_price,
                "applied_leverage": applied_leverage,
                "equity": equity,
                "cash": self._accounting.get_cash_balance(),
                "open_orders": len(self._order_manager.get_open_orders()),
                "open_positions": len(self._accounting.get_positions()),
                "exit_plan": exit_plan,
                "expected_pnl": expected_pnl,
                "penalties": penalties,
            },
        )

        # Publish order request
        logger.debug(
            f"Publishing ORDER_REQUEST: {order_request.symbol} {order_request.side} size={order_request.size:.6f}"
        )
        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_REQUEST,
                data={"order_request": order_request},
            )
        )

        metrics.increment("signals_accepted")

        # Final diagnostic: ALLOW
        log_allow_reason(
            "ALLOW",
            symbol=signal.symbol,
            strategy_id=signal.strategy_id,
            note=f"OrderRequest created: size={size:.6f}",
        )

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order fill event."""
        logger.info("[ORDER_FILLED_HANDLER] Received ORDER_FILLED event")

        # Deduplicate: Bybit sends multiple partial fills for the same order_id.
        # Only process the FIRST fill per order_id to prevent phantom positions.
        if not hasattr(self, "_filled_order_ids"):
            self._filled_order_ids: OrderedDict[str, bool] = OrderedDict()

        raw_order_id = event.data.get("order_id") or (event.data.get("order") and getattr(event.data.get("order"), "order_id", None))
        if raw_order_id and raw_order_id in self._filled_order_ids:
            logger.debug(f"[ORDER_FILLED_HANDLER] Skipping duplicate fill for order {raw_order_id}")
            return
        if raw_order_id:
            self._filled_order_ids[raw_order_id] = True
            # Keep OrderedDict bounded by removing oldest entries
            while len(self._filled_order_ids) > 5000:
                self._filled_order_ids.popitem(last=False)

        # Safe extraction with defaults - try multiple sources
        order = event.data.get("order")
        fill_price_raw = event.data.get("fill_price", 0.0)
        fill_size_raw = event.data.get("fill_size", 0.0)

        if not order:
            order_id = event.data.get("order_id")
            if order_id:
                # Try order_manager first to get strategy_id/symbol
                managed = self._order_manager.get_order(order_id)
                # Always construct a FILLED order with actual fill data
                order = Order(
                    order_id=order_id,
                    strategy_id=managed.strategy_id if managed else event.data.get("strategy_id", "unknown"),
                    symbol=managed.symbol if managed else event.data.get("symbol", "UNKNOWN"),
                    side=(managed.side if managed else event.data.get("side", "buy")).lower(),
                    size=fill_size_raw or (managed.size if managed else 0.0),
                    filled_size=fill_size_raw or (managed.size if managed else 0.0),
                    price=fill_price_raw,
                    avg_fill_price=fill_price_raw,
                    order_type=managed.order_type if managed else "limit",
                    status=OrderStatus.FILLED,
                    timestamp=datetime.utcnow(),
                    stop_loss=managed.stop_loss if managed else None,
                    take_profit=managed.take_profit if managed else None,
                    metadata=managed.metadata if managed else event.data.get("metadata", {}),
                )
                logger.info(f"[ORDER_FILLED_HANDLER] Built filled order: {order_id} {order.side} {order.filled_size} {order.symbol} @ {fill_price_raw}")
            else:
                logger.warning(f"[ORDER_FILLED_HANDLER] Missing order_id in event data: {event.data}")
                return

        fill_price = event.data.get("fill_price", 0.0)
        event.data.get("fill_size", 0.0)
        fee = event.data.get("fee", 0.0)

        logger.info(
            f"[ORDER_FILLED_HANDLER] Processing order {order.order_id}: "
            f"status={order.status}, filled_size={order.filled_size}, "
            f"size={order.size}, fill_price={fill_price:.2f}, fee={fee:.4f}"
        )

        # DEBUG: Track order fills
        self._orders_filled += 1
        metrics.increment("orders_filled_total")

        # Record trade for density tracking
        trade_density.record_trade(order.strategy_id, event.timestamp)

        # Update accounting (cash adjustment for the fill)
        self._accounting.record_fill(order, fill_price, fee)
        logger.info(f"[ORDER_FILLED_HANDLER] Accounting updated for order {order.order_id}")

        # Detect close fills: check if this fill reduces an existing position
        # A close fill occurs when we have a position and the fill is on the opposite side
        is_close_fill = False
        closing_position = None

        # Check metadata first (our own close orders)
        if order.metadata and order.metadata.get("is_close"):
            is_close_fill = True

        # Check Bybit closedSize indicator
        if event.data.get("closedSize") or event.data.get("reduce_only"):
            is_close_fill = True

        # Heuristic: if we have an existing position for this symbol on the opposite side
        if not is_close_fill:
            fill_side = "long" if order.side == "buy" else "short"
            for pos in self._accounting.get_positions():
                if pos.symbol == order.symbol and pos.side != fill_side:
                    is_close_fill = True
                    break

        if is_close_fill:
            # Find the position being closed
            for pos in self._accounting.get_positions():
                if pos.symbol == order.symbol:
                    closing_position = pos
                    break

            if closing_position:
                # Calculate realized PnL
                if closing_position.side == "long":
                    realized_pnl = (fill_price - closing_position.entry_price) * closing_position.size
                else:
                    realized_pnl = (closing_position.entry_price - fill_price) * closing_position.size

                closing_position.current_price = fill_price
                closing_position.realized_pnl = realized_pnl

                logger.info(
                    f"[ORDER_FILLED_HANDLER] Close fill detected: closing {closing_position.position_id} "
                    f"({closing_position.symbol} {closing_position.side}) PnL={realized_pnl:.4f}"
                )

                await self._bus.publish(
                    Event(
                        event_type=EventType.POSITION_CLOSED,
                        data={
                            "position": closing_position,
                            "close_reason": "exchange_fill",
                            "exchange_close": True,
                        },
                    )
                )
                metrics.increment("positions_closed")
            else:
                logger.warning(
                    f"[ORDER_FILLED_HANDLER] Close fill for {order.symbol} but no matching position found"
                )

            await self._emit_account_state()
            return

        # Open fill: Create position if fully filled
        if order.status == OrderStatus.FILLED and order.avg_fill_price:
            logger.info(
                f"[ORDER_FILLED_HANDLER] Order {order.order_id} is fully filled, "
                f"creating position: avg_fill_price={order.avg_fill_price:.2f}"
            )
            take_profit_1 = None
            if "take_profit_1" in order.metadata:
                take_profit_1 = order.metadata["take_profit_1"]

            position = Position(
                position_id=str(uuid.uuid4()),
                symbol=order.symbol,
                side="long" if order.side == "buy" else "short",
                size=order.filled_size,
                entry_price=order.avg_fill_price,
                current_price=order.avg_fill_price,
                opened_at=order.timestamp,
                strategy_id=order.strategy_id,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                take_profit_1=take_profit_1,
                max_time_sec=(
                    settings.impulse_max_time_in_trade_sec
                    if order.strategy_id == "impulse_engine"
                    else None
                ),
                metadata=order.metadata or {},
            )

            self._accounting.add_position(position)
            self._risk_limits.register_position(position)
            logger.info(
                f"[ORDER_FILLED_HANDLER] Position {position.position_id} created and registered: "
                f"{position.side} {position.size} {position.symbol} @ {position.entry_price:.2f}"
            )

            await self._bus.publish(
                Event(
                    event_type=EventType.POSITION_OPENED,
                    data={"position": position},
                )
            )

            metrics.increment("positions_opened")
            no_trade_report.increment_pipeline("positions_opened", order.strategy_id)

            # Exchange-side SL/TP
            if order.stop_loss or order.take_profit:
                try:
                    bybit_http = getattr(self._execution_router, "_bybit_http", None)
                    if bybit_http:
                        await bybit_http.set_trading_stop(
                            symbol=order.symbol,
                            stop_loss=order.stop_loss,
                            take_profit=order.take_profit,
                        )
                        logger.info(
                            f"[EXCHANGE PROTECTION] SL/TP set on Bybit for {order.symbol}: "
                            f"SL={order.stop_loss}, TP={order.take_profit}"
                        )
                except Exception as e:
                    logger.error(
                        f"[EXCHANGE PROTECTION] Failed to set SL/TP on Bybit for "
                        f"{order.symbol}: {e}. PositionMonitor still active as fallback."
                    )
        else:
            logger.warning(
                f"[ORDER_FILLED_HANDLER] Order {order.order_id} not fully filled or missing avg_fill_price: "
                f"status={order.status}, avg_fill_price={order.avg_fill_price}"
            )
        await self._emit_account_state()

    async def _handle_position_closed(self, event: Event) -> None:
        """Handle position closed event."""
        position = event.data.get("position")
        if not position:
            logger.warning(f"[POSITION_CLOSED] Missing 'position' in event data: {event.data}")
            return
        self._accounting.remove_position(position.position_id)
        self._risk_limits.unregister_position(position.position_id)

        # Record win/loss
        if position.realized_pnl > 0:
            self._risk_limits.record_win(position.strategy_id)
        else:
            self._risk_limits.record_loss(position.strategy_id)

        # Record realized PnL per strategy and regime
        regime = self._current_regime.get(position.symbol, Regime.NORMAL)
        self._accounting.record_realized_pnl(
            position.realized_pnl, position.strategy_id, regime.value
        )

        # Build context key for decision memory
        spread_bps = None
        volatility = None
        if position.metadata:
            spread_bps = position.metadata.get("spread_bps")
            volatility = position.metadata.get("volatility")

        # Get volatility from regime detector if not in metadata
        if volatility is None:
            volatility = self._regime_detector.get_volatility(position.symbol)

        context_key = self._decision_memory.build_context(
            regime=regime,
            spread_bps=spread_bps,
            volatility=volatility,
            timestamp=event.timestamp,
        )

        # Record close with new API
        self._decision_memory.record_close(
            strategy_id=position.strategy_id,
            context_key=context_key,
            pnl=position.realized_pnl,
            timestamp=event.timestamp,
        )

        # Record profit in profit target tracker
        self._profit_tracker.record_profit(position.realized_pnl)

        # УМНАЯ РЕИНВЕСТИЦИЯ после каждой прибыльной сделки
        if position.realized_pnl > 0:
            profit = position.realized_pnl
            current_equity = self._accounting.get_equity()
            initial_capital = self._accounting.initial_capital
            _, drawdown_pct = self._accounting.get_drawdown(current_equity)

            # Получить rolling PF для стратегии
            strategy_metrics = self._accounting.get_strategy_metrics()
            rolling_pf = 1.0
            if strategy_metrics and position.strategy_id in strategy_metrics:
                rolling_pf = strategy_metrics[position.strategy_id].get("profit_factor", 1.0)

            # Рассчитать умную реинвестицию
            reinvestor = SmartReinvestor()
            reinvest_amount = reinvestor.calculate_smart_reinvestment(
                profit=profit,
                current_equity=current_equity,
                initial_capital=initial_capital,
                drawdown_pct=drawdown_pct,
                rolling_pf=rolling_pf,
            )

            # Немедленно реинвестировать
            if reinvest_amount > 0:
                self._accounting.update_cash(reinvest_amount)
                logger.info(
                    f"SMART REINVESTMENT: Profit=${profit:.2f}, "
                    f"Reinvest=${reinvest_amount:.2f} ({reinvest_amount / profit * 100:.1f}%), "
                    f"New Equity=${self._accounting.get_equity():.2f}, "
                    f"Drawdown={drawdown_pct:.1f}%, PF={rolling_pf:.2f}"
                )

        metrics.increment("positions_closed")
        no_trade_report.increment_pipeline("positions_closed", position.strategy_id)
        await self._emit_account_state()

    async def _handle_stop_trading(self, event: Event) -> None:
        """Handle stop trading event."""
        reason = event.data.get("reason", "Unknown")
        logger.warning(f"Stop trading triggered: {reason}")
        self._stop_trading = True

    async def _handle_killswitch(self, event: Event) -> None:
        """Handle killswitch triggered event."""
        reason = event.data.get("reason", "Unknown")
        logger.critical(f"Killswitch triggered: {reason}")
        self._stop_trading = True

    async def _handle_position_close_request(self, event: Event) -> None:
        """Handle POSITION_CLOSE_REQUEST from Oracle or other modules.

        Routes close requests to ExecutionRouter to actually close positions.
        """
        position_id = event.data.get("position_id")
        reason = event.data.get("reason", "external_close_request")
        if not position_id:
            logger.warning("[CLOSE_REQUEST] Missing position_id in event data")
            return

        # Find the position
        position = None
        for p in self._accounting.get_positions():
            if p.position_id == position_id:
                position = p
                break

        if not position:
            logger.warning(f"[CLOSE_REQUEST] Position {position_id} not found")
            return

        logger.info(f"[CLOSE_REQUEST] Closing position {position_id} reason={reason}")
        price = (
            self._execution_router._current_prices.get(position.symbol)
            if hasattr(self._execution_router, "_current_prices")
            else None
        )
        price = price or position.current_price or position.entry_price
        await self._close_position_at_price(position, price, reason=reason)

    async def _handle_meta_learning_patch(self, event: Event) -> None:
        """Handle META_LEARNING_PATCH — apply parameter updates to strategies."""
        patch = event.data.get("patch", {})
        strategy_id = event.data.get("strategy_id")
        if not patch:
            return

        applied = False
        for strategy in self._strategies:
            if strategy_id and strategy.strategy_id != strategy_id:
                continue
            # Apply patch parameters to strategy
            for key, value in patch.items():
                if hasattr(strategy, key):
                    old_val = getattr(strategy, key)
                    setattr(strategy, key, value)
                    logger.info(
                        f"[META_LEARNING] {strategy.strategy_id}.{key}: {old_val} -> {value}"
                    )
                    applied = True

        if applied:
            await self._bus.publish(Event(
                event_type=EventType.STRATEGY_PARAMS_UPDATED,
                data={"strategy_id": strategy_id, "patch": patch, "source": "meta_learning"},
            ))

    async def panic_close_all(self, reason: str = "panic_close_all") -> dict[str, Any]:
        """Force-close all positions and cancel open orders (paper-safe)."""
        closed_positions: list[str] = []
        for position in list(self._accounting.get_positions()):
            price = (
                self._execution_router._current_prices.get(position.symbol)
                if hasattr(self._execution_router, "_current_prices")
                else None
            )
            price = price or position.current_price or position.entry_price
            await self._emit_order_exit_decision(
                position=position,
                decision="FORCE_CLOSE",
                reason_code="RISK_EXIT",
                tick_price=price,
                thresholds={
                    "tp": position.take_profit,
                    "sl": position.stop_loss,
                    "time_stop_seconds": position.max_time_sec or settings.max_hold_seconds,
                },
                hold_seconds=(datetime.utcnow() - position.opened_at).total_seconds()
                if position.opened_at
                else None,
                note=reason,
            )
            await self._close_position_at_price(position, price, reason=reason)
            closed_positions.append(position.position_id)

        # Cancel open orders
        for order in self._order_manager.get_open_orders():
            order.status = OrderStatus.CANCELLED
            await self._bus.publish(
                Event(
                    event_type=EventType.ORDER_CANCELLED,
                    data={"order": order},
                )
            )

        await self._emit_account_state()
        return {"closed_positions": len(closed_positions), "cancelled_orders": len(self._order_manager.get_open_orders())}

    async def reset_paper_state(self) -> dict[str, Any]:
        """Clear paper-trading state (positions, orders, cached decisions)."""
        # Remove positions
        for pos in list(self._accounting.get_positions()):
            self._accounting.remove_position(pos.position_id)

        # Reset orders
        self._order_manager._orders.clear()
        self._order_manager._orders_by_strategy.clear()
        self._order_manager._orders_by_symbol.clear()

        # Reset diagnostics / histories
        self._order_exit_decision_history.clear()
        self._order_decision_history.clear()
        self._last_exit_decision_ts.clear()
        self._last_tick_at.clear()

        # Reset killswitch and stop_trading
        if hasattr(self, "_killswitch"):
            self._killswitch.reset()
            logger.info("Killswitch reset via reset_paper_state")
        self._stop_trading = False
        logger.info("Stop trading flag reset via reset_paper_state")

        await self._emit_account_state()
        return {"status": "reset", "positions": 0, "orders": 0}

    def _calculate_expected_pnl(
        self,
        side: str,
        entry_price: float,
        qty: float,
        take_profit: float | None,
        stop_loss: float | None,
    ) -> dict[str, float | None]:
        """Calculate expected PnL at TP/SL for telemetry."""
        if entry_price is None or qty <= 0:
            return {"at_tp": None, "at_sl": None, "breakeven_price": None, "rr_ratio": None}

        at_tp = None
        at_sl = None

        if take_profit is not None:
            if side == "buy":
                at_tp = (take_profit - entry_price) * qty
            else:
                at_tp = (entry_price - take_profit) * qty

        if stop_loss is not None:
            if side == "buy":
                at_sl = (stop_loss - entry_price) * qty
            else:
                at_sl = (entry_price - stop_loss) * qty

        rr_ratio = None
        if at_tp is not None and at_sl not in (None, 0):
            try:
                rr_ratio = abs(at_tp / abs(at_sl))
            except ZeroDivisionError:
                rr_ratio = None

        fee_rate = settings.backtest_taker_fee or 0.0
        direction = 1 if side == "buy" else -1
        breakeven_price = entry_price + (entry_price * fee_rate * 2 * direction)

        return {
            "at_tp": at_tp,
            "at_sl": at_sl,
            "breakeven_price": breakeven_price,
            "rr_ratio": rr_ratio,
        }

    def _build_trading_state(self) -> dict[str, Any]:
        """Build unified trading state snapshot for UI/telemetry."""
        mark_prices = getattr(self._execution_router, "_current_prices", {})
        positions_payload: list[dict[str, Any]] = []
        orders_payload: list[dict[str, Any]] = []
        used_margin = 0.0
        reserved_margin = 0.0

        # Positions
        for pos in self._accounting.get_positions():
            mark_price = mark_prices.get(pos.symbol, pos.current_price)
            if mark_price:
                self._accounting.update_position_price(pos.position_id, mark_price)
            metadata = pos.metadata or {}
            try:
                leverage = max(1.0, float(metadata.get("applied_leverage", 1.0)))
            except (TypeError, ValueError):
                leverage = 1.0
            notional = (mark_price or pos.entry_price) * pos.size
            margin_used = notional / leverage if leverage else notional
            used_margin += margin_used
            exit_plan = metadata.get("exit_plan") or {
                "tp": pos.take_profit,
                "sl": pos.stop_loss,
                "trailing": metadata.get("trailing"),
                "time_stop_seconds": metadata.get("max_time_sec"),
                "invalidation_rule": metadata.get("invalidation_rule"),
            }
            positions_payload.append(
                {
                    "position_id": pos.position_id,
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "qty": pos.size,
                    "entry_price": pos.entry_price,
                    "mark_price": mark_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "leverage": leverage,
                    "margin_used": margin_used,
                    "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                    "exit_plan": exit_plan,
                    "expected_pnl": metadata.get("expected_pnl"),
                    "strategy_id": pos.strategy_id,
                    "status": "open",
                    "rationale": metadata.get("rationale"),
                }
            )

        # Orders (all known states)
        all_orders = self._order_manager.get_all_orders()
        for order in all_orders:
            metadata = order.metadata or {}
            status_value = order.status.value if hasattr(order.status, "value") else str(
                order.status
            )
            mark_price = mark_prices.get(order.symbol, order.price)
            notional = (mark_price or order.price or 0.0) * order.size
            try:
                leverage = max(1.0, float(metadata.get("applied_leverage", 1.0)))
            except (TypeError, ValueError):
                leverage = 1.0
            if order.status in {
                OrderStatus.PENDING,
                OrderStatus.PLACED,
                OrderStatus.PARTIALLY_FILLED,
            }:
                reserved_margin += notional / leverage if leverage else notional

            orders_payload.append(
                {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.order_type,
                    "qty": order.size,
                    "price": order.price,
                    "status": status_value,
                    "created_at": order.timestamp.isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "filled_qty": order.filled_size,
                    "avg_fill_price": order.avg_fill_price,
                    "fee": metadata.get("fee"),
                    "strategy_id": order.strategy_id,
                    "signal_id": metadata.get("signal_id"),
                    "rationale": metadata.get("rationale"),
                    "exit_plan": metadata.get("exit_plan"),
                    "expected_pnl": metadata.get("expected_pnl"),
                    "status_timeline": metadata.get("status_timeline", []),
                }
            )

        unrealized_total = self._accounting.get_unrealized_pnl_total()
        equity_value = self._accounting.get_equity(mark_prices)

        # Use equity as wallet_balance since accounting already includes all positions
        # Don't add used_margin - it's already accounted for in equity calculation
        wallet_balance = equity_value
        available_balance = equity_value - used_margin - reserved_margin

        account_state = {
            "wallet_balance": wallet_balance,
            "available_balance": max(0.0, available_balance),  # Prevent negative available
            "equity": equity_value,
            "used_margin": used_margin,
            "reserved_margin": reserved_margin,
            "unrealized_pnl": unrealized_total,
            "realized_pnl": self._accounting.get_realized_pnl_total(),
            "fees": self._accounting.get_total_fees(),
            "fees_24h": self._accounting.get_total_fees(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return {
            "account_state": account_state,
            "positions": positions_payload,
            "orders": orders_payload,
        }

    async def _emit_account_state(self) -> None:
        """Publish account/position state for UI and monitoring."""
        snapshot = self._build_trading_state()
        await self._bus.publish(
            Event(
                event_type=EventType.PNL_UPDATE,
                data=snapshot,
            )
        )

    async def _handle_tick_forced_exit(self, event: Event) -> None:
        """Mark-to-market on every tick and evaluate exit plans/TTL."""
        tick = event.data["tick"]
        symbol = tick.symbol
        now = datetime.utcnow()
        self._last_tick_at[symbol] = now

        # Mark-to-market update for open positions on this symbol
        for position in self._accounting.get_positions():
            if position.symbol == symbol:
                self._accounting.update_position_price(position.position_id, tick.price)

        # Broadcast fresh account snapshot (PnL_UPDATE for UI/exit triggers)
        await self._emit_account_state()

        # Evaluate exits for positions on this symbol
        for position in list(self._accounting.get_positions()):
            if position.symbol != symbol:
                continue
            await self._evaluate_exit_for_position(position, tick, now)

    async def _evaluate_exit_for_position(
        self,
        position: Position,
        tick: Tick,
        now: datetime | None = None,
    ) -> None:
        """Evaluate TP/SL/TTL for a position and emit exit telemetry."""
        now = now or datetime.utcnow()
        metadata = position.metadata or {}
        exit_plan = metadata.get("exit_plan") or {
            "tp": position.take_profit,
            "sl": position.stop_loss,
            "trailing": metadata.get("trailing"),
            "time_stop_seconds": metadata.get("max_time_sec"),
            "invalidation_rule": metadata.get("invalidation_rule"),
        }

        tp = exit_plan.get("tp") or position.take_profit
        sl = exit_plan.get("sl") or position.stop_loss
        time_stop = exit_plan.get("time_stop_seconds") or position.max_time_sec

        # Paper-assist micro trades may specify max_time_min (minutes)
        if metadata.get("micro_trade"):
            if time_stop is None:
                max_time_min = metadata.get(
                    "max_time_min", settings.paper_trade_assist_micro_trade_max_time_min
                )
                time_stop = max_time_min * 60

        # Global anti-stuck TTL
        if time_stop is None:
            time_stop = settings.max_hold_seconds

        hold_seconds = None
        if position.opened_at:
            hold_seconds = (now - position.opened_at).total_seconds()

        # Use best available price
        price = (
            tick.price
            or self._execution_router._current_prices.get(position.symbol)
            or position.current_price
            or position.entry_price
        )

        thresholds = {
            "tp": tp,
            "sl": sl,
            "time_stop_seconds": time_stop,
        }

        # Exit plan missing guard
        if tp is None and sl is None and time_stop is None:
            await self._emit_order_exit_decision(
                position=position,
                decision="HOLD",
                reason_code="EXIT_PLAN_MISSING",
                tick_price=price,
                thresholds=thresholds,
                hold_seconds=hold_seconds,
                note="No TP/SL/TTL configured",
            )
            return

        # TTL enforcement
        if time_stop and hold_seconds is not None and hold_seconds >= time_stop:
            await self._emit_order_exit_decision(
                position=position,
                decision="FORCE_CLOSE",
                reason_code="TIMEOUT_TTL",
                tick_price=price,
                thresholds=thresholds,
                hold_seconds=hold_seconds,
            )
            await self._close_position_at_price(position, price, reason="timeout_ttl")
            return

        # Take-profit checks
        if tp is not None:
            if position.side == "long" and price >= tp:
                await self._emit_order_exit_decision(
                    position=position,
                    decision="CLOSE",
                    reason_code="TP_HIT",
                    tick_price=price,
                    thresholds=thresholds,
                    hold_seconds=hold_seconds,
                )
                await self._close_position_at_price(position, tp, reason="take_profit")
                return
            if position.side == "short" and price <= tp:
                await self._emit_order_exit_decision(
                    position=position,
                    decision="CLOSE",
                    reason_code="TP_HIT",
                    tick_price=price,
                    thresholds=thresholds,
                    hold_seconds=hold_seconds,
                )
                await self._close_position_at_price(position, tp, reason="take_profit")
                return

        # Stop-loss checks
        if sl is not None:
            if position.side == "long" and price <= sl:
                await self._emit_order_exit_decision(
                    position=position,
                    decision="CLOSE",
                    reason_code="SL_HIT",
                    tick_price=price,
                    thresholds=thresholds,
                    hold_seconds=hold_seconds,
                )
                await self._close_position_at_price(position, sl, reason="stop_loss")
                return
            if position.side == "short" and price >= sl:
                await self._emit_order_exit_decision(
                    position=position,
                    decision="CLOSE",
                    reason_code="SL_HIT",
                    tick_price=price,
                    thresholds=thresholds,
                    hold_seconds=hold_seconds,
                )
                await self._close_position_at_price(position, sl, reason="stop_loss")
                return

        # Nothing to do -> emit throttled HOLD
        last_ts = self._last_exit_decision_ts.get(position.position_id)
        if not last_ts or (now - last_ts).total_seconds() >= 2:
            await self._emit_order_exit_decision(
                position=position,
                decision="HOLD",
                reason_code="HOLD",
                tick_price=price,
                thresholds=thresholds,
                hold_seconds=hold_seconds,
            )
            self._last_exit_decision_ts[position.position_id] = now

    async def _micro_trade_fallback_loop(self) -> None:
        """Deprecated: micro-trade fallback disabled. System uses Bybit testnet directly."""
        logger.warning(
            "[DEPRECATED] _micro_trade_fallback_loop is disabled. "
            "System operates on Bybit testnet — no paper trading fallback needed."
        )
        return

    async def _check_position_timeouts(self) -> None:
        """Periodic safety check for stale ticks and TTL enforcement."""
        if not self._running:
            return

        now = datetime.utcnow()
        positions = list(self._accounting.get_positions())
        for position in positions:
            last_tick = self._last_tick_at.get(position.symbol)
            stale = last_tick is None or (now - last_tick).total_seconds() > 3
            if stale:
                price = (
                    self._execution_router._current_prices.get(position.symbol)
                    if hasattr(self._execution_router, "_current_prices")
                    else None
                )
                price = price or position.current_price or position.entry_price
                thresholds = {
                    "tp": position.take_profit,
                    "sl": position.stop_loss,
                    "time_stop_seconds": position.max_time_sec or settings.max_hold_seconds,
                }
                # Emit NO_TICKS hold event (throttled via _evaluate_exit_for_position)
                synthetic_tick = Tick(
                    symbol=position.symbol,
                    price=price,
                    timestamp=now,
                    volume=0.0,
                )
                await self._emit_order_exit_decision(
                    position=position,
                    decision="HOLD",
                    reason_code="NO_TICKS",
                    tick_price=price,
                    thresholds=thresholds,
                    hold_seconds=(now - position.opened_at).total_seconds()
                    if position.opened_at
                    else None,
                    note="No recent market ticks; using last known price",
                )
                # Re-evaluate exits/TTL using synthetic tick to prevent stuck positions
                await self._evaluate_exit_for_position(position, synthetic_tick, now)

        # Emit refreshed account state so UI sees updated unrealized PnL
        await self._emit_account_state()

    async def _close_position_at_price(
        self,
        position: Position,
        close_price: float,
        reason: str = "forced_exit",
    ) -> None:
        """Close a position at a specific price."""
        close_side = "sell" if position.side == "long" else "buy"
        exchange_close = False

        # Send close order to Bybit BEFORE updating internal accounting
        if not settings.dry_run and settings.is_live:
            if hasattr(self._execution_router, "_bybit_http") and self._execution_router._bybit_http:
                try:
                    close_request = OrderRequest(
                        signal_id=f"close_{position.position_id}_{int(datetime.utcnow().timestamp())}",
                        symbol=position.symbol,
                        side=close_side,
                        size=position.size,
                        order_type="market",
                        strategy_id=position.strategy_id,
                        reduce_only=True,
                        metadata={"is_close": True, "position_id": position.position_id},
                    )

                    logger.info(f"Closing position on Bybit: {position.position_id} ({position.symbol} {position.side} {position.size}) - {reason}")
                    await self._execution_router._bybit_http.place_order(close_request)
                    logger.info(f"Successfully closed position on Bybit: {position.position_id}")
                    exchange_close = True
                    # On live: ORDER_FILLED will fire from WS → record_fill adjusts cash
                    # We still proceed to update internal state below

                except Exception as e:
                    logger.critical(
                        f"FAILED TO CLOSE POSITION ON BYBIT: {position.position_id} ({position.symbol}) - {e}. "
                        f"Position remains open on exchange! Manual intervention required."
                    )
                    return

        # Update position price
        self._accounting.update_position_price(position.position_id, close_price)

        # Calculate realized PnL
        if position.side == "long":
            realized_pnl = (close_price - position.entry_price) * position.size
        else:  # short
            realized_pnl = (position.entry_price - close_price) * position.size

        # Paper mode: simulate sell proceeds in cash (record_fill won't run)
        if not exchange_close:
            notional = close_price * position.size
            est_fee = notional * 0.00055  # Bybit taker fee
            if close_side == "sell":
                self._accounting.update_cash(notional - est_fee)
            else:
                self._accounting.update_cash(-(notional + est_fee))

        position.current_price = close_price
        position.realized_pnl = realized_pnl

        # Publish position closed event with exchange_close flag
        await self._bus.publish(
            Event(
                event_type=EventType.POSITION_CLOSED,
                data={
                    "position": position,
                    "close_reason": reason,
                    "exchange_close": exchange_close,
                },
            )
        )

    async def _handle_regime_update(self, event: Event) -> None:
        """Handle regime update event."""
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        if symbol is None or regime is None:
            logger.warning("REGIME_UPDATE missing symbol/regime: %s", event.data)
            return
        self._current_regime[symbol] = regime
        logger.debug(f"Regime updated: {symbol} -> {regime.value}")

    async def _print_status(self) -> None:
        """Print periodic status update."""
        if not self._running:
            return

        # Get current prices (simplified - would get from price feed)
        current_prices: dict[str, float] = {}
        for pos in self._accounting.get_positions():
            if pos.symbol not in current_prices:
                # Use entry price as proxy (in real system, get from feed)
                current_prices[pos.symbol] = pos.current_price

        snapshot = self._accounting.snapshot(current_prices)

        # Check killswitch — use REAL Bybit equity to avoid phantom position inflation.
        # Internal PortfolioAccounting can have inflated equity from duplicate fills.
        equity = snapshot.equity
        if settings.is_live and not settings.dry_run:
            try:
                from hean.api.routers.engine import _fetch_bybit_balance
                bybit_bal = await _fetch_bybit_balance()
                if bybit_bal and bybit_bal.get("equity", 0) > 0:
                    real_equity = bybit_bal["equity"]
                    logger.debug(f"Killswitch using Bybit equity: ${real_equity:.2f} (internal: ${equity:.2f})")
                    equity = real_equity
            except Exception as e:
                logger.debug(f"Bybit equity fetch failed for killswitch: {e}")
        # Use initial_capital as peak to avoid inflated peak from phantom positions
        peak_equity = max(settings.initial_capital or 10000.0, equity)
        # Get regime for killswitch check (use first symbol's regime or NORMAL)
        regime = Regime.NORMAL
        if self._current_regime:
            regime = list(self._current_regime.values())[0]

        # Get strategy metrics first (needed for killswitch check)
        strategy_metrics = self._accounting.get_strategy_metrics()

        # Get rolling PF for killswitch
        rolling_pf = None
        if strategy_metrics:
            total_wins = sum(m.get("wins", 0) for m in strategy_metrics.values())
            total_losses = sum(m.get("losses", 0) for m in strategy_metrics.values())
            if total_losses > 0:
                rolling_pf = total_wins / total_losses
            elif total_wins > 0:
                rolling_pf = total_wins
        await self._killswitch.check_drawdown(equity, peak_equity, regime, rolling_pf)

        # Update strategy equity tracking
        for pos in self._accounting.get_positions():
            strategy_equity = pos.current_price * pos.size + pos.unrealized_pnl
            self._accounting.update_strategy_equity(pos.strategy_id, strategy_equity)

        # Update adaptive capital allocation (daily)
        if strategy_metrics:
            # Update weights with LLM optimization
            self._allocator.update_weights(strategy_metrics)

            # Apply LLM-based capital optimization
            if self._capital_optimizer and strategy_metrics:
                try:
                    current_weights = {
                        s: self._allocator._weights.get(s, 0.0) for s in strategy_metrics.keys()
                    }
                    optimized_weights = self._capital_optimizer.optimize_allocation(
                        strategy_metrics=strategy_metrics,
                        current_weights=current_weights,
                    )
                    # Apply optimized weights (with limits to prevent sudden changes)
                    if optimized_weights:
                        logger.info(f"Capital optimizer suggested weights: {optimized_weights}")
                except Exception as e:
                    logger.debug(f"Capital optimization error: {e}")

        # Update killswitch with initial capital and rolling PF
        if self._killswitch:
            self._killswitch.set_initial_capital(self._accounting.initial_capital)
            # Get rolling PF from strategy metrics
            if strategy_metrics:
                # Use average PF across all strategies
                total_wins = sum(m.get("wins", 0) for m in strategy_metrics.values())
                total_losses = sum(m.get("losses", 0) for m in strategy_metrics.values())
                if total_losses > 0:
                    avg_pf = total_wins / total_losses
                elif total_wins > 0:
                    avg_pf = total_wins
                else:
                    avg_pf = 1.0
                self._killswitch.update_rolling_pf(avg_pf)

        # Update trade density metrics for all strategies
        for strategy in self._strategies:
            density_state = trade_density.get_density_state(strategy.strategy_id)
            metrics.set_gauge(
                f"trade_density_idle_days_{strategy.strategy_id}", density_state["idle_days"]
            )
            metrics.set_gauge(
                f"trade_density_relaxation_level_{strategy.strategy_id}",
                float(density_state["density_relaxation_level"]),
            )

        # Get current weights for display
        weights = self._allocator.get_weights()
        weights_str = ", ".join([f"{k}: {v:.1%}" for k, v in weights.items()])

        # Get profit target progress
        progress = self._profit_tracker.get_progress()

        # Check profit capture (if enabled)
        if self._profit_capture and self._profit_capture._enabled:
            try:
                await self._profit_capture.check_and_trigger(equity, self)
            except Exception as e:
                logger.debug(f"Profit capture check error: {e}")

        # Generate daily report (once per day)
        if self._report_generator:
            try:
                current_date = datetime.utcnow().date()
                if (
                    not hasattr(self, "_last_report_date")
                    or getattr(self, "_last_report_date", None) != current_date
                ):
                    strategy_metrics = self._accounting.get_strategy_metrics()
                    performance_data = {
                        "equity": snapshot.equity,
                        "drawdown_pct": snapshot.drawdown_pct,
                        "total_trades": sum(
                            m.get("trades", 0) for m in (strategy_metrics or {}).values()
                        ),
                    }
                    report = self._report_generator.generate_daily_report(
                        performance_data=performance_data,
                        strategy_metrics=strategy_metrics or {},
                    )
                    if report:
                        logger.info(
                            f"📊 Daily Report: PnL=${report['summary']['total_pnl']:.2f}, "
                            f"Best Strategy={report.get('best_strategy', {}).get('id', 'N/A')} "
                            f"(PF={report.get('best_strategy', {}).get('profit_factor', 0):.2f})"
                        )
                    self._last_report_date = current_date
            except Exception as e:
                logger.debug(f"Report generation error: {e}")

        print(
            f"[{snapshot.timestamp.strftime('%H:%M:%S')}] "
            f"Equity: ${snapshot.equity:,.2f} | "
            f"Daily PnL: ${snapshot.daily_pnl:,.2f} | "
            f"Profit: ${progress['current']:.2f}/${progress['target']:.2f} ({progress['progress_pct']:.1f}%) | "
            f"Trades: {int(progress['trades'])} | "
            f"Drawdown: {snapshot.drawdown_pct:.2f}% | "
            f"Positions: {len(self._accounting.get_positions())} | "
            f"Weights: {weights_str}"
        )

    async def run(self) -> None:
        """Run the trading system."""
        await self.start()
        try:
            while self._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.stop()


async def run_evaluation(days: int) -> dict[str, Any]:
    """Run readiness evaluation using TradingSystem.

    ARCHITECTURAL FIX: Previously, run_evaluation() created components
    independently, bypassing ExecutionRouter and TradingSystem's signal→order→fill
    pipeline. This resulted in zero trades and invalid metrics (PF=0, MaxDD=100%).

    Now, evaluation uses TradingSystem as the execution backbone, ensuring:
    - Signals are routed through risk limits
    - Orders are executed via ExecutionRouter
    - Fills update PortfolioAccounting
    - Same logic as run/backtest commands

    Returns:
        Evaluation result dictionary
    """
    from hean.evaluation.readiness import ReadinessEvaluator

    logger.info(f"[run_evaluation] Starting readiness evaluation for {days} days...")

    # Reset no-trade counters at the start of each evaluation run
    no_trade_report.reset()

    # Create EventSimulator as price feed
    start_date = datetime.utcnow() - timedelta(days=days)
    simulator = EventSimulator(None, ["BTCUSDT", "ETHUSDT"], start_date, days)
    # Note: EventSimulator will get bus from TradingSystem.start()

    # Create TradingSystem in evaluate mode
    system = TradingSystem(mode="evaluate")

    # Start system with EventSimulator as price feed
    await system.start(price_feed=simulator)

    logger.info("[run_evaluation] TradingSystem started, beginning simulation...")

    # Run simulation (EventSimulator will publish ticks)
    await simulator.run()

    logger.info("[run_evaluation] Simulation completed, processing remaining events...")

    # Give event bus time to process remaining events
    await asyncio.sleep(0.5)

    # Stop system
    await system.stop()

    logger.info("[run_evaluation] TradingSystem stopped, calculating metrics...")

    # Build metrics from TradingSystem's accounting
    # Get strategies dict for metrics
    strategies_dict: dict[str, BaseStrategy] = {}
    if system._impulse_engine:
        strategies_dict["impulse_engine"] = system._impulse_engine
    for strategy in system._strategies:
        if strategy.strategy_id not in strategies_dict:
            strategies_dict[strategy.strategy_id] = strategy

    # Get paper broker from execution router
    paper_broker = system._execution_router._paper_broker

    metrics_calc = BacktestMetrics(
        accounting=system._accounting,
        paper_broker=paper_broker,
        strategies=strategies_dict,
        allocator=system._allocator,
    )

    metrics = metrics_calc.calculate()

    # Check metrics and log warnings instead of raising errors
    total_trades = metrics.get("total_trades", 0)
    total_pnl = metrics.get("total_pnl", 0.0)
    profit_factor = metrics.get("profit_factor", 0.0)

    if total_trades < 5:
        logger.warning(
            f"[METRICS_WARNING] total_trades={total_trades} < 5. "
            f"Expected at least 5 trades in evaluation."
        )
    else:
        logger.info(f"[METRICS] total_trades={total_trades}")

    if total_pnl <= 0:
        logger.warning(
            f"[METRICS_WARNING] total_pnl={total_pnl:.2f} <= 0. "
            f"Expected positive PnL in evaluation."
        )
    else:
        logger.info(f"[METRICS] total_pnl={total_pnl:.2f}")

    if profit_factor <= 1.0:
        logger.warning(
            f"[METRICS_WARNING] profit_factor={profit_factor:.2f} <= 1.0. "
            f"Expected profit_factor > 1.0 in evaluation."
        )
    else:
        logger.info(f"[METRICS] profit_factor={profit_factor:.2f}")

    logger.info(
        f"[METRICS_SUMMARY] total_trades={total_trades}, "
        f"total_pnl={total_pnl:.2f}, profit_factor={profit_factor:.2f}"
    )

    # Evaluate readiness
    evaluator = ReadinessEvaluator()
    result = evaluator.evaluate(metrics)

    # Print readiness report
    logger.info("[run_evaluation] Printing readiness report...")
    evaluator.print_report(result)

    # Print no-trade / signal block summary
    summary = no_trade_report.get_summary()
    if summary.totals:
        logger.info("[no_trade] Evaluation no-trade totals: %s", summary.totals)
    if summary.per_strategy:
        logger.info("[no_trade] Evaluation no-trade per-strategy: %s", summary.per_strategy)

    # DEBUG: Print forced order flow metrics
    logger.info(
        f"[DEBUG_METRICS] signals_generated={system._signals_generated} "
        f"signals_after_filters={system._signals_after_filters} "
        f"orders_sent={system._orders_sent} "
        f"orders_filled={system._orders_filled}"
    )

    # Run truth layer / self-explanation using metrics, readiness result, and no-trade stats
    diagnosis = analyze_truth(metrics, summary, result)
    print_truth(diagnosis)

    logger.info("[run_evaluation] Evaluation completed")

    return {
        "passed": result.passed,
        "criteria": result.criteria,
        "recommendations": result.recommendations,
        "regime_results": result.regime_results,
    }


async def run_backtest(days: int, output_file: str | None = None) -> None:
    """Run a backtest.

    ARCHITECTURAL FIX: Now uses TradingSystem like evaluation to ensure
    signals go through the full pipeline (signal → order → fill) with
    proper tracing and retry queue support.
    """
    logger.info(f"[run_backtest] Starting backtest for {days} days...")

    # Enable debug mode for backtest to allow trades to execute
    original_debug_mode = settings.debug_mode
    settings.debug_mode = True
    logger.info("[run_backtest] Debug mode enabled for backtest")

    # Reset no-trade counters at the start of each backtest run
    no_trade_report.reset()

    # Create EventSimulator as price feed
    start_date = datetime.utcnow() - timedelta(days=days)
    simulator = EventSimulator(None, ["BTCUSDT", "ETHUSDT"], start_date, days)

    # Create TradingSystem (use "evaluate" mode to skip health check)
    system = TradingSystem(mode="evaluate")

    # Start system with EventSimulator as price feed
    await system.start(price_feed=simulator)
    logger.info("[run_backtest] TradingSystem started, beginning simulation...")

    # Run simulation (EventSimulator will publish ticks)
    await simulator.run()
    logger.info("[run_backtest] Simulation completed, processing remaining events...")

    # Process remaining events in batches
    max_batch_size = 10000  # Increased batch size for faster processing
    max_iterations = 5000  # Increased iterations to process more events
    total_processed = 0
    for iteration in range(max_iterations):
        queue_size = system._bus._queue.qsize()
        if queue_size == 0:
            logger.info(
                f"[run_backtest] Event queue empty after processing {total_processed} events"
            )
            break
        processed = await system._bus.flush(max_events=max_batch_size)
        total_processed += processed
        if processed == 0:
            # No events processed, wait a bit for async handlers to complete
            await asyncio.sleep(0.05)  # Reduced wait time
        if iteration % 50 == 0:
            logger.info(
                f"[run_backtest] Processing events... queue_size={queue_size}, processed={total_processed}"
            )
    if system._bus._queue.qsize() > 0:
        remaining = system._bus._queue.qsize()
        logger.warning(
            f"[run_backtest] Still {remaining} events in queue after processing {total_processed} events"
        )
        # Try to process a few more batches of remaining events
        for _ in range(100):
            if system._bus._queue.qsize() == 0:
                break
            await system._bus.flush(max_events=10000)
            await asyncio.sleep(0.01)
    logger.info("[run_backtest] Calculating metrics...")

    # Stop system
    await system.stop()

    # Build metrics from TradingSystem's accounting
    strategies_dict: dict[str, BaseStrategy] = {}
    if system._impulse_engine:
        strategies_dict["impulse_engine"] = system._impulse_engine
    for strategy in system._strategies:
        if strategy.strategy_id not in strategies_dict:
            strategies_dict[strategy.strategy_id] = strategy

    # Get paper broker from execution router
    paper_broker = system._execution_router._paper_broker

    # DEBUG: Check paper broker stats before calculating metrics
    if paper_broker:
        fill_stats = paper_broker.get_fill_stats()
        logger.info(f"[DEBUG_BACKTEST] Paper broker fill stats: {fill_stats}")

    metrics_calc = BacktestMetrics(
        accounting=system._accounting,
        paper_broker=paper_broker,
        strategies=strategies_dict,
        allocator=system._allocator,
        execution_router=system._execution_router,
    )

    metrics = metrics_calc.calculate()

    # DEBUG: Log calculated metrics
    logger.info(
        f"[DEBUG_BACKTEST] Calculated metrics: total_trades={metrics.get('total_trades', 0)}"
    )

    # Check metrics and log warnings instead of raising errors
    total_trades = metrics.get("total_trades", 0)
    total_pnl = metrics.get("total_pnl", 0.0)
    profit_factor = metrics.get("profit_factor", 0.0)

    if total_trades < 5:
        logger.warning(
            f"[METRICS_WARNING] total_trades={total_trades} < 5. "
            f"Expected at least 5 trades in backtest."
        )
    else:
        logger.info(f"[METRICS] total_trades={total_trades}")

    if total_pnl <= 0:
        logger.warning(
            f"[METRICS_WARNING] total_pnl={total_pnl:.2f} <= 0. "
            f"Expected positive PnL in backtest."
        )
    else:
        logger.info(f"[METRICS] total_pnl={total_pnl:.2f}")

    if profit_factor <= 1.0:
        logger.warning(
            f"[METRICS_WARNING] profit_factor={profit_factor:.2f} <= 1.0. "
            f"Expected profit_factor > 1.0 in backtest."
        )
    else:
        logger.info(f"[METRICS] profit_factor={profit_factor:.2f}")

    logger.info(
        f"[METRICS_SUMMARY] total_trades={total_trades}, "
        f"total_pnl={total_pnl:.2f}, profit_factor={profit_factor:.2f}"
    )

    logger.info("[run_backtest] Printing backtest report...")
    metrics_calc.print_report(metrics)

    # Print no-trade / signal block summary
    summary = no_trade_report.get_summary()
    if summary.totals:
        logger.info("[no_trade] Backtest no-trade totals: %s", summary.totals)
    if summary.per_strategy:
        logger.info("[no_trade] Backtest no-trade per-strategy: %s", summary.per_strategy)

    # DEBUG: Print forced order flow metrics
    logger.info(
        f"[DEBUG_METRICS] signals_generated={system._signals_generated} "
        f"signals_after_filters={system._signals_after_filters} "
        f"orders_sent={system._orders_sent} "
        f"orders_filled={system._orders_filled}"
    )

    # Save JSON if requested
    if output_file:
        logger.info(f"[run_backtest] Saving metrics to {output_file}...")
        metrics_calc.save_json(output_file)

    logger.info("[run_backtest] Backtest completed")

    # Restore original debug mode
    settings.debug_mode = original_debug_mode
    logger.info(f"[run_backtest] Debug mode restored to {original_debug_mode}")


def main() -> None:
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="HEAN Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("run", help="Run the trading system")
    subparsers.add_parser("report", help="Show trading diagnostics report (attempts, blocks, reasons)")
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest")
    backtest_parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    backtest_parser.add_argument(
        "--out", type=str, default=None, help="Output JSON report file path"
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate system readiness")
    evaluate_parser.add_argument("--days", type=int, default=60, help="Number of days to evaluate")

    # Process Factory commands (experimental)
    process_parser = subparsers.add_parser("process", help="Process Factory commands (experimental)")
    process_subparsers = process_parser.add_subparsers(dest="process_command", help="Process Factory command")

    process_subparsers.add_parser(
        "scan",
        help="Scan environment and create snapshot. Prints snapshot ID and staleness warnings.",
    )
    process_plan_parser = process_subparsers.add_parser(
        "plan",
        help="Plan daily capital allocation. Prints top opportunities, capital plan, and selection rationale.",
    )
    process_plan_parser.add_argument(
        "--capital", type=float, default=400.0, help="Total capital in USD"
    )
    process_run_parser = process_subparsers.add_parser(
        "run",
        help="Run a process. Prints net contribution and kill/scale suggestions.",
    )
    process_run_parser.add_argument(
        "--process-id", type=str, required=True, help="Process ID to run"
    )
    process_run_parser.add_argument(
        "--mode",
        type=str,
        default="sandbox",
        choices=["sandbox", "live"],
        help="Execution mode",
    )
    process_run_parser.add_argument(
        "--capital", type=float, default=0.0, help="Capital allocated in USD"
    )
    process_run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even if daily_run_key exists (bypass idempotency)",
    )
    process_subparsers.add_parser(
        "report",
        help="Generate daily report. Includes top contributors (net), profit illusion list, portfolio health score.",
    )
    process_evaluate_parser = process_subparsers.add_parser(
        "evaluate",
        help="Evaluate portfolio over date range. Replays stored runs and prints stable metrics.",
    )
    process_evaluate_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to look back (default 30)"
    )
    process_subparsers.add_parser(
        "exec-smoke-test",
        help="Run execution smoke test (place and cancel a small test order)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Validate configuration
    if settings.trading_mode == "live" and not settings.is_live:
        logger.error("Live trading requires LIVE_CONFIRM=YES")
        sys.exit(1)

    if args.command == "run":
        system = TradingSystem()
        loop = asyncio.get_event_loop()

        # Handle signals for graceful shutdown
        def signal_handler(sig, frame):
            sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
            logger.critical(f"Received {sig_name}, initiating graceful shutdown...")

            # Create async shutdown task
            async def graceful_shutdown():
                try:
                    # Panic close all positions first
                    logger.critical("EMERGENCY: Closing all positions...")
                    close_result = await system.panic_close_all(reason=f"graceful_shutdown_{sig_name}")
                    logger.critical(
                        f"Panic close complete: closed={close_result.get('positions_closed', 0)}, "
                        f"cancelled={close_result.get('orders_cancelled', 0)}"
                    )

                    # Stop the trading system
                    logger.info("Stopping trading system...")
                    await system.stop()
                    logger.info("Trading system stopped gracefully")

                except Exception as e:
                    logger.error(f"Error during graceful shutdown: {e}", exc_info=True)
                finally:
                    # Stop the event loop
                    loop.stop()

            # Schedule graceful shutdown
            loop.create_task(graceful_shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers installed for graceful shutdown")

        try:
            loop.run_until_complete(system.run())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    elif args.command == "report":
        # Show trading diagnostics report
        from hean.observability.no_trade_report import no_trade_report

        summary = no_trade_report.get_summary()

        print("=" * 70)
        print("📊 TRADING DIAGNOSTICS REPORT")
        print("=" * 70)
        print()

        # Pipeline counters
        pipeline = summary.pipeline_counters
        print("📈 PIPELINE COUNTERS:")
        print(f"  Signals emitted: {pipeline.get('signals_emitted', 0)}")
        print(f"  Orders created: {pipeline.get('orders_created', 0)}")
        print(f"  Positions opened: {pipeline.get('positions_opened', 0)}")
        print(f"  Positions closed: {pipeline.get('positions_closed', 0)}")
        print()

        # Block reasons
        totals = summary.totals
        if totals:
            print("🚫 BLOCK REASONS (Total):")
            sorted_reasons = sorted(totals.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_reasons[:10]:  # Top 10
                print(f"  {reason}: {count}")
            print()

        # Per strategy
        per_strategy = summary.per_strategy
        if per_strategy:
            print("📊 PER STRATEGY:")
            for strategy_id, reasons in per_strategy.items():
                total = sum(reasons.values())
                if total > 0:
                    print(f"  {strategy_id}: {total} blocks")
                    top_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                    for reason, count in top_reasons:
                        print(f"    - {reason}: {count}")
            print()

        # Per symbol
        per_symbol = summary.per_symbol
        if per_symbol:
            print("📊 PER SYMBOL:")
            for symbol, reasons in per_symbol.items():
                total = sum(reasons.values())
                if total > 0:
                    print(f"  {symbol}: {total} blocks")
                    top_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:3]
                    for reason, count in top_reasons:
                        print(f"    - {reason}: {count}")
            print()

        # Paper assist status
        from hean.paper_trade_assist import is_paper_assist_enabled
        if is_paper_assist_enabled():
            print("✅ PAPER_TRADE_ASSIST: ENABLED")
            print(f"  Micro-trade interval: {settings.paper_trade_assist_micro_trade_interval_sec}s")
            print(f"  Micro-trade notional: ${settings.paper_trade_assist_micro_trade_notional_usd}")
            print()

        print("=" * 70)

    elif args.command == "backtest":
        asyncio.run(run_backtest(args.days, args.out))

    elif args.command == "evaluate":
        result = asyncio.run(run_evaluation(args.days))
        # Exit with code 0 if passed, 1 if failed
        sys.exit(0 if result["passed"] else 1)

    elif args.command == "process":
        # Process Factory commands (experimental)
        if not settings.process_factory_enabled:
            logger.error(
                "Process Factory is disabled. Enable it in config with process_factory_enabled=true"
            )
            sys.exit(1)

        from hean.process_factory.engine import ProcessEngine
        from hean.process_factory.report import ProcessReportGenerator
        from hean.process_factory.storage import SQLiteStorage

        storage = SQLiteStorage("process_factory.db")
        engine = ProcessEngine(storage)

        try:
            if args.process_command == "scan":
                async def scan_with_output():
                    snapshot = await engine.scan_environment()
                    print("\n✓ Environment scan completed")
                    print(f"  Snapshot ID: {snapshot.snapshot_id}")
                    print(f"  Timestamp: {snapshot.timestamp.isoformat()}")
                    if snapshot.is_stale():
                        print(f"  ⚠ WARNING: Snapshot is stale ({snapshot.staleness_hours:.1f} hours old)")
                    else:
                        print(f"  ✓ Snapshot is fresh ({snapshot.staleness_hours:.1f} hours old)")
                    print(f"  Balances: {len(snapshot.balances)} assets")
                    print(f"  Positions: {len(snapshot.positions)}")
                    print(f"  Funding rates: {len(snapshot.funding_rates)} symbols")
                    return snapshot
                asyncio.run(scan_with_output())

            elif args.process_command == "plan":
                async def plan_with_output():
                    ranked, plan = await engine.plan(args.capital)
                    print(f"\n✓ Planning completed: {len(ranked)} opportunities")
                    print("\n📊 Capital Plan:")
                    print(f"  Reserve: ${plan.reserve_usd:.2f} ({plan.reserve_usd/plan.total_capital_usd*100:.1f}%)")
                    print(f"  Active: ${plan.active_usd:.2f} ({plan.active_usd/plan.total_capital_usd*100:.1f}%)")
                    print(f"  Experimental: ${plan.experimental_usd:.2f} ({plan.experimental_usd/plan.total_capital_usd*100:.1f}%)")
                    print("\n🎯 Top Opportunities:")
                    for i, (opp, _score) in enumerate(ranked[:5], 1):
                        print(f"  {i}. {opp.id}: ${opp.expected_profit_usd:.2f} profit, {opp.time_hours:.1f}h, risk={opp.risk_factor:.1f}")
                    print("\n💰 Allocations:")
                    for process_id, allocation in sorted(plan.allocations.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {process_id}: ${allocation:.2f}")
                    return ranked, plan
                asyncio.run(plan_with_output())

            elif args.process_command == "run":
                async def run_with_output():
                    try:
                        run = await engine.run_process(
                            args.process_id, {}, args.mode, args.capital, force=args.force
                        )
                    except ValueError as e:
                        if "stale" in str(e).lower():
                            print(f"\n❌ {e}")
                            sys.exit(1)
                        raise
                    from hean.process_factory.truth_layer import TruthLayer
                    truth_layer = TruthLayer()
                    attribution = truth_layer.compute_attribution(run)

                    print(f"\n✓ Process run completed: {run.run_id}")
                    print(f"  Status: {run.status.value}")
                    print(f"  Process: {run.process_id}")
                    print(f"  Mode: {args.mode}")
                    print("\n📊 Attribution:")
                    print(f"  Gross PnL: ${attribution.gross_pnl_usd:.2f}")
                    print(f"  Net PnL: ${attribution.net_pnl_usd:.2f}")
                    print(f"  Fees: ${attribution.total_fees_usd:.2f}")
                    print(f"  Funding: ${attribution.total_funding_usd:.2f}")
                    print(f"  Rewards: ${attribution.total_rewards_usd:.2f}")
                    print(f"  Opportunity Cost: ${attribution.opportunity_cost_usd:.2f}")
                    if attribution.profit_illusion:
                        print("  ⚠ Profit Illusion: Gross positive but net negative!")

                    # Get kill/scale suggestions
                    portfolio = await engine.update_portfolio()
                    entry = next((e for e in portfolio if e.process_id == run.process_id), None)
                    if entry:
                        print("\n💡 Recommendations:")
                        if entry.state.value == "KILLED":
                            print("  ❌ Process is KILLED")
                        elif entry.fail_rate > 0.7:
                            print(f"  ⚠ High fail rate ({entry.fail_rate:.1%}), consider killing")
                        elif entry.runs_count >= 5 and entry.avg_roi > 0 and entry.fail_rate < 0.4:
                            print("  ✓ Good performance, eligible for scaling")
                        else:
                            print(f"  → Continue testing ({entry.runs_count} runs)")

                    if run.error:
                        print(f"\n❌ Error: {run.error}")
                    return run
                asyncio.run(run_with_output())

            elif args.process_command == "report":
                async def generate_report():
                    portfolio = await engine.update_portfolio()
                    capital_plan = await storage.load_latest_capital_plan()
                    recent_runs = await storage.list_runs(limit=50)

                    # Compute attribution for profit illusion detection
                    from hean.process_factory.truth_layer import TruthLayer
                    truth_layer = TruthLayer()
                    attributions = truth_layer.compute_portfolio_attribution(recent_runs)

                    generator = ProcessReportGenerator()
                    result = generator.generate_daily_report(
                        portfolio, capital_plan, recent_runs
                    )

                    if result is None:
                        # Idempotent: report already exists
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        md_path = generator.output_dir / f"process_factory_report_{date_str}.md"
                        json_path = generator.output_dir / f"process_factory_report_{date_str}.json"
                        print("\n✓ Report already exists (idempotent skip):")
                        print(f"  Markdown: {md_path}")
                        print(f"  JSON: {json_path}")
                        return None, None

                    md_path, json_path = result
                    print("\n✓ Report generated:")
                    print(f"  Markdown: {md_path}")
                    print(f"  JSON: {json_path}")

                    # Print summary
                    print("\n📊 Portfolio Summary:")
                    total_net = sum(a.net_pnl_usd for a in attributions.values())
                    total_gross = sum(a.gross_pnl_usd for a in attributions.values())
                    profit_illusion_count = sum(1 for a in attributions.values() if a.profit_illusion)
                    print(f"  Total Net Contribution: ${total_net:.2f}")
                    print(f"  Total Gross PnL: ${total_gross:.2f}")
                    print(f"  Profit Illusion Processes: {profit_illusion_count}")

                    # Top contributors (net)
                    sorted_attributions = sorted(
                        attributions.items(), key=lambda x: x[1].net_pnl_usd, reverse=True
                    )[:5]
                    print("\n🏆 Top Contributors (Net):")
                    for process_id, attr in sorted_attributions:
                        if attr.net_pnl_usd > 0:
                            print(f"  {process_id}: ${attr.net_pnl_usd:.2f} net")

                    # Profit illusion list
                    if profit_illusion_count > 0:
                        print("\n⚠ Profit Illusion Processes:")
                        for process_id, attr in attributions.items():
                            if attr.profit_illusion:
                                print(f"  {process_id}: ${attr.gross_pnl_usd:.2f} gross → ${attr.net_pnl_usd:.2f} net")

                    # Portfolio health
                    from hean.process_factory.evaluation import PortfolioEvaluator
                    evaluator = PortfolioEvaluator(storage)
                    health_score, _ = await evaluator.evaluate_portfolio(days=30)
                    print("\n💚 Portfolio Health Score:")
                    print(f"  Stability: {health_score.stability_score:.2%}")
                    print(f"  Concentration Risk: {health_score.concentration_risk:.2%}")
                    print(f"  Churn Rate: {health_score.churn_rate:.2f}")
                    print(f"  Core Processes: {health_score.core_process_count}")
                    print(f"  Testing Processes: {health_score.testing_process_count}")
                    print(f"  Killed Processes: {health_score.killed_process_count}")

                    return md_path, json_path

                asyncio.run(generate_report())

            elif args.process_command == "evaluate":
                async def evaluate_with_output():
                    from hean.process_factory.evaluation import PortfolioEvaluator
                    evaluator = PortfolioEvaluator(storage)
                    health_score, process_results = await evaluator.evaluate_portfolio(
                        days=args.days
                    )

                    print(f"\n✓ Portfolio Evaluation ({args.days} days)")
                    print("\n💚 Portfolio Health Score:")
                    print(f"  Stability: {health_score.stability_score:.2%}")
                    print(f"  Concentration Risk: {health_score.concentration_risk:.2%}")
                    print(f"  Churn Rate: {health_score.churn_rate:.2f}")
                    print(f"  Net Contribution: ${health_score.net_contribution_usd:.2f}")
                    print(f"  Profit Illusion Count: {health_score.profit_illusion_count}")
                    print(f"  Core: {health_score.core_process_count}, Testing: {health_score.testing_process_count}, Killed: {health_score.killed_process_count}")

                    print("\n📋 Process Recommendations:")
                    for result in sorted(process_results, key=lambda x: x.net_contribution_usd, reverse=True):
                        print(f"\n  {result.process_id}:")
                        print(f"    Recommendation: {result.recommendation}")
                        print(f"    Net Contribution: ${result.net_contribution_usd:.2f}")
                        print(f"    Runs: {result.runs_count}, Win Rate: {result.win_rate:.1%}, Stability: {result.stability:.1%}")
                        if result.reasons:
                            print(f"    Reasons: {', '.join(result.reasons)}")

                    return health_score, process_results

                asyncio.run(evaluate_with_output())

            elif args.process_command == "exec-smoke-test":
                from hean.process_factory.integrations.smoke_test import run_smoke_test

                async def smoke_test_with_output():
                    try:
                        result = await run_smoke_test()
                    except ValueError as e:
                        print(f"\n❌ Validation error: {e}")
                        sys.exit(1)

                    if result["success"]:
                        print("\n✓ SUCCESS: Execution smoke test passed")
                        print(f"  Order ID: {result['order_id']}")
                        print(f"  Symbol: {result['symbol']}")
                        print(f"  Side: {result['side']}")
                        print(f"  Quantity: {result['qty']:.6f}")
                        print(f"  Price: ${result['price']:.2f}")
                        print("\n  Order placed and cancelled successfully.")
                    else:
                        print("\n❌ FAIL: Execution smoke test failed")
                        print(f"  Error type: {result.get('error_type', 'unknown')}")
                        print(f"  Error: {result.get('error', 'unknown error')}")
                        print("\n  Suggested fixes:")
                        if result.get('error_type') == 'not_enabled':
                            print("    - Set PROCESS_FACTORY_ALLOW_ACTIONS=true")
                            print("    - Set DRY_RUN=false")
                        elif result.get('error_type') == 'min_notional':
                            print(f"    - Increase EXECUTION_SMOKE_TEST_NOTIONAL_USD (min ${result.get('error', '').split()[-1] if '$' in result.get('error', '') else '5'})")
                        elif result.get('error_type') == 'validation_error':
                            print("    - Check configuration flags")
                        else:
                            print("    - Check API credentials (BYBIT_API_KEY, BYBIT_API_SECRET)")
                            print("    - Check network connectivity")
                            print(f"    - Verify symbol is valid: {result.get('symbol', 'unknown')}")
                        sys.exit(1)

                    return result

                asyncio.run(smoke_test_with_output())

            else:
                process_parser.print_help()
                sys.exit(1)

        finally:
            asyncio.run(storage.close())

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
