"""Main entrypoint for HEAN system."""

import argparse
import asyncio
import random
import signal
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from hean.agent_generation.capital_optimizer import CapitalOptimizer
from hean.agent_generation.catalyst import ImprovementCatalyst
from hean.agent_generation.report_generator import ReportGenerator
from hean.backtest.event_sim import EventSimulator
from hean.backtest.metrics import BacktestMetrics
from hean.config import settings
from hean.core.bus import EventBus
from hean.core.clock import Clock
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
from hean.exchange.synthetic_feed import SyntheticPriceFeed
from hean.execution.order_manager import OrderManager
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
from hean.strategies.base import BaseStrategy
from hean.strategies.basis_arbitrage import BasisArbitrage
from hean.strategies.funding_harvester import FundingHarvester
from hean.strategies.impulse_engine import ImpulseEngine
from hean.core.intelligence.correlation_engine import CorrelationEngine
from hean.risk.tail_risk import GlobalSafetyNet
from hean.observability.monitoring.self_healing import SelfHealingMiddleware
from hean.core.intelligence.meta_learning_engine import MetaLearningEngine
from hean.core.intelligence.causal_inference_engine import CausalInferenceEngine
from hean.core.intelligence.multimodal_swarm import MultimodalSwarm
from hean.core.intelligence.metamorphic_integration import MetamorphicIntegration
from hean.core.intelligence.causal_discovery import CausalDiscoveryEngine
from hean.execution.atomic_executor import AtomicExecutor
from hean.core.arb.triangular_scanner import TriangularScanner

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
        self._price_feed: SyntheticPriceFeed | None = None
        self._strategies: list = []
        self._income_streams: list = []
        self._candle_aggregator: CandleAggregator | None = None

        # Auto-improvement systems
        self._improvement_catalyst: ImprovementCatalyst | None = None
        self._capital_optimizer = CapitalOptimizer()
        self._report_generator = ReportGenerator()

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
                       instead of creating SyntheticPriceFeed. Used for evaluation mode
                       with EventSimulator.
        """
        logger.info("Starting HEAN trading system...")
        logger.info(f"Trading mode: {settings.trading_mode}")
        logger.info(f"DRY_RUN: {settings.dry_run}")
        logger.info(f"bybit_testnet: {settings.bybit_testnet}")
        logger.info(f"PAPER_TRADE_ASSIST: {settings.paper_trade_assist}")
        logger.info(f"LIVE_CONFIRM: {settings.live_confirm}")
        initial_capital = (
            settings.backtest_initial_capital
            if self._mode == "evaluate"
            else settings.initial_capital
        )
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

        # Only start HealthCheck in run mode
        if self._health_check:
            await self._health_check.start()

        # Start regime detector
        await self._regime_detector.start()
        self._bus.subscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        
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
                from hean.risk.kelly_criterion import KellyCriterion
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
            # 1. Meta-Learning Engine (Recursive Intelligence Core)
            try:
                cpp_source_dir = Path(__file__).parent.parent.parent / "src" / "hean" / "core" / "cpp"
                self._meta_learning_engine = MetaLearningEngine(
                    bus=self._bus,
                    cpp_source_dir=cpp_source_dir,
                    simulation_rate=getattr(settings, 'meta_learning_rate', 1_000_000),
                    auto_patch_enabled=getattr(settings, 'meta_learning_auto_patch', True),
                    max_concurrent_simulations=getattr(settings, 'meta_learning_max_workers', 100)
                )
                await self._meta_learning_engine.start()
                logger.info("⚡ ABSOLUTE+: Meta-Learning Engine started (Recursive Intelligence Core)")
            except Exception as e:
                logger.warning(f"Failed to start Meta-Learning Engine: {e}")
            
            # 2. Causal Inference Engine
            try:
                # Source symbols for cross-asset pre-echo detection
                source_symbols = getattr(settings, 'causal_source_symbols', [
                    "BTCUSDT", "ETHUSDT", "BNBUSDT",  # Other exchanges could be added
                    "SPX", "NDX",  # Stock indices (if available)
                    "DXY"  # Dollar index
                ])
                self._causal_inference_engine = CausalInferenceEngine(
                    bus=self._bus,
                    target_symbols=settings.trading_symbols,
                    source_symbols=source_symbols,
                    window_size=getattr(settings, 'causal_window_size', 500),
                    min_causality_threshold=getattr(settings, 'causal_min_threshold', 0.3),
                    min_transfer_entropy=getattr(settings, 'causal_min_te', 0.1)
                )
                await self._causal_inference_engine.start()
                logger.info("🔮 ABSOLUTE+: Causal Inference Engine started (Granger Causality + Transfer Entropy)")
            except Exception as e:
                logger.warning(f"Failed to start Causal Inference Engine: {e}")
            
            # 3. Multimodal Swarm (Unified Tensor Processing)
            try:
                self._multimodal_swarm = MultimodalSwarm(
                    bus=self._bus,
                    symbols=settings.trading_symbols,
                    window_size=getattr(settings, 'multimodal_window_size', 100),
                    num_agents=getattr(settings, 'multimodal_num_agents', 50)
                )
                await self._multimodal_swarm.start()
                logger.info("🌐 ABSOLUTE+: Multimodal Swarm started (Price + Sentiment + On-Chain + Macro)")
            except Exception as e:
                logger.warning(f"Failed to start Multimodal Swarm: {e}")

        # Start price feed
        symbols = settings.trading_symbols
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
                    # Bybit returns: {"result": {"list": [{"coin": [{"walletBalance": "...", "coin": "USDT"}]}]}}
                    balance = 0.0
                    if account_info.get("retCode") == 0:
                        result = account_info.get("result", {})
                        account_list = result.get("list", [])
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
                        logger.info(f"✅ Synced with exchange balance: ${balance:,.2f} USDT")
                        logger.info("📊 System ready to trade with any capital amount")
                    else:
                        logger.warning(f"⚠️ Could not get balance from exchange, using config: ${initial_capital:,.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not sync with exchange balance: {e}")
                    logger.info(f"Using configured initial capital: ${initial_capital:,.2f}")
            except Exception as e:
                logger.warning(f"MARKET_STREAM_FALLBACK: Bybit price feed failed, switching to synthetic. Error: {e}", exc_info=True)
                self._price_feed = SyntheticPriceFeed(self._bus, symbols)
                await self._price_feed.start()
                logger.info(f"MARKET_STREAM_STARTED: Synthetic price feed for symbols {symbols}")
        else:
            # Create default synthetic price feed for paper trading
            self._price_feed = SyntheticPriceFeed(self._bus, symbols)
            await self._price_feed.start()
            logger.info(f"MARKET_STREAM_STARTED: Synthetic price feed for symbols {symbols}")

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
            strategy = ImpulseEngine(self._bus, symbols)
            await strategy.start()
            self._strategies.append(strategy)
            self._impulse_engine = strategy  # Store reference for metrics
        else:
            self._impulse_engine = None

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

        # Schedule periodic status updates (skip in evaluate mode)
        if self._mode == "run":
            self._clock.schedule_periodic(self._print_status, timedelta(seconds=10))
            # Anti-stuck: periodic TTL/NO_TICKS checks even if feed stalls
            self._clock.schedule_periodic(self._check_position_timeouts, timedelta(seconds=5))
            
            # Start fallback micro-trade task if paper assist enabled
            if is_paper_assist_enabled():
                self._micro_trade_task = asyncio.create_task(self._micro_trade_fallback_loop())
                logger.info("Paper Trade Assist: Fallback micro-trade loop started")

        # Start improvement catalyst (only in run mode, with LLM enabled)
        if self._mode == "run":
            try:
                # Create strategy dict for catalyst
                strategy_dict = {strategy.strategy_id: strategy for strategy in self._strategies}
                self._improvement_catalyst = ImprovementCatalyst(
                    accounting=self._accounting,
                    strategies=strategy_dict,
                    check_interval_minutes=30,  # Check every 30 minutes
                    min_trades_for_analysis=10,
                )
                await self._improvement_catalyst.start()
                logger.info("Improvement Catalyst started")
            except Exception as e:
                logger.warning(f"Could not start Improvement Catalyst: {e}")

        self._running = True
        logger.info("Trading system started")

    async def stop(self) -> None:
        """Stop the trading system."""
        logger.info("Stopping trading system...")
        self._running = False

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
        
        # Track signal attempt for diagnostics
        signal_attempt_key = f"{signal.strategy_id}:{signal.symbol}"

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

        # МНОЖЕСТВЕННАЯ ЗАЩИТА
        protection_size_multiplier = 1.0
        if not settings.debug_mode:
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
        else:
            logger.debug(f"[DEBUG] Protection bypassed for {signal.symbol} {signal.strategy_id}")

        # Get equity for later use
        equity = self._accounting.get_equity()

        # Capital Preservation Mode
        if not settings.debug_mode:
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
                # Применить консервативные параметры
                if signal.metadata is None:
                    signal.metadata = {}
                signal.metadata["preservation_mode"] = True
                signal.metadata["risk_multiplier"] = preservation_mode.get_risk_pct(1.0)
                logger.warning(
                    f"Capital Preservation Mode active: drawdown={drawdown_pct:.1f}%, "
                    f"PF={rolling_pf:.2f}, consecutive_losses={consecutive_losses}"
                )
        else:
            logger.debug(
                f"[DEBUG] Capital Preservation Mode bypassed for {signal.symbol} {signal.strategy_id}"
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
        
        # Check risk limits with calculated size
        if not settings.debug_mode:
            allowed, reason = self._risk_limits.check_order_request(
                OrderRequest(
                    signal_id=str(uuid.uuid4()),
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    side=signal.side,
                    size=base_size,  # Use calculated size
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
        else:
            logger.debug(f"[DEBUG] Risk limits bypassed for {signal.symbol} {signal.strategy_id}")

        # Check daily attempts and cooldown
        if not settings.debug_mode:
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

            # Check cooldown (bypassed in DEBUG_MODE / Aggressive Mode)
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
        else:
            logger.debug(
                f"[DEBUG] Daily attempts and cooldown bypassed for {signal.symbol} {signal.strategy_id}"
            )

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

        # Check if context is blocked
        has_dm_history = self._decision_memory.has_history(signal.strategy_id, context)
        if not settings.debug_mode and has_dm_history:
            if self._decision_memory.blocked(signal.strategy_id, context):
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
        else:
            logger.debug(
                f"[DEBUG] Decision memory bypassed for {signal.symbol} {signal.strategy_id}"
            )

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
        if not settings.debug_mode:
            penalty_multiplier = self._decision_memory.penalty(signal.strategy_id, context)
        else:
            penalty_multiplier = 1.0  # DEBUG: No penalty
            logger.debug("[DEBUG] Penalty multiplier bypassed, using 1.0")

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
        order: Order = event.data["order"]
        fill_price = event.data["fill_price"]
        event.data["fill_size"]
        fee = event.data["fee"]

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

        # Update accounting
        self._accounting.record_fill(order, fill_price, fee)
        logger.info(f"[ORDER_FILLED_HANDLER] Accounting updated for order {order.order_id}")

        # Create position if fully filled
        if order.status == OrderStatus.FILLED and order.avg_fill_price:
            logger.info(
                f"[ORDER_FILLED_HANDLER] Order {order.order_id} is fully filled, "
                f"creating position: avg_fill_price={order.avg_fill_price:.2f}"
            )
            # Get take_profit_1 from signal metadata if available
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
            logger.info(
                f"[ORDER_FILLED_HANDLER] POSITION_OPENED event published for {position.position_id}"
            )

            metrics.increment("positions_opened")
            no_trade_report.increment_pipeline("positions_opened", order.strategy_id)
        else:
            logger.warning(
                f"[ORDER_FILLED_HANDLER] Order {order.order_id} not fully filled or missing avg_fill_price: "
                f"status={order.status}, avg_fill_price={order.avg_fill_price}"
            )
        await self._emit_account_state()

    async def _handle_position_closed(self, event: Event) -> None:
        """Handle position closed event."""
        position: Position = event.data["position"]
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
        wallet_balance = self._accounting.get_cash_balance() + used_margin
        available_balance = wallet_balance - used_margin - reserved_margin

        account_state = {
            "wallet_balance": wallet_balance,
            "available_balance": available_balance,
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
        """Emit periodic micro-trade signals for paper assist mode."""
        interval_sec = settings.paper_trade_assist_micro_trade_interval_sec
        notional_usd = settings.paper_trade_assist_micro_trade_notional_usd
        tp_pct = settings.paper_trade_assist_micro_trade_tp_pct / 100.0
        sl_pct = settings.paper_trade_assist_micro_trade_sl_pct / 100.0

        while self._running and is_paper_assist_enabled():
            try:
                await asyncio.sleep(interval_sec)

                if self._stop_trading:
                    continue

                prices = getattr(self._execution_router, "_current_prices", {})
                if not prices:
                    continue

                symbol = random.choice(list(prices.keys()))
                price = prices.get(symbol)
                if not price:
                    continue

                last_time = self._last_micro_trade_time.get(symbol)
                now = datetime.utcnow()
                if last_time and (now - last_time).total_seconds() < interval_sec:
                    continue

                side = random.choice(["buy", "sell"])
                size = max(0.000001, notional_usd / price)

                if side == "buy":
                    take_profit = price * (1 + tp_pct)
                    stop_loss = price * (1 - sl_pct)
                else:
                    take_profit = price * (1 - tp_pct)
                    stop_loss = price * (1 + sl_pct)

                signal = Signal(
                    strategy_id="paper_assist_micro",
                    symbol=symbol,
                    side=side,
                    entry_price=price,
                    size=size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "micro_trade": True,
                        "created_at": now.isoformat(),
                        "max_time_min": settings.paper_trade_assist_micro_trade_max_time_min,
                    },
                )

                await self._bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
                self._last_micro_trade_time[symbol] = now
                logger.info(
                    "[PAPER_ASSIST] Micro-trade signal: %s %s size=%.6f",
                    symbol,
                    side,
                    size,
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Micro-trade fallback loop error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

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
        # Update position price first
        self._accounting.update_position_price(position.position_id, close_price)

        # Calculate realized PnL
        if position.side == "long":
            realized_pnl = (close_price - position.entry_price) * position.size
        else:  # short
            realized_pnl = (position.entry_price - close_price) * position.size

        # Record realized PnL
        regime = self._current_regime.get(position.symbol, Regime.NORMAL)
        self._accounting.record_realized_pnl(realized_pnl, position.strategy_id, regime.value)

        # Update position
        position.current_price = close_price
        position.realized_pnl = realized_pnl

        # Publish position closed event
        await self._bus.publish(
            Event(
                event_type=EventType.POSITION_CLOSED,
                data={
                    "position": position,
                    "close_reason": reason,
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

        # Check killswitch (use most common regime or NORMAL)
        equity = snapshot.equity
        peak_equity = self._accounting._peak_equity
        if peak_equity is None:
            peak_equity = equity
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

    process_scan_parser = process_subparsers.add_parser(
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
    process_report_parser = process_subparsers.add_parser(
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
    process_smoke_test_parser = process_subparsers.add_parser(
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

        # Handle signals
        def signal_handler(sig, frame):
            logger.info("Received signal, shutting down...")
            loop.create_task(system.stop())
            loop.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

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
                    print(f"\n✓ Environment scan completed")
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
                    print(f"\n📊 Capital Plan:")
                    print(f"  Reserve: ${plan.reserve_usd:.2f} ({plan.reserve_usd/plan.total_capital_usd*100:.1f}%)")
                    print(f"  Active: ${plan.active_usd:.2f} ({plan.active_usd/plan.total_capital_usd*100:.1f}%)")
                    print(f"  Experimental: ${plan.experimental_usd:.2f} ({plan.experimental_usd/plan.total_capital_usd*100:.1f}%)")
                    print(f"\n🎯 Top Opportunities:")
                    for i, (opp, score) in enumerate(ranked[:5], 1):
                        print(f"  {i}. {opp.id}: ${opp.expected_profit_usd:.2f} profit, {opp.time_hours:.1f}h, risk={opp.risk_factor:.1f}")
                    print(f"\n💰 Allocations:")
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
                    print(f"\n📊 Attribution:")
                    print(f"  Gross PnL: ${attribution.gross_pnl_usd:.2f}")
                    print(f"  Net PnL: ${attribution.net_pnl_usd:.2f}")
                    print(f"  Fees: ${attribution.total_fees_usd:.2f}")
                    print(f"  Funding: ${attribution.total_funding_usd:.2f}")
                    print(f"  Rewards: ${attribution.total_rewards_usd:.2f}")
                    print(f"  Opportunity Cost: ${attribution.opportunity_cost_usd:.2f}")
                    if attribution.profit_illusion:
                        print(f"  ⚠ Profit Illusion: Gross positive but net negative!")
                    
                    # Get kill/scale suggestions
                    portfolio = await engine.update_portfolio()
                    entry = next((e for e in portfolio if e.process_id == run.process_id), None)
                    if entry:
                        print(f"\n💡 Recommendations:")
                        if entry.state.value == "KILLED":
                            print(f"  ❌ Process is KILLED")
                        elif entry.fail_rate > 0.7:
                            print(f"  ⚠ High fail rate ({entry.fail_rate:.1%}), consider killing")
                        elif entry.runs_count >= 5 and entry.avg_roi > 0 and entry.fail_rate < 0.4:
                            print(f"  ✓ Good performance, eligible for scaling")
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
                        print(f"\n✓ Report already exists (idempotent skip):")
                        print(f"  Markdown: {md_path}")
                        print(f"  JSON: {json_path}")
                        return None, None
                    
                    md_path, json_path = result
                    print(f"\n✓ Report generated:")
                    print(f"  Markdown: {md_path}")
                    print(f"  JSON: {json_path}")
                    
                    # Print summary
                    print(f"\n📊 Portfolio Summary:")
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
                    print(f"\n🏆 Top Contributors (Net):")
                    for process_id, attr in sorted_attributions:
                        if attr.net_pnl_usd > 0:
                            print(f"  {process_id}: ${attr.net_pnl_usd:.2f} net")
                    
                    # Profit illusion list
                    if profit_illusion_count > 0:
                        print(f"\n⚠ Profit Illusion Processes:")
                        for process_id, attr in attributions.items():
                            if attr.profit_illusion:
                                print(f"  {process_id}: ${attr.gross_pnl_usd:.2f} gross → ${attr.net_pnl_usd:.2f} net")
                    
                    # Portfolio health
                    from hean.process_factory.evaluation import PortfolioEvaluator
                    evaluator = PortfolioEvaluator(storage)
                    health_score, _ = await evaluator.evaluate_portfolio(days=30)
                    print(f"\n💚 Portfolio Health Score:")
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
                    print(f"\n💚 Portfolio Health Score:")
                    print(f"  Stability: {health_score.stability_score:.2%}")
                    print(f"  Concentration Risk: {health_score.concentration_risk:.2%}")
                    print(f"  Churn Rate: {health_score.churn_rate:.2f}")
                    print(f"  Net Contribution: ${health_score.net_contribution_usd:.2f}")
                    print(f"  Profit Illusion Count: {health_score.profit_illusion_count}")
                    print(f"  Core: {health_score.core_process_count}, Testing: {health_score.testing_process_count}, Killed: {health_score.killed_process_count}")
                    
                    print(f"\n📋 Process Recommendations:")
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
                        print(f"\n✓ SUCCESS: Execution smoke test passed")
                        print(f"  Order ID: {result['order_id']}")
                        print(f"  Symbol: {result['symbol']}")
                        print(f"  Side: {result['side']}")
                        print(f"  Quantity: {result['qty']:.6f}")
                        print(f"  Price: ${result['price']:.2f}")
                        print(f"\n  Order placed and cancelled successfully.")
                    else:
                        print(f"\n❌ FAIL: Execution smoke test failed")
                        print(f"  Error type: {result.get('error_type', 'unknown')}")
                        print(f"  Error: {result.get('error', 'unknown error')}")
                        print(f"\n  Suggested fixes:")
                        if result.get('error_type') == 'not_enabled':
                            print(f"    - Set PROCESS_FACTORY_ALLOW_ACTIONS=true")
                            print(f"    - Set DRY_RUN=false")
                        elif result.get('error_type') == 'min_notional':
                            print(f"    - Increase EXECUTION_SMOKE_TEST_NOTIONAL_USD (min ${result.get('error', '').split()[-1] if '$' in result.get('error', '') else '5'})")
                        elif result.get('error_type') == 'validation_error':
                            print(f"    - Check configuration flags")
                        else:
                            print(f"    - Check API credentials (BYBIT_API_KEY, BYBIT_API_SECRET)")
                            print(f"    - Check network connectivity")
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
