"""RL-based Risk Manager using trained PPO agent.

Dynamically adjusts:
- Leverage based on market conditions
- Position size multipliers
- Stop-loss placement

Integrates with Physics engine for market phase/temperature/entropy.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)

# Conditional import for stable-baselines3 (optional dependency)
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not available - RLRiskManager will use rule-based fallback")


class RLRiskManager:
    """RL-powered risk manager that adapts parameters based on market conditions.

    Uses a trained PPO agent to dynamically adjust:
    - Leverage (1.0x - 10.0x)
    - Position size multiplier (0.5x - 2.0x)
    - Stop loss distance (0.5% - 10.0%)

    Falls back to rule-based logic if model not available.
    """

    def __init__(
        self,
        bus: EventBus,
        model_path: str | None = None,
        adjustment_interval: int = 60,
        enabled: bool = True,
    ):
        """Initialize RL Risk Manager.

        Args:
            bus: Event bus for publishing risk adjustments
            model_path: Path to trained PPO model (.zip)
            adjustment_interval: Seconds between risk parameter adjustments
            enabled: Whether RL adjustments are active
        """
        self._bus = bus
        self._model_path = model_path
        self._adjustment_interval = adjustment_interval
        self._enabled = enabled

        # PPO model (loaded from checkpoint if available)
        self._model: Any = None
        if SB3_AVAILABLE and model_path and Path(model_path).exists():
            try:
                self._model = PPO.load(model_path)
                logger.info(f"Loaded RL risk model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load RL model: {e}")
                self._model = None

        # Current risk parameters (will be adjusted by RL agent)
        self._current_leverage = 3.0
        self._current_size_mult = 1.0
        self._current_stop_loss_pct = 2.0

        # Market state tracking (from EventBus)
        self._market_phase: dict[str, int] = {}  # symbol -> phase index
        self._market_temperature: dict[str, float] = {}
        self._market_entropy: dict[str, float] = {}
        self._market_volatility: dict[str, float] = {}

        # Performance tracking
        self._trades: list[dict[str, Any]] = []
        self._equity = 0.0
        self._initial_capital = 0.0
        self._peak_equity = 0.0
        self._consecutive_losses = 0

        # Task for periodic adjustments
        self._adjustment_task: asyncio.Task | None = None
        self._running = False
        self._last_adjustment_time = datetime.utcnow()

    async def start(self) -> None:
        """Start RL Risk Manager."""
        if not self._enabled:
            logger.info("RLRiskManager disabled via config")
            return

        self._running = True

        # Subscribe to market state updates
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self._bus.subscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.subscribe(EventType.EQUITY_UPDATE, self._handle_equity_update)

        # Start periodic adjustment task
        self._adjustment_task = asyncio.create_task(self._adjustment_loop())

        mode = "RL-based" if self._model else "rule-based"
        logger.info(f"RLRiskManager started ({mode} mode)")

    async def stop(self) -> None:
        """Stop RL Risk Manager."""
        self._running = False

        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self._bus.unsubscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.unsubscribe(EventType.EQUITY_UPDATE, self._handle_equity_update)

        if self._adjustment_task:
            self._adjustment_task.cancel()
            try:
                await self._adjustment_task
            except asyncio.CancelledError:
                pass

        logger.info("RLRiskManager stopped")

    async def _handle_physics_update(self, event: Event) -> None:
        """Handle physics update events."""
        data = event.data
        symbol = data.get("symbol")
        if not symbol:
            return

        # Extract physics state
        self._market_temperature[symbol] = data.get("temperature", 0.5)
        self._market_entropy[symbol] = data.get("entropy", 0.5)

        # Map phase string to index
        phase_map = {
            "accumulation": 0,
            "markup": 1,
            "distribution": 2,
            "markdown": 3,
            "unknown": 0,
        }
        phase_str = data.get("phase", "unknown")
        self._market_phase[symbol] = phase_map.get(phase_str, 0)

    async def _handle_regime_update(self, event: Event) -> None:
        """Handle regime update events for volatility."""
        data = event.data
        symbol = data.get("symbol")
        if not symbol:
            return

        # Extract volatility from regime data
        volatility = data.get("volatility", 0.02)
        self._market_volatility[symbol] = volatility

    async def _handle_order_filled(self, event: Event) -> None:
        """Track filled orders for performance calculation."""
        order = event.data.get("order")
        if not order:
            return

        # Add to trades list (will calculate PnL when position closes)
        self._trades.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "size": order.size,
            "price": order.avg_fill_price or order.price,
            "timestamp": datetime.utcnow(),
            "pnl": None,  # Filled when position closes
        })

    async def _handle_position_closed(self, event: Event) -> None:
        """Update trade PnL when position closes."""
        position = event.data.get("position")
        if not position:
            return

        pnl = position.realized_pnl

        # Update consecutive loss counter
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Find matching trade and update PnL
        for trade in reversed(self._trades):
            if trade["symbol"] == position.symbol and trade["pnl"] is None:
                trade["pnl"] = pnl
                trade["is_win"] = pnl > 0
                break

    async def _handle_equity_update(self, event: Event) -> None:
        """Track equity for drawdown calculation."""
        equity = event.data.get("equity", 0.0)
        self._equity = equity

        if self._initial_capital == 0.0:
            self._initial_capital = equity

        if equity > self._peak_equity:
            self._peak_equity = equity

    async def _adjustment_loop(self) -> None:
        """Periodically adjust risk parameters based on RL model."""
        while self._running:
            try:
                await asyncio.sleep(self._adjustment_interval)
                await self._adjust_risk_parameters()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in RL adjustment loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _adjust_risk_parameters(self) -> None:
        """Adjust risk parameters using RL model or rule-based fallback."""
        # Build observation vector
        obs = self._build_observation()

        if self._model is not None:
            # Use RL model to predict action
            action, _ = self._model.predict(obs, deterministic=True)
            leverage_adj, size_mult, stop_loss_pct = action

            # Map leverage adjustment from [-1, 1] to [1.0, 10.0]
            new_leverage = 1.0 + (leverage_adj + 1.0) * 4.5
            new_leverage = np.clip(new_leverage, 1.0, 10.0)

            self._current_leverage = float(new_leverage)
            self._current_size_mult = float(np.clip(size_mult, 0.5, 2.0))
            self._current_stop_loss_pct = float(np.clip(stop_loss_pct, 0.5, 10.0))

            logger.info(
                f"RL Risk Adjustment: leverage={self._current_leverage:.2f}x "
                f"size_mult={self._current_size_mult:.2f}x "
                f"stop_loss={self._current_stop_loss_pct:.2f}%"
            )
        else:
            # Rule-based fallback
            self._apply_rule_based_adjustments(obs)

        # Publish risk policy update event
        await self._bus.publish(Event(
            event_type=EventType.CONTEXT_UPDATE,
            data={
                "type": "rl_risk_adjustment",
                "leverage": self._current_leverage,
                "size_multiplier": self._current_size_mult,
                "stop_loss_pct": self._current_stop_loss_pct,
                "timestamp": datetime.utcnow(),
            }
        ))

        self._last_adjustment_time = datetime.utcnow()

    def _build_observation(self) -> np.ndarray:
        """Build observation vector for RL model.

        Matches the observation space from gym_env.py.
        """
        # Calculate performance metrics
        win_rate = 0.5
        profit_factor = 1.0
        if len(self._trades) > 0:
            completed_trades = [t for t in self._trades if t.get("pnl") is not None]
            if completed_trades:
                wins = sum(1 for t in completed_trades if t["pnl"] > 0)
                win_rate = wins / len(completed_trades) if completed_trades else 0.5

                gross_wins = sum(t["pnl"] for t in completed_trades if t["pnl"] > 0)
                gross_losses = abs(sum(t["pnl"] for t in completed_trades if t["pnl"] < 0))
                if gross_losses > 0:
                    profit_factor = gross_wins / gross_losses

        # Drawdown
        drawdown_pct = 0.0
        if self._peak_equity > 0:
            drawdown_pct = -((self._peak_equity - self._equity) / self._peak_equity) * 100

        # Average market state across all symbols
        avg_volatility = np.mean(list(self._market_volatility.values())) if self._market_volatility else 0.02
        avg_temperature = np.mean(list(self._market_temperature.values())) if self._market_temperature else 0.5
        avg_entropy = np.mean(list(self._market_entropy.values())) if self._market_entropy else 0.5

        # Dominant market phase (mode)
        dominant_phase = 0
        if self._market_phase:
            phases = list(self._market_phase.values())
            dominant_phase = max(set(phases), key=phases.count)

        # Phase one-hot
        phase_one_hot = [0.0, 0.0, 0.0, 0.0]
        phase_one_hot[dominant_phase] = 1.0

        # Open positions (estimate from recent trades)
        open_positions = len([t for t in self._trades[-10:] if t.get("pnl") is None])

        # Equity ratio
        equity_ratio = self._equity / self._initial_capital if self._initial_capital > 0 else 1.0

        # Hours since last adjustment
        hours_since = (datetime.utcnow() - self._last_adjustment_time).total_seconds() / 3600

        obs = np.array([
            drawdown_pct,
            win_rate,
            profit_factor,
            avg_volatility,
            *phase_one_hot,
            avg_temperature,
            avg_entropy,
            self._current_leverage,
            float(open_positions),
            equity_ratio,
            float(self._consecutive_losses),
            hours_since,
        ], dtype=np.float32)

        return obs

    def _apply_rule_based_adjustments(self, obs: np.ndarray) -> None:
        """Apply rule-based risk adjustments when RL model is not available.

        Rules:
        1. Reduce leverage in high volatility
        2. Reduce size after consecutive losses
        3. Widen stops in choppy markets (high entropy, low temperature)
        4. Increase leverage in favorable phases (markup) with low volatility
        """
        drawdown_pct = -obs[0]
        volatility = obs[3]
        temperature = obs[8]
        entropy = obs[9]
        consecutive_losses = obs[13]

        # Base values
        base_leverage = 3.0
        base_size_mult = 1.0
        base_stop_loss = 2.0

        # Rule 1: Volatility-based leverage adjustment
        if volatility > 0.04:  # High volatility
            leverage_mult = 0.5  # Halve leverage
        elif volatility > 0.03:
            leverage_mult = 0.75
        elif volatility < 0.015:  # Low volatility
            leverage_mult = 1.2
        else:
            leverage_mult = 1.0

        # Rule 2: Drawdown-based size reduction
        size_mult = base_size_mult
        if drawdown_pct > 15:
            size_mult = 0.5
        elif drawdown_pct > 10:
            size_mult = 0.75

        # Rule 3: Consecutive loss protection
        if consecutive_losses >= 3:
            size_mult *= 0.5
            leverage_mult *= 0.75

        # Rule 4: Market phase adjustments
        # Markup phase (index 1) → slightly more aggressive if low vol
        markup_phase = obs[5] == 1.0
        if markup_phase and volatility < 0.02:
            leverage_mult *= 1.1
            size_mult *= 1.1

        # Rule 5: Entropy-based stop widening
        stop_loss = base_stop_loss
        if entropy > 0.7 and temperature < 0.4:  # Choppy, cold market
            stop_loss = 3.5  # Wider stops

        # Apply adjustments
        self._current_leverage = np.clip(base_leverage * leverage_mult, 1.0, 10.0)
        self._current_size_mult = np.clip(size_mult, 0.5, 2.0)
        self._current_stop_loss_pct = np.clip(stop_loss, 0.5, 10.0)

        logger.info(
            f"Rule-based Risk Adjustment: leverage={self._current_leverage:.2f}x "
            f"size_mult={self._current_size_mult:.2f}x stop_loss={self._current_stop_loss_pct:.2f}% "
            f"(vol={volatility:.4f}, dd={drawdown_pct:.1f}%, losses={int(consecutive_losses)})"
        )

    def get_current_leverage(self) -> float:
        """Get current recommended leverage."""
        return self._current_leverage

    def get_current_size_multiplier(self) -> float:
        """Get current recommended position size multiplier."""
        return self._current_size_mult

    def get_current_stop_loss_pct(self) -> float:
        """Get current recommended stop loss percentage."""
        return self._current_stop_loss_pct

    def get_risk_parameters(self) -> dict[str, float]:
        """Get all current risk parameters."""
        return {
            "leverage": self._current_leverage,
            "size_multiplier": self._current_size_mult,
            "stop_loss_pct": self._current_stop_loss_pct,
        }
