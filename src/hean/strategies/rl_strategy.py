"""
RL-based trading strategy.

Integrates trained PPO agent into HEAN trading system.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional

import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, Signal, Side
from hean.logging import get_logger
from hean.rl.agent import RLTradingAgent
from hean.rl.trading_environment import Action, TradingConfig
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class RLStrategy(BaseStrategy):
    """
    Strategy that uses trained RL agent for trading decisions.

    The RL agent is trained offline and then deployed to make real-time
    trading decisions based on market data.
    """

    def __init__(
        self,
        strategy_id: str,
        bus: EventBus,
        agent: RLTradingAgent,
        symbol: str = "BTCUSDT",
        lookback_window: int = 100,
        min_confidence: float = 0.0,
        trading_config: Optional[TradingConfig] = None,
    ):
        """
        Initialize RL strategy.

        Args:
            strategy_id: Unique strategy identifier
            bus: Event bus for communication
            agent: Trained RL agent
            symbol: Trading symbol
            lookback_window: Number of candles to maintain for features
            min_confidence: Minimum confidence threshold for signals (0-1)
            trading_config: Trading configuration (for sizing)
        """
        super().__init__(strategy_id, bus)

        self.agent = agent
        self.symbol = symbol
        self.lookback_window = lookback_window
        self.min_confidence = min_confidence
        self.config = trading_config or TradingConfig()

        # Market data buffer (for computing features)
        self.price_buffer: deque = deque(maxlen=lookback_window)
        self.volume_buffer: deque = deque(maxlen=lookback_window)

        # Current position tracking
        self.current_position: float = 0.0  # BTC held
        self.avg_entry_price: float = 0.0
        self.cash: float = self.config.initial_capital

        # Performance tracking
        self.total_signals: int = 0
        self.total_trades: int = 0
        self.last_action: Optional[Action] = None

        # Observation computation
        self._last_observation: Optional[np.ndarray] = None

        logger.info(f"RLStrategy initialized for {symbol}")

    async def start(self) -> None:
        """Start the strategy."""
        await super().start()
        logger.info(f"RL Strategy {self.strategy_id} started for {self.symbol}")

    async def stop(self) -> None:
        """Stop the strategy."""
        await super().stop()
        logger.info(
            f"RL Strategy {self.strategy_id} stopped. "
            f"Total signals: {self.total_signals}, Total trades: {self.total_trades}"
        )

    async def on_tick(self, event: Event) -> None:
        """
        Handle tick events and generate signals using RL agent.

        Args:
            event: Tick event containing price data
        """
        tick = event.data.get("tick")
        if not tick or tick.symbol != self.symbol:
            return

        # Update price buffer
        self.price_buffer.append(tick.price)
        self.volume_buffer.append(tick.volume if hasattr(tick, 'volume') else 0.0)

        # Need enough data for features
        if len(self.price_buffer) < 50:  # Min for technical indicators
            return

        # Compute observation
        try:
            observation = self._compute_observation(tick)
        except Exception as e:
            logger.error(f"Error computing observation: {e}")
            return

        # Get action from RL agent
        try:
            action = self.agent.predict(observation, explore=False)
            action_enum = Action(action)
        except Exception as e:
            logger.error(f"Error getting action from agent: {e}")
            return

        self.last_action = action_enum

        # Convert action to signal
        signal = self._action_to_signal(action_enum, tick.price)

        if signal is not None:
            self.total_signals += 1
            await self._publish_signal(signal)

    async def on_funding(self, event: Event) -> None:
        """Handle funding events (not used by RL strategy)."""
        pass

    def _compute_observation(self, tick) -> np.ndarray:
        """
        Compute observation vector for RL agent.

        Mirrors the observation computation in BitcoinTradingEnv.

        Args:
            tick: Current tick data

        Returns:
            Observation array (25 features)
        """
        prices = np.array(list(self.price_buffer), dtype=np.float32)
        volumes = np.array(list(self.volume_buffer), dtype=np.float32)

        current_price = tick.price

        # 1. Price features (5)
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0.0])
        current_return = returns[-1] if len(returns) > 0 else 0.0
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.0

        # Simple high/low from recent prices
        recent_high = np.max(prices[-20:]) if len(prices) >= 20 else current_price
        recent_low = np.min(prices[-20:]) if len(prices) >= 20 else current_price

        price_features = np.array([
            current_price / 10000.0,
            current_return,
            volatility,
            (recent_high - current_price) / current_price,
            (recent_low - current_price) / current_price,
        ], dtype=np.float32)

        # 2. Volume features (3)
        current_volume = volumes[-1] if len(volumes) > 0 else 0.0
        volume_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else current_volume
        volume_change = (current_volume - volumes[-2]) / volumes[-2] if len(volumes) >= 2 else 0.0

        volume_features = np.array([
            current_volume / 1000.0,
            volume_ma / 1000.0,
            volume_change,
        ], dtype=np.float32)

        # 3. Technical indicators (8) - simplified
        # RSI
        rsi = self._compute_rsi(prices) if len(prices) >= 14 else 50.0

        # MACD (simplified)
        ema_12 = self._ema(prices, 12) if len(prices) >= 12 else current_price
        ema_26 = self._ema(prices, 26) if len(prices) >= 26 else current_price
        macd = ema_12 - ema_26

        # Bollinger Bands
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        std_20 = np.std(prices[-20:]) if len(prices) >= 20 else 0.0
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20

        # ATR (simplified)
        atr = volatility * current_price

        # SMA 50
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price

        technical_features = np.array([
            rsi / 100.0,
            macd / current_price,
            0.0,  # MACD signal (simplified)
            (bb_upper - current_price) / current_price,
            (sma_20 - current_price) / current_price,
            (bb_lower - current_price) / current_price,
            atr / current_price,
            (current_price - sma_50) / sma_50 if sma_50 > 0 else 0.0,
        ], dtype=np.float32)

        # 4. Position features (4)
        unrealized_pnl_pct = 0.0
        if self.current_position > 0 and self.avg_entry_price > 0:
            unrealized_pnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price

        position_features = np.array([
            self.current_position / self.config.max_position_size,
            self.avg_entry_price / 10000.0,
            unrealized_pnl_pct,
            0.0,  # Position duration (not tracked in real-time)
        ], dtype=np.float32)

        # 5. Portfolio features (5)
        position_value = self.current_position * current_price
        equity = self.cash + position_value

        portfolio_features = np.array([
            self.cash / equity if equity > 0 else 0.0,
            equity / self.config.initial_capital,
            0.0,  # Drawdown (not tracked in real-time)
            (equity - self.config.initial_capital) / self.config.initial_capital,
            0.0,  # Win rate (not tracked in real-time)
        ], dtype=np.float32)

        # Concatenate all features
        observation = np.concatenate([
            price_features,
            volume_features,
            technical_features,
            position_features,
            portfolio_features,
        ])

        # Handle NaN/Inf
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        self._last_observation = observation
        return observation

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Compute RSI for price series."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    @staticmethod
    def _ema(prices: np.ndarray, span: int) -> float:
        """Compute EMA for price series."""
        if len(prices) < span:
            return prices[-1]

        alpha = 2.0 / (span + 1)
        ema = prices[-span]

        for price in prices[-span+1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def _action_to_signal(self, action: Action, current_price: float) -> Optional[Signal]:
        """
        Convert RL action to trading signal.

        Args:
            action: RL agent action
            current_price: Current market price

        Returns:
            Signal or None if action is HOLD
        """
        if action == Action.HOLD:
            return None

        # Determine side and size
        if action in [Action.BUY_SMALL, Action.BUY_MEDIUM, Action.BUY_LARGE]:
            side = Side.BUY

            if action == Action.BUY_SMALL:
                size_pct = self.config.small_size
            elif action == Action.BUY_MEDIUM:
                size_pct = self.config.medium_size
            else:
                size_pct = self.config.large_size

            # Calculate size in BTC
            available_cash = self.cash * size_pct
            size = available_cash / current_price

        else:  # SELL actions
            side = Side.SELL

            if action == Action.SELL_SMALL:
                size_pct = self.config.small_size
            elif action == Action.SELL_MEDIUM:
                size_pct = self.config.medium_size
            else:
                size_pct = self.config.large_size

            # Calculate size in BTC
            size = self.current_position * size_pct

        # Create signal
        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            side=side,
            entry_price=current_price,
            size=size,
            stop_loss=None,  # RL agent manages risk internally
            take_profit=None,
            confidence=1.0,  # RL actions are deterministic in deployment
        )

        logger.info(
            f"RL Action: {action.name} -> Signal: {side.name} {size:.6f} BTC @ ${current_price:.2f}"
        )

        return signal

    def update_position(
        self,
        side: Side,
        size: float,
        price: float,
    ) -> None:
        """
        Update internal position tracking (called by execution layer).

        Args:
            side: Trade side
            size: Trade size
            price: Trade price
        """
        if side == Side.BUY:
            # Update average entry price
            total_cost = self.current_position * self.avg_entry_price + size * price
            self.current_position += size
            self.avg_entry_price = total_cost / self.current_position if self.current_position > 0 else price

            # Update cash
            self.cash -= size * price

        else:  # SELL
            self.current_position -= size
            self.cash += size * price

            # Reset avg entry if position closed
            if self.current_position <= 0:
                self.current_position = 0.0
                self.avg_entry_price = 0.0

        self.total_trades += 1

        logger.debug(
            f"Position updated: {self.current_position:.6f} BTC @ avg ${self.avg_entry_price:.2f}, "
            f"Cash: ${self.cash:.2f}"
        )

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        return {
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'position': self.current_position,
            'avg_entry_price': self.avg_entry_price,
            'cash': self.cash,
            'total_signals': self.total_signals,
            'total_trades': self.total_trades,
            'last_action': self.last_action.name if self.last_action else None,
        }
