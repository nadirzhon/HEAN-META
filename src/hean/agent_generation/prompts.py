"""Prompts for agent generation with full HEAN architecture context."""

# Backward-compatible constants (referenced in __init__.py)
PROMPT_INITIAL = "initial"
PROMPT_EVOLUTION = "evolution"
PROMPT_MUTATION = "mutation"
PROMPT_MARKET_CONDITIONS = "market_conditions"
PROMPT_HYBRID = "hybrid"
PROMPT_PROBLEM_FOCUSED = "problem_focused"
PROMPT_EVALUATION = "evaluation"
PROMPT_CREATIVE = "creative"
PROMPT_ANALYTICAL = "analytical"

SYSTEM_PROMPT = """You are an expert Python developer specializing in algorithmic trading systems.
You generate trading strategy code for the HEAN trading system.

## HEAN Architecture

All strategies MUST inherit from BaseStrategy and follow this interface:

```python
from hean.strategies.base import BaseStrategy
from hean.core.bus import EventBus
from hean.core.types import Event, Signal, Tick
from hean.logging import get_logger

class YourStrategy(BaseStrategy):
    def __init__(self, bus: EventBus, symbols: list[str] | None = None) -> None:
        super().__init__("your_strategy_id", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        # ... your state variables

    async def on_tick(self, event: Event) -> None:
        tick: Tick = event.data["tick"]
        if tick.symbol not in self._symbols:
            return
        # ... your logic, emit signals via await self._publish_signal(signal)

    async def on_funding(self, event: Event) -> None:
        pass  # Handle funding events if needed
```

## Signal class (all fields):
```python
Signal(
    strategy_id: str,        # Your strategy ID
    symbol: str,             # e.g. "BTCUSDT"
    side: str,               # "buy" or "sell"
    entry_price: float,      # Current price
    stop_loss: float | None, # REQUIRED for risk management
    take_profit: float | None,
    size: float | None,      # If None, risk layer will size
    confidence: float,       # 0.0-1.0, signal quality
    urgency: float,          # 0.0-1.0, time sensitivity
    metadata: dict,          # Extra info (reason, indicators, etc.)
)
```

## Tick class:
```python
Tick(symbol: str, price: float, volume: float, timestamp: datetime)
```

## Allowed imports:
- collections (deque), datetime (datetime, timedelta), math, typing
- hean.core.bus.EventBus, hean.core.types.{Event, Signal, Tick}
- hean.strategies.base.BaseStrategy, hean.logging.get_logger

## CRITICAL RULES:
1. Class MUST inherit from BaseStrategy
2. Constructor signature: `__init__(self, bus: EventBus, symbols: list[str] | None = None)`
3. Call `super().__init__("strategy_id", bus)` in __init__
4. Implement `async on_tick(self, event: Event)` and `async on_funding(self, event: Event)`
5. Emit signals via `await self._publish_signal(signal)`
6. Always set stop_loss on signals
7. Return ONLY the complete Python file, no explanations

## Reference example (working MomentumTrader):
```python
from collections import deque
from hean.core.bus import EventBus
from hean.core.types import Event, Signal, Tick
from hean.logging import get_logger
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)

class MomentumTrader(BaseStrategy):
    def __init__(self, bus: EventBus, symbols: list[str] | None = None) -> None:
        super().__init__("momentum_trader", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._price_history: dict[str, deque[float]] = {}
        self._window_size = 10
        self._momentum_threshold = 0.001

    async def on_tick(self, event: Event) -> None:
        tick: Tick = event.data["tick"]
        if tick.symbol not in self._symbols:
            return
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._window_size)
        self._price_history[tick.symbol].append(tick.price)
        if len(self._price_history[tick.symbol]) < self._window_size:
            return
        prices = list(self._price_history[tick.symbol])
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        momentum = sum(returns) / len(returns) if returns else 0.0
        if momentum > self._momentum_threshold:
            side = "buy"
        elif momentum < -self._momentum_threshold:
            side = "sell"
        else:
            return
        signal = Signal(
            strategy_id=self.strategy_id, symbol=tick.symbol, side=side,
            entry_price=tick.price,
            stop_loss=tick.price * (0.98 if side == "buy" else 1.02),
            take_profit=tick.price * (1.02 if side == "buy" else 0.98),
            metadata={"momentum": momentum},
        )
        await self._publish_signal(signal)

    async def on_funding(self, event: Event) -> None:
        pass
```
"""


def get_prompt(prompt_type: str, **kwargs) -> str:
    """Get prompt template by type."""
    dispatch = {
        "initial": _get_initial_prompt,
        "evolution": _get_evolution_prompt,
        "mutation": _get_mutation_prompt,
        "market_conditions": _get_market_conditions_prompt,
        "hybrid": _get_hybrid_prompt,
        "problem_focused": _get_problem_focused_prompt,
        "evaluation": _get_evaluation_prompt,
        "creative": _get_creative_prompt,
        "analytical": _get_analytical_prompt,
    }
    fn = dispatch.get(prompt_type)
    if fn is None:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return fn(**kwargs)


def _get_initial_prompt(**kwargs) -> str:
    return """Generate a new trading strategy that:
- Detects a specific market pattern (choose one: mean reversion, breakout, volatility expansion, or order flow imbalance)
- Has configurable thresholds
- Uses proper risk management with stop_loss and take_profit on every signal
- Tracks price history using deque for efficient memory usage
- Has a cooldown between signals to avoid overtrading
Return ONLY the complete Python file."""


def _get_evolution_prompt(**kwargs) -> str:
    best = kwargs.get("best_agents_info", "N/A")
    worst = kwargs.get("worst_agents_info", "N/A")
    market = kwargs.get("market_conditions", "N/A")
    perf = kwargs.get("performance_metrics", "N/A")
    return f"""Evolve a new trading strategy based on these performance observations:

## Best performing strategies:
{best}

## Worst performing strategies (avoid these patterns):
{worst}

## Current market conditions:
{market}

## Performance metrics:
{perf}

Generate an improved strategy that:
1. Combines the best features from top performers
2. Avoids patterns from worst performers
3. Adapts to current market conditions
4. Has proper stop_loss and take_profit on every signal

Return ONLY the complete Python file."""


def _get_mutation_prompt(**kwargs) -> str:
    code = kwargs.get("agent_code", "")
    pf = kwargs.get("profit_factor", 1.0)
    pnl = kwargs.get("total_pnl", 0.0)
    dd = kwargs.get("max_drawdown_pct", 0.0)
    wr = kwargs.get("win_rate", 0.0)
    issues = kwargs.get("issues", "")
    return f"""Improve this underperforming trading strategy.

## Current performance metrics:
- Profit Factor: {pf:.2f}
- Win Rate: {wr:.1f}%
- Max Drawdown: {dd:.1f}%
- Total PnL: ${pnl:.2f}

## Identified problems:
{issues}

## Current strategy code:
```python
{code}
```

## Your task:
1. Analyze WHY this strategy underperforms based on the metrics and code
2. Fix the entry/exit logic to improve profit factor and win rate
3. Adjust thresholds, add filters, or improve signal quality
4. Ensure every signal has stop_loss and take_profit
5. Keep the same class structure (inherit BaseStrategy, same constructor signature)
6. Give the strategy a NEW unique strategy_id (e.g. "improved_momentum_v2")

Return ONLY the complete improved Python file."""


def _get_market_conditions_prompt(**kwargs) -> str:
    vol = kwargs.get("volatility_level", "medium")
    vol_val = kwargs.get("volatility_value", 0.0)
    trend = kwargs.get("trend_direction", "neutral")
    strength = kwargs.get("trend_strength", "weak")
    volume = kwargs.get("volume_level", "normal")
    regime = kwargs.get("market_regime", "unknown")
    return f"""Generate a trading strategy optimized for these market conditions:

- Volatility: {vol} ({vol_val:.4f})
- Trend: {trend} (strength: {strength})
- Volume: {volume}
- Market regime: {regime}

The strategy should:
1. Be specifically tuned for these conditions
2. Use appropriate thresholds for the volatility level
3. Follow the trend direction if trending, or mean-revert if ranging
4. Have proper stop_loss and take_profit scaled to volatility

Return ONLY the complete Python file."""


def _get_hybrid_prompt(**kwargs) -> str:
    a1 = kwargs.get("agent1_code", "")
    pf1 = kwargs.get("pf1", 0.0)
    a2 = kwargs.get("agent2_code", "")
    wr2 = kwargs.get("wr2", 0.0)
    a3 = kwargs.get("agent3_code", "")
    sharpe3 = kwargs.get("sharpe3", 0.0)
    return f"""Create a hybrid strategy combining these three approaches:

## Strategy 1 (Best Profit Factor: {pf1:.2f}):
```python
{a1}
```

## Strategy 2 (Best Win Rate: {wr2:.1f}%):
```python
{a2}
```

## Strategy 3 (Best Sharpe Ratio: {sharpe3:.2f}):
```python
{a3}
```

Combine the best features into a single strategy that:
1. Uses the entry logic from the highest PF strategy
2. Uses the exit/risk logic from the highest win rate strategy
3. Applies the position sizing from the best Sharpe strategy
4. Has a voting/consensus mechanism if signals conflict

Return ONLY the complete Python file."""


def _get_problem_focused_prompt(**kwargs) -> str:
    problem = kwargs.get("problem_description", "")
    pf = kwargs.get("current_pf", 1.0)
    areas = kwargs.get("problem_areas", "")
    failed = kwargs.get("failed_patterns", "")
    f1 = kwargs.get("focus_area_1", "")
    f2 = kwargs.get("focus_area_2", "")
    f3 = kwargs.get("focus_area_3", "")
    return f"""Generate a strategy focused on solving this specific problem:

## Problem: {problem}
## Current profit factor: {pf:.2f}
## Problem areas: {areas}
## Patterns that failed (avoid these): {failed}

## Focus areas:
1. {f1}
2. {f2}
3. {f3}

Design a strategy that directly addresses these issues.
Return ONLY the complete Python file."""


def _get_evaluation_prompt(**kwargs) -> str:
    code = kwargs.get("agent_code", "")
    pf = kwargs.get("pf", 1.0)
    pnl = kwargs.get("pnl", 0.0)
    dd = kwargs.get("dd", 0.0)
    wr = kwargs.get("wr", 0.0)
    sharpe = kwargs.get("sharpe", 0.0)
    trades = kwargs.get("trades", 0)
    return f"""Evaluate and improve this strategy:

## Metrics:
- Profit Factor: {pf:.2f}, PnL: ${pnl:.2f}, Drawdown: {dd:.1f}%
- Win Rate: {wr:.1f}%, Sharpe: {sharpe:.2f}, Trades: {trades}

## Code:
```python
{code}
```

Analyze the strategy and generate an improved version.
Return ONLY the complete Python file."""


def _get_creative_prompt(**kwargs) -> str:
    return """Generate a creative, innovative trading strategy that uses an unconventional approach.

Ideas to explore (pick one or combine):
- Statistical anomaly detection (z-score based entries)
- Volatility regime switching (different logic for calm vs volatile markets)
- Price action pattern recognition (double bottoms, head and shoulders approximation)
- Mean reversion with dynamic bands
- Momentum divergence detection

The strategy must be practical and generate real signals with proper risk management.
Return ONLY the complete Python file."""


def _get_analytical_prompt(**kwargs) -> str:
    return """Generate an analytical trading strategy that uses statistical methods:

- Track rolling statistics (mean, std, skewness, kurtosis)
- Use z-scores or percentile rankings for signal generation
- Implement proper statistical significance checks before signaling
- Use adaptive thresholds that adjust to market conditions

Return ONLY the complete Python file."""
