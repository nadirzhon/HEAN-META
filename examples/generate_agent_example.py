#!/usr/bin/env python3
"""Example of using the agent generation system."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hean.agent_generation import AgentGenerator
from hean.logging import get_logger

logger = get_logger(__name__)


def example_initial_generation() -> None:
    """Example: Generate initial agents."""
    print("Example 1: Generating initial agents")

    # Initialize generator
    generator = AgentGenerator()

    # Generate a single agent
    code = generator.generate_agent(
        prompt_type="initial",
        output_path="examples/generated_agent_1.py"
    )
    print(f"Generated agent with {len(code)} characters")
    print("Code saved to examples/generated_agent_1.py")


def example_batch_generation() -> None:
    """Example: Generate multiple agents."""
    print("\nExample 2: Generating multiple agents")

    generator = AgentGenerator()

    # Generate 5 agents
    codes = generator.generate_initial_agents(
        count=5,
        output_dir="examples/generated_agents"
    )
    print(f"Generated {len(codes)} agents in examples/generated_agents/")


def example_evolution() -> None:
    """Example: Evolve agent based on best/worst performers."""
    print("\nExample 3: Evolving agent")

    generator = AgentGenerator()

    # Example data
    best_agents = """
    Agent1: Profit Factor=2.5, Win Rate=60%, Sharpe=1.8
    - Uses RSI + MACD combination
    - Tight stop losses (0.5%)
    - Active in IMPULSE regime
    """

    worst_agents = """
    Agent2: Profit Factor=0.8, Win Rate=40%, Sharpe=0.3
    - Uses only moving averages
    - Wide stop losses (2%)
    - Trades in all regimes
    """

    market_conditions = """
    Current market: High volatility (0.05), Bullish trend, High volume
    Recent performance: Strong uptrend over last 30 days
    """

    performance_metrics = """
    Average Profit Factor: 1.5
    Average Win Rate: 50%
    Average Sharpe Ratio: 1.0
    """

    code = generator.evolve_agent(
        best_agents_info=best_agents,
        worst_agents_info=worst_agents,
        market_conditions=market_conditions,
        performance_metrics=performance_metrics,
        output_path="examples/evolved_agent.py"
    )
    print("Evolved agent saved to examples/evolved_agent.py")


def example_mutation() -> None:
    """Example: Mutate existing agent."""
    print("\nExample 4: Mutating existing agent")

    generator = AgentGenerator()

    # Example agent code (simplified)
    agent_code = """
from hean.core.bus import EventBus
from hean.core.types import Event, Signal, Tick
from hean.strategies.base import BaseStrategy

class ExampleAgent(BaseStrategy):
    def __init__(self, bus: EventBus, symbols=None):
        super().__init__("example_agent", bus)
        self._symbols = symbols or ["BTCUSDT"]
    
    async def on_tick(self, event: Event):
        tick = event.data["tick"]
        # Simple logic
        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=tick.symbol,
            side="buy",
            entry_price=tick.price,
            stop_loss=tick.price * 0.99,
            take_profit=tick.price * 1.01,
        )
        await self._publish_signal(signal)
    
    async def on_funding(self, event: Event):
        pass
    
    async def on_regime_update(self, event: Event):
        pass
"""

    code = generator.mutate_agent(
        agent_code=agent_code,
        profit_factor=1.2,
        total_pnl=1000.0,
        max_drawdown_pct=15.0,
        win_rate=55.0,
        issues="Low win rate, high drawdown, needs better entry filters",
        output_path="examples/mutated_agent.py"
    )
    print("Mutated agent saved to examples/mutated_agent.py")


def example_creative() -> None:
    """Example: Generate creative agent."""
    print("\nExample 5: Generating creative agent")

    generator = AgentGenerator()

    code = generator.generate_creative_agent(
        output_path="examples/creative_agent.py"
    )
    print("Creative agent saved to examples/creative_agent.py")


def main() -> None:
    """Run all examples."""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: No LLM API key found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    try:
        example_initial_generation()
        # Uncomment to run other examples (they use API calls)
        # example_batch_generation()
        # example_evolution()
        # example_mutation()
        # example_creative()

        print("\n✅ Examples completed successfully!")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

