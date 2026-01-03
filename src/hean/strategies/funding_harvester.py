"""Funding harvester strategy - low risk directional bias based on funding."""

from hean.core.bus import EventBus
from hean.core.regime import Regime
from hean.core.types import Event, FundingRate, Signal
from hean.logging import get_logger
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class FundingHarvester(BaseStrategy):
    """Harvests funding by taking small directional positions based on funding rate.

    When funding is positive (longs pay shorts), we take a small short bias.
    When funding is negative (shorts pay longs), we take a small long bias.

    Active in all regimes.
    """

    def __init__(self, bus: EventBus, symbols: list[str] | None = None) -> None:
        """Initialize the funding harvester."""
        super().__init__("funding_harvester", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._last_funding: dict[str, FundingRate] = {}
        self._positions: dict[str, str] = {}  # symbol -> side
        # Active in all regimes
        self._allowed_regimes = {Regime.RANGE, Regime.NORMAL, Regime.IMPULSE}

    async def on_tick(self, event: Event) -> None:
        """Handle tick events - not used for this strategy."""
        pass

    async def on_funding(self, event: Event) -> None:
        """Handle funding rate events."""
        funding: FundingRate = event.data["funding"]

        if funding.symbol not in self._symbols:
            return

        self._last_funding[funding.symbol] = funding

        # Generate signal based on funding rate
        await self._evaluate_funding(funding)

    async def _evaluate_funding(self, funding: FundingRate) -> None:
        """Evaluate funding rate and generate signal if appropriate."""
        # Threshold: only act if funding rate is significant
        funding_threshold = 0.0001  # 0.01%

        if abs(funding.rate) < funding_threshold:
            return

        # If funding is positive (longs pay shorts), we want to be short
        # If funding is negative (shorts pay longs), we want to be long
        if funding.rate > 0:
            side = "sell"  # Short position
        else:
            side = "buy"  # Long position

        # Check if we already have a position
        if funding.symbol in self._positions:
            current_side = self._positions[funding.symbol]
            if current_side == side:
                return  # Already positioned correctly
            # Otherwise, we'd close and reverse, but for simplicity, just skip

        # Generate signal with small size (low risk)
        # Use synthetic price for entry (in real system, get from tick)
        entry_price = 50000.0 if "BTC" in funding.symbol else 3000.0  # Placeholder

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=funding.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=entry_price * (0.98 if side == "buy" else 1.02),  # 2% stop
            take_profit=entry_price * (1.01 if side == "buy" else 0.99),  # 1% target
            metadata={"funding_rate": funding.rate, "reason": "funding_harvest"},
        )

        await self._publish_signal(signal)
        self._positions[funding.symbol] = side
