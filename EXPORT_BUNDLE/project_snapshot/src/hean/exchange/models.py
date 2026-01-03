"""Exchange client protocol and base models."""

from abc import ABC, abstractmethod
from typing import Protocol

from hean.core.types import Order, OrderRequest, Tick


class ExchangeClient(Protocol):
    """Protocol for exchange clients."""

    async def connect(self) -> None:
        """Connect to the exchange."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        ...

    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place an order on the exchange."""
        ...

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        ...

    async def get_ticker(self, symbol: str) -> Tick:
        """Get current ticker for a symbol."""
        ...


class PriceFeed(ABC):
    """Abstract price feed interface."""

    @abstractmethod
    async def start(self) -> None:
        """Start the price feed."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the price feed."""
        ...

    @abstractmethod
    async def subscribe(self, symbol: str) -> None:
        """Subscribe to price updates for a symbol."""
        ...
