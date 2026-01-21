"""Engine facade for unified orchestration of TradingSystem and ProcessFactory."""

import asyncio
from typing import Any, Literal

from hean.config import settings
from hean.logging import get_logger
from hean.main import TradingSystem

logger = get_logger(__name__)


class EngineFacade:
    """Unified facade for trading engine orchestration."""

    def __init__(self) -> None:
        """Initialize the engine facade."""
        self._trading_system: TradingSystem | None = None
        self._running = False
        self._lock = asyncio.Lock()
        
        # Expose advanced systems (will be set when trading system starts)
        self._meta_learning_engine = None
        self._causal_inference_engine = None
        self._multimodal_swarm = None

    async def start(self) -> dict[str, Any]:
        """Start the trading engine.

        Returns:
            Status dictionary with engine state
        """
        async with self._lock:
            if self._running:
                return {"status": "already_running", "message": "Engine is already running"}

            try:
                # Create and start trading system
                self._trading_system = TradingSystem(mode="run")
                await self._trading_system.start()
                
                # Expose advanced systems for API access
                if hasattr(self._trading_system, '_meta_learning_engine'):
                    self._meta_learning_engine = self._trading_system._meta_learning_engine
                if hasattr(self._trading_system, '_causal_inference_engine'):
                    self._causal_inference_engine = self._trading_system._causal_inference_engine
                if hasattr(self._trading_system, '_multimodal_swarm'):
                    self._multimodal_swarm = self._trading_system._multimodal_swarm
                
                self._running = True

                logger.info("Engine started successfully")
                return {
                    "status": "started",
                    "message": "Engine started successfully",
                    "trading_mode": settings.trading_mode,
                    "is_live": settings.is_live,
                    "dry_run": settings.dry_run,
                }
            except Exception as e:
                logger.error(f"Failed to start engine: {e}", exc_info=True)
                self._running = False
                raise

    async def stop(self) -> dict[str, Any]:
        """Stop the trading engine.

        Returns:
            Status dictionary
        """
        async with self._lock:
            if not self._running or self._trading_system is None:
                return {"status": "not_running", "message": "Engine is not running"}

            try:
                await self._trading_system.stop()
                self._trading_system = None
                self._running = False

                logger.info("Engine stopped successfully")
                return {"status": "stopped", "message": "Engine stopped successfully"}
            except Exception as e:
                logger.error(f"Failed to stop engine: {e}", exc_info=True)
                raise

    async def get_status(self) -> dict[str, Any]:
        """Get current engine status.

        Returns:
            Status dictionary
        """
        async with self._lock:
            if not self._running or self._trading_system is None:
                return {
                    "status": "stopped",
                    "running": False,
                    "trading_mode": settings.trading_mode,
                    "is_live": settings.is_live,
                    "dry_run": settings.dry_run,
                }

            # Get portfolio state
            accounting = self._trading_system._accounting
            equity = accounting.get_equity()
            daily_pnl = accounting.get_daily_pnl()

            return {
                "status": "running",
                "running": True,
                "trading_mode": settings.trading_mode,
                "is_live": settings.is_live,
                "dry_run": settings.dry_run,
                "equity": equity,
                "daily_pnl": daily_pnl,
                "initial_capital": accounting._initial_capital,
            }

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions.

        Returns:
            List of position dictionaries
        """
        if not self._running or self._trading_system is None:
            return []

        accounting = self._trading_system._accounting
        positions_list = accounting.get_positions()

        return [
            {
                "symbol": pos.symbol,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "side": pos.side,
                "position_id": pos.position_id,
            }
            for pos in positions_list
        ]

    async def get_orders(self, status: Literal["all", "open", "filled"] = "all") -> list[dict[str, Any]]:
        """Get orders.

        Args:
            status: Filter by status (all, open, filled)

        Returns:
            List of order dictionaries
        """
        if not self._running or self._trading_system is None:
            return []

        order_manager = self._trading_system._order_manager

        if status == "open":
            orders = order_manager.get_open_orders()
        elif status == "filled":
            orders = order_manager.get_filled_orders()
        else:
            orders = list(order_manager._orders.values())

        return [
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "size": order.size,
                "filled_size": order.filled_size,
                "price": order.price,
                "status": order.status.value,
                "strategy_id": order.strategy_id,
                "timestamp": order.timestamp.isoformat(),
            }
            for order in orders
        ]

    async def pause(self) -> dict[str, Any]:
        """Pause the trading engine (stop accepting new signals).

        Returns:
            Status dictionary
        """
        async with self._lock:
            if not self._running or self._trading_system is None:
                return {"status": "not_running", "message": "Engine is not running"}

            # Set stop trading flag
            self._trading_system._stop_trading = True
            logger.info("Engine paused")
            return {"status": "paused", "message": "Engine paused"}

    async def resume(self) -> dict[str, Any]:
        """Resume the trading engine.

        Returns:
            Status dictionary
        """
        async with self._lock:
            if not self._running or self._trading_system is None:
                return {"status": "not_running", "message": "Engine is not running"}

            # Clear stop trading flag
            self._trading_system._stop_trading = False
            logger.info("Engine resumed")
            return {"status": "resumed", "message": "Engine resumed"}

    async def get_risk_status(self) -> dict[str, Any]:
        """Get risk management status.

        Returns:
            Risk status dictionary
        """
        if not self._running or self._trading_system is None:
            return {
                "killswitch_triggered": False,
                "stop_trading": False,
                "risk_limits": {},
            }

        killswitch = self._trading_system._killswitch
        risk_limits = self._trading_system._risk_limits
        accounting = self._trading_system._accounting

        return {
            "killswitch_triggered": killswitch._triggered,
            "stop_trading": self._trading_system._stop_trading,
            "equity": accounting.get_equity(),
            "daily_pnl": accounting.get_daily_pnl(),
            "drawdown": accounting.get_drawdown(),
            "drawdown_pct": accounting.get_drawdown_pct(),
            "max_open_positions": risk_limits._max_open_positions,
            "current_positions": len(accounting.get_positions()),
        }

    async def get_strategies(self) -> list[dict[str, Any]]:
        """Get list of strategies.

        Returns:
            List of strategy dictionaries
        """
        if not self._running or self._trading_system is None:
            return []

        strategies = []
        for strategy in self._trading_system._strategies:
            strategies.append({
                "strategy_id": strategy.strategy_id,
                "enabled": strategy._running,
                "type": type(strategy).__name__,
            })

        return strategies

    async def enable_strategy(self, strategy_id: str, enabled: bool) -> dict[str, Any]:
        """Enable or disable a strategy.

        Args:
            strategy_id: Strategy ID
            enabled: Enable or disable

        Returns:
            Status dictionary
        """
        if not self._running or self._trading_system is None:
            return {"status": "error", "message": "Engine is not running"}

        for strategy in self._trading_system._strategies:
            if strategy.strategy_id == strategy_id:
                if enabled:
                    await strategy.start()
                else:
                    await strategy.stop()
                return {"status": "success", "message": f"Strategy {strategy_id} {'enabled' if enabled else 'disabled'}"}

        return {"status": "error", "message": f"Strategy {strategy_id} not found"}

    async def get_bus(self):
        """Get event bus instance."""
        if not self._running or self._trading_system is None:
            return None
        return self._trading_system._bus

    async def get_orderbook_presence(self, symbol: str | None = None) -> dict | list[dict]:
        """Get orderbook presence (Phase 3: Smart Limit Engine).
        
        Args:
            symbol: Optional symbol filter
        
        Returns:
            Orderbook presence data
        """
        if not self._running or self._trading_system is None:
            return {} if symbol else []
        
        execution_router = self._trading_system._execution_router
        return execution_router.get_orderbook_presence(symbol)

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

