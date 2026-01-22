"""Engine facade for unified orchestration of TradingSystem and ProcessFactory."""

import asyncio
from typing import Any, Literal

from hean.config import settings
from hean.logging import get_logger
from hean.core.bus import EventBus
from hean.main import TradingSystem
from hean.api.telemetry import telemetry_service
from hean.core.types import Event, EventType, OrderStatus

logger = get_logger(__name__)


class EngineFacade:
    """Unified facade for trading engine orchestration."""

    def __init__(self, bus: EventBus | None = None) -> None:
        """Initialize the engine facade."""
        self._trading_system: TradingSystem | None = None
        self._running = False
        self._lock = asyncio.Lock()
        self._bus = bus
        self._state = "STOPPED"
        telemetry_service.set_engine_state(self._state)
        
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
                self._trading_system = TradingSystem(mode="run", bus=self._bus)
                await self._trading_system.start()
                
                # Expose advanced systems for API access
                if hasattr(self._trading_system, '_meta_learning_engine'):
                    self._meta_learning_engine = self._trading_system._meta_learning_engine
                if hasattr(self._trading_system, '_causal_inference_engine'):
                    self._causal_inference_engine = self._trading_system._causal_inference_engine
                if hasattr(self._trading_system, '_multimodal_swarm'):
                    self._multimodal_swarm = self._trading_system._multimodal_swarm
                
                self._running = True
                self._state = "RUNNING"
                telemetry_service.set_engine_state("RUNNING")

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
                self._state = "ERROR"
                telemetry_service.set_engine_state("ERROR")
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
                self._state = "STOPPED"
                telemetry_service.set_engine_state("STOPPED")

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
                    "engine_state": self._state,
                    "trading_mode": settings.trading_mode,
                    "is_live": settings.is_live,
                    "dry_run": settings.dry_run,
                }

            # Get portfolio state
            accounting = self._trading_system._accounting
            equity = accounting.get_equity()
            daily_pnl = accounting.get_daily_pnl(equity)

            return {
                "status": "running",
                "running": True,
                "engine_state": self._state,
                "trading_mode": settings.trading_mode,
                "is_live": settings.is_live,
                "dry_run": settings.dry_run,
                "equity": equity,
                "daily_pnl": daily_pnl,
                "initial_capital": accounting._initial_capital,
            }

    async def get_trading_state(self) -> dict[str, Any]:
        """Get unified trading state (account, positions, orders)."""
        if not self._running or self._trading_system is None:
            return {"account_state": None, "positions": [], "orders": []}

        return self._trading_system._build_trading_state()

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
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "side": pos.side,
                "position_id": pos.position_id,
                "take_profit": pos.take_profit,
                "stop_loss": pos.stop_loss,
                "strategy_id": pos.strategy_id,
                "status": "open",
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

    async def close_position(self, position_id: str, reason: str = "api_close") -> dict[str, Any]:
        """Force-close a specific position (paper-safe)."""
        if not self._running or self._trading_system is None:
            return {"status": "error", "message": "Engine is not running"}

        accounting = self._trading_system._accounting
        positions = accounting.get_positions()
        target = next((p for p in positions if p.position_id == position_id), None)
        if not target:
            return {"status": "not_found", "message": f"Position {position_id} not found"}

        price = (
            self._trading_system._execution_router._current_prices.get(target.symbol)
            if hasattr(self._trading_system, "_execution_router")
            else None
        )
        price = price or target.current_price or target.entry_price

        await self._trading_system._emit_order_exit_decision(
            position=target,
            decision="FORCE_CLOSE",
            reason_code="RISK_EXIT",
            tick_price=price,
            thresholds={
                "tp": target.take_profit,
                "sl": target.stop_loss,
                "time_stop_seconds": target.max_time_sec,
            },
            hold_seconds=None,
            note=reason,
        )
        await self._trading_system._close_position_at_price(target, price, reason=reason)
        return {"status": "closed", "position_id": position_id, "price": price}

    async def close_all_positions(self, reason: str = "panic_close_all") -> dict[str, Any]:
        """Force-close all positions (paper-safe)."""
        if not self._running or self._trading_system is None:
            return {"status": "error", "message": "Engine is not running"}
        result = await self._trading_system.panic_close_all(reason=reason)
        return {"status": "closed_all", **result}

    async def reset_paper_state(self) -> dict[str, Any]:
        """Reset paper state (positions/orders/decisions)."""
        if not self._running or self._trading_system is None:
            return {"status": "error", "message": "Engine is not running"}
        return await self._trading_system.reset_paper_state()

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
            self._state = "PAUSED"
            telemetry_service.set_engine_state("PAUSED")
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
            self._state = "RUNNING"
            telemetry_service.set_engine_state("RUNNING")
            logger.info("Engine resumed")
            return {"status": "resumed", "message": "Engine resumed"}

    async def kill(self, reason: str = "api_kill") -> dict[str, Any]:
        """Emergency kill: block new trades, cancel orders, stop engine."""
        async with self._lock:
            cancelled = 0
            if self._trading_system:
                # Stop accepting new trades
                self._trading_system._stop_trading = True
                try:
                    for order in self._trading_system._order_manager.get_open_orders():
                        order.status = OrderStatus.CANCELLED
                        if self._bus:
                            await self._bus.publish(
                                Event(event_type=EventType.ORDER_CANCELLED, data={"order": order})
                            )
                        cancelled += 1
                except Exception as e:
                    logger.warning(f"Failed to cancel open orders during kill: {e}", exc_info=True)

                try:
                    await self._trading_system.stop()
                except Exception as e:
                    logger.error(f"Error stopping trading system during kill: {e}", exc_info=True)

            self._trading_system = None
            self._running = False
            self._state = "KILLED"
            telemetry_service.set_engine_state("KILLED")
            return {
                "status": "killed",
                "message": "Engine killed and halted",
                "cancelled_orders": cancelled,
            }

    async def restart(self) -> dict[str, Any]:
        """Restart the engine (stop then start)."""
        try:
            await self.stop()
            telemetry_service.set_engine_state("STOPPED")
            start_result = await self.start()
            return {"status": "restarted", "start_result": start_result}
        except Exception as e:
            logger.error(f"Failed to restart engine: {e}", exc_info=True)
            self._state = "ERROR"
            telemetry_service.set_engine_state("ERROR")
            raise

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
        equity = accounting.get_equity()
        drawdown_amount, drawdown_pct = accounting.get_drawdown(equity)

        return {
            "killswitch_triggered": killswitch._triggered,
            "stop_trading": self._trading_system._stop_trading,
            "equity": equity,
            "daily_pnl": accounting.get_daily_pnl(equity),
            "drawdown": drawdown_amount,
            "drawdown_pct": drawdown_pct,
            "max_open_positions": settings.max_open_positions,
            "current_positions": len(accounting.get_positions()),
            "max_open_orders": settings.max_open_orders,
            "current_orders": len(self._trading_system._order_manager.get_open_orders()),
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

    @property
    def engine_state(self) -> str:
        """Return last known engine lifecycle state."""
        return self._state