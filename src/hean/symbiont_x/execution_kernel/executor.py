"""
Execution Kernel - Ядро исполнения

Python имплементация (будет заменена на Rust для max скорости)
"""

import time
import uuid
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    """Сторона ордера"""
    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    """Тип ордера"""
    MARKET = "Market"
    LIMIT = "Limit"
    POST_ONLY = "PostOnly"


class OrderStatus(Enum):
    """Статус ордера"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderRequest:
    """Запрос на ордер"""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None  # For limit orders
    strategy_id: str | None = None

    # Advanced params
    reduce_only: bool = False
    post_only: bool = False
    time_in_force: str = "GTC"  # GoodTilCancelled

    # Internal
    request_id: str = ""
    requested_at_ns: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if self.requested_at_ns == 0:
            self.requested_at_ns = time.time_ns()


@dataclass
class OrderResult:
    """Результат исполнения ордера"""

    request_id: str
    order_id: str | None = None

    status: OrderStatus = OrderStatus.PENDING

    # Execution details
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    filled_at_ns: int = 0

    # Performance
    submission_latency_ns: int = 0  # Time to submit
    execution_latency_ns: int = 0   # Time to fill

    # Errors
    error_code: str | None = None
    error_message: str | None = None

    def get_submission_latency_ms(self) -> float:
        """Latency до submission в ms"""
        return self.submission_latency_ns / 1_000_000

    def get_execution_latency_ms(self) -> float:
        """Latency до execution в ms"""
        return self.execution_latency_ns / 1_000_000


class ExecutionKernel:
    """
    Execution Kernel

    Отвечает за ultra-low latency исполнение ордеров

    TODO: Реализовать на Rust для максимальной скорости
    """

    def __init__(self, exchange_connector: object) -> None:
        self.exchange_connector = exchange_connector

        # State
        self.pending_orders: dict[str, OrderRequest] = {}
        self.order_results: dict[str, OrderResult] = {}

        # Statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0
        self.total_latency_ns = 0

    async def execute_order(self, request: OrderRequest) -> OrderResult:
        """
        Исполняет ордер

        Returns OrderResult
        """

        start_ns = time.time_ns()

        # Validate request
        if not self._validate_order(request):
            return OrderResult(
                request_id=request.request_id,
                status=OrderStatus.REJECTED,
                error_code="VALIDATION_FAILED",
                error_message="Order validation failed"
            )

        # Add to pending
        self.pending_orders[request.request_id] = request

        # Submit to exchange
        try:
            # Call exchange API
            if hasattr(self.exchange_connector, 'place_order'):
                # Real exchange integration
                exchange_result = await self._submit_to_exchange(request)

                submission_ns = time.time_ns()
                submission_latency = submission_ns - start_ns

                # Extract order ID from exchange response
                order_id = exchange_result.get('orderId') or exchange_result.get('order_id')

                # Check if immediately filled (for market orders)
                is_filled = exchange_result.get('status') == 'Filled'
                filled_quantity = float(exchange_result.get('cumExecQty', 0))
                filled_price = float(exchange_result.get('avgPrice', request.price or 0))

                filled_at_ns = time.time_ns() if is_filled else 0
                execution_latency = filled_at_ns - start_ns if is_filled else 0

                status = OrderStatus.FILLED if is_filled else OrderStatus.SUBMITTED

                result = OrderResult(
                    request_id=request.request_id,
                    order_id=order_id,
                    status=status,
                    filled_quantity=filled_quantity,
                    filled_price=filled_price,
                    filled_at_ns=filled_at_ns,
                    submission_latency_ns=submission_latency,
                    execution_latency_ns=execution_latency,
                )
            else:
                # Fallback: simulated execution (for testing without exchange)
                order_id = f"sim_{uuid.uuid4().hex[:8]}"

                submission_ns = time.time_ns()
                submission_latency = submission_ns - start_ns

                # Simulate execution
                filled_at_ns = time.time_ns()
                execution_latency = filled_at_ns - start_ns

                result = OrderResult(
                    request_id=request.request_id,
                    order_id=order_id,
                    status=OrderStatus.FILLED,
                    filled_quantity=request.quantity,
                    filled_price=request.price if request.price else 50000.0,  # Simulated
                    filled_at_ns=filled_at_ns,
                    submission_latency_ns=submission_latency,
                    execution_latency_ns=execution_latency,
                )

            # Update stats
            self.total_orders += 1
            self.filled_orders += 1
            self.total_latency_ns += execution_latency

            # Store result
            self.order_results[request.request_id] = result

            # Remove from pending
            del self.pending_orders[request.request_id]

            return result

        except Exception as e:
            # Handle errors
            result = OrderResult(
                request_id=request.request_id,
                status=OrderStatus.REJECTED,
                error_code="EXECUTION_ERROR",
                error_message=str(e),
            )

            self.rejected_orders += 1
            self.order_results[request.request_id] = result

            return result

    def _validate_order(self, request: OrderRequest) -> bool:
        """
        Валидирует ордер перед исполнением

        Проверяет:
        - Базовые параметры (quantity, price)
        - Лимиты позиций
        - Доступный капитал
        - Рыночные часы
        - Параметры символа
        """

        # Basic validation
        if request.quantity <= 0:
            return False

        if request.order_type == OrderType.LIMIT and not request.price:
            return False

        if request.order_type == OrderType.LIMIT and (request.price is not None and request.price <= 0):
            return False

        # Symbol validation
        if not request.symbol or len(request.symbol) < 3:
            return False

        # Check for valid side
        if request.side not in [OrderSide.BUY, OrderSide.SELL]:
            return False

        # Position limits validation
        if not self._check_position_limits(request):
            return False

        # Capital validation
        if not self._check_capital_available(request):
            return False

        # Market hours validation (if applicable)
        if not self._check_market_hours(request):
            return False

        return True

    def _check_position_limits(self, request: OrderRequest) -> bool:
        """
        Проверяет лимиты позиций

        Предотвращает превышение максимального размера позиции
        """
        # Get current position for symbol (would need position manager integration)
        # For now, always return True - integration point for position manager

        # Example implementation:
        # current_position = self.position_manager.get_position(request.symbol)
        # max_position_size = self.risk_limits.get_max_position(request.symbol)
        #
        # if request.side == OrderSide.BUY:
        #     new_position = current_position + request.quantity
        # else:
        #     new_position = current_position - request.quantity
        #
        # if abs(new_position) > max_position_size:
        #     return False

        return True

    def _check_capital_available(self, request: OrderRequest) -> bool:
        """
        Проверяет доступность капитала для ордера

        Предотвращает открытие позиций без достаточного капитала
        """
        # Calculate required capital
        # For now, always return True - integration point for account manager

        # Example implementation:
        # if request.order_type == OrderType.LIMIT:
        #     required_capital = request.quantity * request.price
        # else:
        #     # For market orders, estimate using last price
        #     last_price = self.market_data.get_last_price(request.symbol)
        #     required_capital = request.quantity * last_price
        #
        # available_capital = self.account_manager.get_available_capital()
        # if required_capital > available_capital:
        #     return False

        return True

    def _check_market_hours(self, request: OrderRequest) -> bool:
        """
        Проверяет торговые часы

        Для крипто-рынков обычно 24/7, но может быть полезно
        для проверки технических работ биржи
        """
        # Crypto markets are 24/7, but can check exchange status
        # For now, always return True

        # Example implementation:
        # exchange_status = self.exchange_connector.get_status()
        # if exchange_status != "ONLINE":
        #     return False

        return True

    async def _submit_to_exchange(self, request: OrderRequest) -> dict[str, object]:
        """
        Отправляет ордер на биржу

        Args:
            request: Запрос на ордер

        Returns:
            Ответ биржи с order_id и статусом
        """
        # Prepare order params for exchange
        order_params: dict[str, object] = {
            'symbol': request.symbol,
            'side': request.side.value,
            'orderType': request.order_type.value,
            'qty': str(request.quantity),
            'timeInForce': request.time_in_force,
        }

        # Add price for limit orders
        if request.order_type == OrderType.LIMIT and request.price:
            order_params['price'] = str(request.price)

        # Add advanced params
        if request.reduce_only:
            order_params['reduceOnly'] = True

        if request.post_only or request.order_type == OrderType.POST_ONLY:
            order_params['postOnly'] = True

        # Submit to exchange
        try:
            result = await self.exchange_connector.place_order(order_params)
            return dict(result) if result else {}
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Exchange order submission failed: {e}") from e

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        """
        Отменяет ордер на бирже

        Args:
            order_id: ID ордера на бирже
            symbol: Символ (требуется для некоторых бирж)

        Returns:
            True если отменен успешно, False иначе
        """
        # Find request by order_id
        target_request_id = None
        target_symbol = symbol

        for request_id, request in list(self.pending_orders.items()):
            result = self.order_results.get(request_id)
            if result and result.order_id == order_id:
                target_request_id = request_id
                if not target_symbol:
                    target_symbol = request.symbol
                break

        if not target_request_id:
            # Order not found in pending
            return False

        # Call exchange API if available
        if hasattr(self.exchange_connector, 'cancel_order'):
            try:
                # Real cancellation via exchange
                await self.exchange_connector.cancel_order(
                    order_id=order_id,
                    symbol=target_symbol
                )

                # Update local state
                result = self.order_results.get(target_request_id)
                if result:
                    result.status = OrderStatus.CANCELLED

                # Remove from pending
                if target_request_id in self.pending_orders:
                    del self.pending_orders[target_request_id]

                return True

            except Exception as e:
                # Cancellation failed
                result = self.order_results.get(target_request_id)
                if result:
                    result.error_code = "CANCEL_FAILED"
                    result.error_message = str(e)
                return False
        else:
            # Fallback: simulated cancellation
            result = self.order_results.get(target_request_id)
            if result:
                result.status = OrderStatus.CANCELLED

            if target_request_id in self.pending_orders:
                del self.pending_orders[target_request_id]

            return True

    def get_order_result(self, request_id: str) -> OrderResult | None:
        """Получает результат ордера"""
        return self.order_results.get(request_id)

    def get_pending_orders(self) -> list[OrderRequest]:
        """Возвращает pending ордера"""
        return list(self.pending_orders.values())

    def get_statistics(self) -> dict:
        """Статистика исполнения"""

        avg_latency_ns = (
            self.total_latency_ns / self.filled_orders
            if self.filled_orders > 0 else 0
        )

        fill_rate = (
            self.filled_orders / self.total_orders
            if self.total_orders > 0 else 0
        )

        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'rejected_orders': self.rejected_orders,
            'pending_orders': len(self.pending_orders),
            'fill_rate': fill_rate,
            'avg_latency_ns': avg_latency_ns,
            'avg_latency_ms': avg_latency_ns / 1_000_000,
            'avg_latency_us': avg_latency_ns / 1_000,
        }
