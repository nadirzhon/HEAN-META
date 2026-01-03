# Промпт: Отладка проблемы с обнаружением заполненных ордеров

## Проблема

`BacktestMetrics.calculate()` вызывает `OrderManager.get_filled_orders()` и получает пустой список, хотя ордера должны быть заполнены. В логах отсутствуют сообщения `[FORCED_FILL]`, что указывает на возможную проблему в цепочке выполнения.

## Анализ цепочки выполнения ордеров

### Что работает ✅

1. **Сигналы генерируются** ✅
   - В логах видны сигналы от стратегий

2. **OrderRequest создаются и публикуются** ✅
   - `_handle_signal` создает `OrderRequest` и публикует в EventBus

3. **Order регистрируются в OrderManager** ✅
   - В `ExecutionRouter._route_maker_first()` (строка 243) ордер регистрируется:
     ```python
     self._order_manager.register_order(order)
     ```

4. **Order отправляются в PaperBroker** ✅
   - В `ExecutionRouter._route_maker_first()` (строка 256) вызывается:
     ```python
     await self._paper_broker.submit_order(order)
     ```

### Что нужно проверить ❓❌

5. **PaperBroker заполняет ордера** ❓
   - В `PaperBroker._fill_order()` (строка 299) статус обновляется:
     ```python
     order.status = OrderStatus.FILLED
     ```
   - В логах нет сообщений `[FORCED_FILL]`, возможно, ордера не заполняются
   - Метод `_process_pending_orders()` вызывается из `_fill_orders_loop()`, который запускается при старте брокера

6. **Статус обновляется в OrderManager** ❓
   - Обновление идет напрямую в объекте `Order`, который хранится в `OrderManager`
   - Если это тот же объект (та же ссылка), обновление должно работать
   - **ПОТЕНЦИАЛЬНАЯ ПРОБЛЕМА**: Если объект копируется при передаче, изменения не будут видны в OrderManager

7. **BacktestMetrics находит заполненные ордера** ❌
   - `BacktestMetrics.calculate()` (строка 83) вызывает:
     ```python
     filled_orders = self._execution_router._order_manager.get_filled_orders()
     ```
   - Возвращается пустой список

## Возможные проблемы

### 1. Ордера не заполняются в PaperBroker
**Симптомы:**
- В логах нет сообщений `[FORCED_FILL]`
- `_process_pending_orders()` может не вызываться или пропускать ордера

**Что проверить:**
- Проверить, что `PaperBroker.start()` вызывается и создает `_fill_orders_loop` task
- Проверить, что `_process_pending_orders()` вызывается периодически
- Проверить, что ордера действительно попадают в `_pending_orders` после `submit_order()`
- Проверить условия пропуска в `_process_pending_orders()` (строки 149-151)

### 2. Проблема с объектными ссылками
**Симптомы:**
- Ордера заполняются в PaperBroker (есть логи `[FORCED_FILL]`)
- Но в OrderManager они остаются со старым статусом

**Что проверить:**
- Убедиться, что в `ExecutionRouter._route_maker_first()` передается тот же объект Order, что регистрируется в OrderManager
- Проверить, что `Order` - это Pydantic BaseModel и изменения статуса должны быть видны везде, если используется та же ссылка
- **ВНИМАНИЕ**: Pydantic модели могут создавать копии при некоторых операциях

### 3. Проблема с сравнением OrderStatus
**Симптомы:**
- Статус обновлен на `OrderStatus.FILLED`
- Но `get_filled_orders()` не находит ордер

**Что проверить:**
- `OrderStatus` - это `str, Enum`, сравнение должно работать через `==`
- Проверить, что `order.status == OrderStatus.FILLED` действительно True
- Возможно, статус сохраняется как строка `"filled"`, а сравнивается с Enum

## Рекомендации по отладке

### Шаг 1: Добавить детальное логирование в `OrderManager.get_filled_orders()`

Добавить в `src/hean/execution/order_manager.py` в метод `get_filled_orders()`:

```python
def get_filled_orders(self, since: datetime | None = None) -> list[Order]:
    """Get all filled orders, optionally filtered by time."""
    logger.info(f"[DEBUG] OrderManager.get_filled_orders: total orders={len(self._orders)}")
    
    # Вывести информацию о всех ордерах
    for order_id, order in self._orders.items():
        logger.info(
            f"[DEBUG] Order {order_id}: "
            f"status={order.status}, "
            f"status_type={type(order.status)}, "
            f"status_repr={repr(order.status)}, "
            f"OrderStatus.FILLED={OrderStatus.FILLED}, "
            f"status == FILLED={order.status == OrderStatus.FILLED}, "
            f"filled_size={order.filled_size}, "
            f"size={order.size}"
        )
    
    filled = [order for order in self._orders.values() if order.status == OrderStatus.FILLED]
    logger.info(f"[DEBUG] OrderManager.get_filled_orders: found {len(filled)} filled orders")
    
    if since:
        filled_before_filter = len(filled)
        filled = [order for order in filled if order.timestamp >= since]
        logger.info(f"[DEBUG] OrderManager.get_filled_orders: after time filter: {len(filled)}/{filled_before_filter}")
    
    return filled
```

### Шаг 2: Добавить логирование в `PaperBroker._fill_order()`

Убедиться, что в `src/hean/execution/paper_broker.py` в методе `_fill_order()` есть логирование после обновления статуса:

```python
if order.filled_size >= order.size:
    order.status = OrderStatus.FILLED
    self._pending_orders.pop(order.order_id, None)
    logger.info(
        f"[FORCED_FILL] Order {order.order_id} fully filled, "
        f"status set to {order.status}, "
        f"status_type={type(order.status)}, "
        f"id(order)={id(order)}"
    )
```

### Шаг 3: Добавить логирование в `ExecutionRouter._route_maker_first()`

В `src/hean/execution/router.py` после регистрации ордера добавить:

```python
# Register with order manager
self._order_manager.register_order(order)
logger.info(
    f"[DEBUG_ROUTER] Order {order.order_id} registered in OrderManager, "
    f"id(order)={id(order)}, "
    f"status={order.status}"
)

# Получить ордер обратно и проверить, что это тот же объект
retrieved_order = self._order_manager.get_order(order.order_id)
if retrieved_order is not None:
    logger.info(
        f"[DEBUG_ROUTER] Retrieved order from OrderManager: "
        f"id(retrieved_order)={id(retrieved_order)}, "
        f"same_object={retrieved_order is order}"
    )
```

### Шаг 4: Проверить вызов `_process_pending_orders()`

В `src/hean/execution/paper_broker.py` в методе `_fill_orders_loop()` добавить логирование:

```python
async def _fill_orders_loop(self) -> None:
    """Continuously process pending orders."""
    logger.info("[FORCED_BROKER] _fill_orders_loop started")
    while self._running:
        try:
            logger.debug(f"[FORCED_BROKER] _fill_orders_loop iteration, pending orders: {len(self._pending_orders)}")
            await self._process_pending_orders()
            await asyncio.sleep(0.1)  # Check every 100ms
        except Exception as e:
            logger.error(f"[FORCED_BROKER] Error in fill loop: {e}", exc_info=True)
            await asyncio.sleep(1)
    logger.info("[FORCED_BROKER] _fill_orders_loop stopped")
```

## Что проверить в первую очередь

1. **Запускается ли `_fill_orders_loop`?**
   - Проверить логи при старте PaperBroker
   - Должно быть сообщение `[FORCED_BROKER] _fill_orders_loop started`

2. **Попадают ли ордера в `_pending_orders`?**
   - В `submit_order()` должно быть логирование: `[FORCED_BROKER] Order added to _pending_orders`

3. **Вызывается ли `_process_pending_orders()`?**
   - Должны быть логи каждые 100ms с количеством pending ордеров

4. **Заполняются ли ордера?**
   - Должны быть логи `[FORCED_FILL] Filling order ...` и `[FORCED_FILL] Order fully filled`

5. **Те же ли объекты в OrderManager и PaperBroker?**
   - Проверить `id(order)` до и после заполнения
   - Если ID разные - проблема с копированием объектов

## Ожидаемый результат

После добавления логирования должно стать ясно:
- Заполняются ли ордера вообще
- Обновляется ли статус в том же объекте, что хранится в OrderManager
- Почему `get_filled_orders()` не находит заполненные ордера

## Файлы для изменения

1. `src/hean/execution/order_manager.py` - метод `get_filled_orders()`
2. `src/hean/execution/paper_broker.py` - методы `_fill_order()` и `_fill_orders_loop()`
3. `src/hean/execution/router.py` - метод `_route_maker_first()`





