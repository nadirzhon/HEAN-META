# Промпт: Диагностика и исправление проблемы с заполнением ордеров

## Контекст проблемы

Изучается цепочка обработки сигналов и создания сделок. Обнаружена проблема: ордера создаются, но не заполняются.

## Цепочка обработки ордеров

1. **ORDER_REQUEST** → EventBus публикует событие
2. **ExecutionRouter._handle_order_request** → получает событие
3. **ExecutionRouter._route_maker_first** или **_route_to_paper** → создает Order со статусом `PENDING`
4. **PaperBroker.submit_order** → получает ордер, меняет статус на `PLACED`, добавляет в `_pending_orders`
5. **PaperBroker._fill_orders_loop** → периодически проверяет `_pending_orders` (каждые 100ms)
6. **PaperBroker._process_pending_orders** → пытается заполнить ордера
7. **PaperBroker._fill_order** → заполняет ордер и публикует `ORDER_FILLED`

## Внесенные исправления

### 1. Логирование в PaperBroker (`paper_broker.py`)

**При старте:**
- `[FORCED_BROKER] Paper broker started, _fill_orders_loop task created, _running={self._running}`

**В `_fill_orders_loop`:**
- При старте: `[FORCED_BROKER] _fill_orders_loop started, _running={self._running}`
- Каждые 10 секунд (100 итераций): `[FORCED_BROKER] _fill_orders_loop iteration {loop_count}, pending_orders={len(self._pending_orders)}`

**В `submit_order`:**
- `[FORCED_BROKER] submit_order called: order_id={order.order_id}, symbol={order.symbol}, side={order.side}, size={order.size}, price={order.price}, current_status={order.status}`
- Если статус PENDING: `[FORCED_BROKER] Changed order status from PENDING to PLACED`
- `[FORCED_BROKER] Order added to _pending_orders, total pending: {len(self._pending_orders)}, final_status={order.status}`

**В `_process_pending_orders`:**
- `[FORCED_BROKER] _process_pending_orders: {pending_count} pending orders`
- Для каждого ордера: `[FORCED_BROKER] Pending order: {order_id}, status={order.status}, symbol={order.symbol}, side={order.side}, size={order.size}`
- Если статус PENDING: `[FORCED_BROKER] Order {order_id} has PENDING status, changing to PLACED`

**В `_fill_order`:**
- `[FORCED_FILL] _fill_order called: order_id={order.order_id}, fill_price={fill_price:.2f}, current_filled_size={order.filled_size}, order_size={order.size}`
- `[FORCED_FILL] remaining_size={remaining_size}`
- `[FORCED_FILL] is_maker={is_maker}, order_type={order.order_type}, is_maker_flag={order.is_maker}, fill_price={fill_price:.2f}, order_price={order.price}`
- `[FORCED_FILL] fee_rate={fee_rate}, fee={fee:.4f}`
- `[FORCED_FILL] After fill: filled_size={order.filled_size}, avg_fill_price={order.avg_fill_price:.2f}`
- `[FORCED_FILL] Order fully filled, removed from _pending_orders` или `[FORCED_FILL] Order partially filled, remaining: {order.size - order.filled_size}`
- `[FORCED_FILL] Publishing ORDER_FILLED event for order {order.order_id}`
- `[FORCED_FILL] ORDER_FILLED event published successfully`

### 2. Логирование в ExecutionRouter (`router.py`)

**В `_handle_order_request`:**
- `[FORCED_ROUTER] _handle_order_request called: {order_request.symbol} {order_request.side} size={order_request.size:.6f}`
- `[FORCED_ROUTER] Routing to maker_first` или `[FORCED_ROUTER] Routing to paper`

**В `_route_maker_first` и `_route_to_paper`:**
- `[FORCED_ROUTER] Calling _paper_broker.submit_order for order {order.order_id}`
- `[FORCED_ROUTER] _paper_broker.submit_order completed for order {order.order_id}`

**В `_route_to_paper`:**
- `[FORCED_ROUTER] Calling _paper_broker.submit_order for order {order.order_id} (route_to_paper)`
- `[FORCED_ROUTER] _paper_broker.submit_order completed for order {order.order_id} (route_to_paper)`

### 3. Логирование в EventBus (`bus.py`)

**В `subscribe`:**
- При подписке на ORDER_REQUEST: `[FORCED_BUS] Subscribed ORDER_REQUEST handler: {handler.__name__}, total subscribers: {len(self._subscribers[event_type])}`

**В `publish`:**
- `[FORCED_BUS] Publishing ORDER_REQUEST event to queue` (для ORDER_REQUEST)
- `[FORCED_BUS] Publishing SIGNAL event to queue` (для SIGNAL)

**В `_dispatch`:**
- Если нет подписчиков: `[FORCED_BUS] No subscribers for {event.event_type}`
- Если нет подписчиков для ORDER_REQUEST: `[FORCED_BUS] CRITICAL: ORDER_REQUEST has no subscribers!`
- `[FORCED_BUS] Dispatching {event.event_type} to {len(handlers)} handlers`
- Для ORDER_REQUEST: `[FORCED_BUS] ORDER_REQUEST handlers: {[h.__name__ for h in handlers]}`

**В `_safe_call_handler`:**
- `[FORCED_BUS] Calling handler {handler.__name__} for {event.event_type}`
- `[FORCED_BUS] Handler {handler.__name__} completed successfully`
- При ошибке: `[FORCED_BUS] Handler {handler.__name__} raised exception for {event.event_type}: {e}`

### 3. Исправления логики

**В `PaperBroker.submit_order`:**
- Если ордер приходит со статусом `PENDING`, статус меняется на `PLACED`
- Немедленная попытка заполнения сразу после добавления в `_pending_orders`:
  ```python
  try:
      await self._process_pending_orders()
  except Exception as e:
      logger.error(f"[FORCED_BROKER] Error in immediate fill attempt: {e}", exc_info=True)
  ```

**В `PaperBroker._process_pending_orders`:**
- Обработка статуса `PENDING` на случай, если ордер все еще имеет этот статус
- Принудительное заполнение всех ордеров со статусом `PLACED` или `PARTIALLY_FILLED` (для отладки)

## Что проверить в логах

При запуске теста проверьте наличие следующих сообщений в порядке появления:

1. ✅ `[FORCED_BUS] Publishing ORDER_REQUEST event to queue` — событие опубликовано
2. ✅ `[FORCED_BUS] Dispatching ORDER_REQUEST to X handlers` — событие отправлено подписчикам
3. ✅ `[FORCED_ROUTER] _handle_order_request called` — ExecutionRouter получил событие
4. ✅ `[FORCED_ROUTER] Calling _paper_broker.submit_order` — вызов submit_order
5. ✅ `[FORCED_BROKER] submit_order called` — PaperBroker получил ордер
6. ✅ `[FORCED_BROKER] Order added to _pending_orders` — ордер добавлен в очередь
7. ✅ `[FORCED_BROKER] _process_pending_orders: X pending orders` — обработка началась
8. ✅ `[FORCED_FILL] Filling order` — попытка заполнения
9. ✅ `[FORCED_FILL] Publishing ORDER_FILLED event` — событие заполнения опубликовано
10. ✅ `[FORCED_FILL] ORDER_FILLED event published successfully` — событие опубликовано успешно

## Диагностика

Если какое-то сообщение отсутствует, это укажет, где обрывается цепочка:

- **Отсутствует сообщение 1-2**: проблема в EventBus
- **Отсутствует сообщение 3**: ExecutionRouter не подписан на ORDER_REQUEST
- **Отсутствует сообщение 4**: проблема в `_route_maker_first` или `_route_to_paper`
- **Отсутствует сообщение 5**: `submit_order` не вызывается или вызывается с ошибкой
- **Отсутствует сообщение 6**: проблема при добавлении в `_pending_orders`
- **Отсутствует сообщение 7**: `_process_pending_orders` не вызывается или `_pending_orders` пуст
- **Отсутствует сообщение 8**: проблема в логике заполнения (нет цены, неправильные условия)
- **Отсутствует сообщение 9-10**: проблема при публикации события `ORDER_FILLED`

## Дополнительные проверки

1. Проверьте, что `PaperBroker.start()` вызывается и `_fill_orders_loop` запущен
2. Проверьте, что есть тики (TICK events) для обновления цен
3. Проверьте, что `_current_prices` содержит цены для символов ордеров
4. Проверьте, что нет исключений в логах между сообщениями

## Следующие шаги

1. Запустите тест и соберите логи
2. Найдите первое отсутствующее сообщение из списка выше
3. Изучите код в соответствующем месте
4. Проверьте условия, которые могут блокировать выполнение
5. Добавьте дополнительное логирование, если нужно

