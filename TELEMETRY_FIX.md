# Исправление Trading Telemetry

## Проблема
UI показывал предупреждение "TRADING TELEMETRY NOT WIRED" - данные trading_metrics не отображались.

## Исправления

### 1. WebSocket Broadcast
- ✅ Добавлен broadcast `trading_metrics` даже когда engine остановлен
- ✅ Broadcast происходит каждые 2 секунды когда engine запущен
- ✅ Broadcast происходит даже когда engine остановлен

### 2. UI Обработка данных
- ✅ Исправлена обработка событий `trading_metrics` в `useTradingData.ts`
- ✅ Добавлена поддержка формата `event.data` и `event.payload`
- ✅ Добавлена поддержка типа `trading_metrics_update`

### 3. REST API Fallback
- ✅ Добавлена функция `fetchTradingMetrics()` в `client.ts`
- ✅ Добавлена загрузка метрик через REST API при старте
- ✅ Метрики загружаются даже если WebSocket еще не подключен

### 4. WebSocket Topic
- ✅ Добавлен `trading_metrics` в тип `WsTopic`

## Проверка

### WebSocket тест:
```bash
python3 -c "
import asyncio, websockets, json
async def test():
    async with websockets.connect('ws://localhost:8000/ws') as ws:
        await ws.send(json.dumps({'action': 'subscribe', 'topic': 'trading_metrics'}))
        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        print('Received:', json.loads(msg).get('topic'))
asyncio.run(test())
"
```

### REST API тест:
```bash
curl http://localhost:8000/trading/metrics
```

## Результат
✅ Trading telemetry теперь работает через WebSocket и REST API fallback
✅ Данные отображаются на сайте даже при первом запуске
✅ Предупреждение "TRADING TELEMETRY NOT WIRED" больше не показывается
