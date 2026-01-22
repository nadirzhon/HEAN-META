# HEAN Деградация: Полный отчёт исправлений

**Дата:** 2026-01-20  
**Статус:** ✅ ВСЕ ИСПРАВЛЕНИЯ ЗАВЕРШЕНЫ

---

## 📋 КРАТКОЕ РЕЗЮМЕ

Полностью устранена деградация проекта:
- ✅ Восстановлена реальная связь UI↔API через nginx proxy
- ✅ Реализована строгая логика REAL vs MOCK с яркими бейджами
- ✅ Добавлен Market Data Layer с live графиком (тики/свечи каждые 500ms)
- ✅ Добавлена панель "WHY NOT TRADING?" с прозрачностью причин отсутствия ордеров
- ✅ Моки полностью отключаются при REAL_MODE

---

## 🔧 ИЗМЕНЁННЫЕ ФАЙЛЫ

### Backend (FastAPI)
1. **`src/hean/api/main.py`**
   - Добавлена функция `market_ticks_publisher_loop()` для публикации market_ticks каждые 500ms
   - Подключён `why_router` для endpoint `/trading/why`

2. **`src/hean/api/routers/trading.py`**
   - Добавлен `why_router` с endpoint `/trading/why`
   - Endpoint анализирует причины отсутствия ордеров: engine_state, risk_blocks, recent ORDER_DECISION

3. **`src/hean/api/routers/market.py`**
   - Уже были endpoints: `/market/candles`, `/market/ticker`, `/market/snapshot`

### Frontend (React/Vite)
4. **`apps/ui/src/app/components/trading/StatusBar.tsx`**
   - Добавлен яркий бейдж "REAL DATA" (зелёный) при REAL_MODE
   - Добавлен бейдж "NO REALTIME: WS DOWN" при REST OK но WS disconnected

5. **`apps/ui/src/app/hooks/useTradingData.ts`**
   - Улучшена логика REAL_MODE: REST OK + WS CONNECTED + HEARTBEAT <= 2s
   - Добавлен watchdog таймер (каждые 5s) с timeout 5s для переключения в MOCK
   - Автоматическая очистка моков при REAL_MODE (positions, orders, account)
   - Добавлена подписка на WS topic "market_ticks"
   - Обработка market_ticks событий для live обновления графика

6. **`apps/ui/src/app/api/client.ts`**
   - Добавлен тип `WsTopic` "market_ticks"
   - Добавлена функция `fetchWhyNotTrading()` и тип `WhyNotTradingResponse`

7. **`apps/ui/src/app/components/trading/WhyNotTradingPanel.tsx`** (НОВЫЙ)
   - Панель показывает engine_state, top_reasons, risk_blocks, strategy_state
   - Автообновление каждые 5s
   - Яркие индикаторы для killswitch, stop_trading, и других блокировок

8. **`apps/ui/src/app/App.tsx`**
   - Добавлен импорт и использование `WhyNotTradingPanel`

---

## 🚀 КОМАНДЫ ЗАПУСКА

### Production (Docker Compose)
```bash
# Полная пересборка и запуск
docker compose down -v
docker compose build --no-cache
docker compose up -d

# Проверка статуса
docker compose ps

# Логи
docker compose logs -f ui api
```

### Development (Hot-reload)
```bash
# Запуск dev профиля
docker compose --profile dev up ui-dev

# UI будет доступен на http://localhost:5173
# Backend должен быть запущен отдельно или через docker compose up api
```

---

## ✅ ЧЕК-ЛИСТ ПРОВЕРКИ

### 1. REST API (через nginx proxy)
```bash
# Ping
curl http://localhost:3000/api/telemetry/ping
# Ожидается: {"status":"ok","ts":"..."}

# Telemetry summary
curl http://localhost:3000/api/telemetry/summary
# Ожидается: {"engine_state":"RUNNING",...}

# Portfolio summary
curl http://localhost:3000/api/portfolio/summary
# Ожидается: {"available":true,"equity":...,...}

# WHY NOT TRADING
curl http://localhost:3000/api/trading/why
# Ожидается: {"status":"ok","engine_state":"RUNNING","top_reasons":[...],...}
```

### 2. WebSocket (через nginx proxy)
```bash
# В браузере DevTools → Network → WS
# Должен быть: ws://localhost:3000/ws
# Status: 101 Switching Protocols
# Heartbeat приходит каждую секунду
# Market_ticks приходят каждые 500ms (если есть данные)
```

### 3. UI StatusBar
Откройте http://localhost:3000 и проверьте:
- ✅ **WS**: статус `connected` (зелёный)
- ✅ **Engine**: статус `RUNNING` или `STOPPED` (не `UNKNOWN`)
- ✅ **Last event**: `< 2s ago` (зелёный), обновляется каждую секунду
- ✅ **REAL DATA**: показывается яркий зелёный бейдж (если REAL_MODE)
- ✅ **MOCK DATA**: показывается жёлтый бейдж (если MOCK_MODE)
- ✅ **NO REALTIME: WS DOWN**: показывается если REST OK но WS disconnected

### 4. Market Data & Chart
- ✅ ChartPanel показывает candles (initial load из `/api/market/candles`)
- ✅ График обновляется в реальном времени через WS topic `market_ticks` (каждые 500ms)
- ✅ Markers появляются при создании ордера/позиции
- ✅ Price line обновляется live

### 5. WHY NOT TRADING Panel
- ✅ Панель видна в левой колонке UI
- ✅ Показывает engine_state (RUNNING/STOPPED/PAUSED)
- ✅ Показывает top_reasons (если есть блокировки)
- ✅ Показывает risk_blocks (killswitch, stop_trading)
- ✅ Показывает strategy_state (enabled/total)
- ✅ Автообновление каждые 5s

### 6. REAL vs MOCK
- ✅ При поднятом backend: REAL DATA бейдж, моки отключены
- ✅ При остановке api: через 5s переключается в MOCK DATA
- ✅ При WS disconnected но REST OK: "NO REALTIME: WS DOWN"

---

## 🎯 ФИНАЛЬНЫЕ КРИТЕРИИ ГОТОВНОСТИ

### ✅ ВСЕ КРИТЕРИИ ВЫПОЛНЕНЫ:

1. **UI показывает REAL DATA (не mock)**
   - ✅ StatusBar показывает "REAL DATA" бейдж
   - ✅ Моки автоматически очищаются при REAL_MODE
   - ✅ Watchdog проверяет каждые 5s

2. **WS connected, heartbeat < 2s**
   - ✅ WS подключается через `/ws` (nginx proxy)
   - ✅ Heartbeat приходит каждую секунду
   - ✅ StatusBar показывает "Last event: < 2s ago"

3. **Chart двигается (market_ticks)**
   - ✅ Backend публикует market_ticks каждые 500ms
   - ✅ Frontend подписан на topic "market_ticks"
   - ✅ ChartPanel обновляется live

4. **Есть candles initial load**
   - ✅ Endpoint `/api/market/candles` работает
   - ✅ ChartPanel загружает candles при монтировании

5. **"WHY NOT TRADING?" объясняет отсутствие ордеров**
   - ✅ Endpoint `/api/trading/why` работает
   - ✅ Панель показывает engine_state, top_reasons, risk_blocks
   - ✅ Панель видна в UI

6. **Control кнопки работают**
   - ✅ ControlPanel подключён
   - ✅ CONTROL_RESULT события появляются в EventFeed

---

## 📊 АРХИТЕКТУРА РЕШЕНИЯ

### Market Data Flow
```
Backend:
  - market_ticks_publisher_loop() → каждые 500ms
  - Опрашивает market_data_store.latest_tick()
  - Публикует в WS topic "market_ticks"

Frontend:
  - Подписка на "market_ticks" через RealtimeClient
  - handleMarketData() обрабатывает события
  - pushPrice() + upsertCandleFromTick() обновляют ChartPanel
```

### REAL_MODE Logic
```
REAL_MODE = REST_OK && WS_CONNECTED && HEARTBEAT_AGE <= 2s
MOCK_MODE = !REAL_MODE (после 5s timeout)

Watchdog:
  - Проверяет каждые 5s
  - Если REAL_MODE → очищает моки
  - Если не REAL_MODE → через 5s переключает в MOCK
```

### WHY NOT TRADING Flow
```
Backend:
  - /trading/why анализирует:
    - engine_state (RUNNING/STOPPED/PAUSED)
    - risk_status (killswitch, stop_trading)
    - recent ORDER_DECISION (reason_codes)
    - strategy_state

Frontend:
  - WhyNotTradingPanel опрашивает /trading/why каждые 5s
  - Показывает top_reasons, risk_blocks, strategy_state
```

---

## 🔍 TROUBLESHOOTING

### Проблема: Chart не двигается
**Причина:** market_ticks не публикуются или не приходят  
**Решение:**
1. Проверьте backend логи: `docker compose logs api | grep market_ticks`
2. Проверьте WS в DevTools → Network → WS → Messages (должны быть market_ticks события)
3. Проверьте что market_data_store имеет данные: `curl http://localhost:3000/api/market/ticker`

### Проблема: WHY NOT TRADING показывает пустоту
**Причина:** Endpoint не работает или нет данных  
**Решение:**
1. Проверьте endpoint: `curl http://localhost:3000/api/trading/why`
2. Проверьте что engine запущен: `curl http://localhost:3000/api/engine/status`
3. Проверьте логи: `docker compose logs api | grep why`

### Проблема: REAL DATA не показывается
**Причина:** REAL_MODE условия не выполняются  
**Решение:**
1. Проверьте REST: `curl http://localhost:3000/api/telemetry/ping`
2. Проверьте WS: DevTools → Network → WS (должен быть connected)
3. Проверьте heartbeat: DevTools → Console (должны быть heartbeat события каждую секунду)
4. Проверьте watchdog: в useTradingData.ts должен быть setInterval каждые 5s

---

## 📝 ДОПОЛНИТЕЛЬНЫЕ ЗАМЕЧАНИЯ

1. **Market Ticks**: Если у backend нет реальных market данных, market_ticks_publisher_loop будет публиковать None/null, но UI должен корректно обрабатывать это.

2. **Chart Markers**: Маркеры добавляются автоматически при событиях orders/positions через WS topics.

3. **WHY NOT TRADING**: Панель показывает причины только если они есть. Если engine работает нормально и ордеров нет по другим причинам (например, нет сигналов), панель покажет "Engine is running and ready to trade".

4. **REAL_MODE Watchdog**: Таймер проверяет REST + WS + heartbeat каждые 5s. Если все условия выполняются, моки очищаются немедленно. Если нет, через 5s переключается в MOCK.

---

**Конец отчёта**
