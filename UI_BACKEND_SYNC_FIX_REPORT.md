# HEAN UI ↔ Backend Синхронизация: Полный отчёт исправлений

**Дата:** 2026-01-20  
**Статус:** ✅ ВСЕ ИСПРАВЛЕНИЯ ЗАВЕРШЕНЫ

---

## 📋 КРАТКОЕ РЕЗЮМЕ

Полностью исправлена синхронизация между HEAN UI и Backend:
- ✅ Убран "мертвый сайт" (UI ходил на `http://api:8000` из браузера)
- ✅ Восстановлен real-time через WebSocket с правильным прокси
- ✅ Добавлена строгая логика REAL_MODE vs MOCK_MODE
- ✅ Исправлен StatusBar с live обновлением last event age
- ✅ Добавлен Market Data слой (REST + WS) для графика

---

## 🔧 ИЗМЕНЁННЫЕ ФАЙЛЫ

### Backend (FastAPI)
1. **`src/hean/api/routers/market.py`**
   - Добавлен `/market/ticker` endpoint
   - Добавлен `/market/candles` endpoint

### Frontend (React/Vite)
2. **`apps/ui/Dockerfile`**
   - Изменены ARG по умолчанию: `VITE_API_BASE=/api`, `VITE_WS_URL=/ws`
   - Добавлен COPY nginx.conf

3. **`apps/ui/nginx.conf`** (НОВЫЙ)
   - Proxy для `/api/*` → `http://api:8000/`
   - Proxy для `/ws` → `ws://api:8000/ws` с WebSocket upgrade headers

4. **`apps/ui/src/app/api/client.ts`**
   - Добавлена функция `resolveWsUrl()` для корректного построения WS URL
   - Поддержка относительных путей (`/ws` → `ws://localhost:3000/ws` в браузере)
   - Автоматическое определение ws/wss по протоколу страницы

5. **`apps/ui/src/app/hooks/useTradingData.ts`**
   - Строгая логика REAL_MODE: REST success + WS connected + heartbeat <= 2s
   - Watchdog таймер (каждые 5s) для проверки REAL_MODE
   - Обновление pulse.mockMode на основе WS + heartbeat

6. **`apps/ui/src/app/components/trading/StatusBar.tsx`**
   - Приоритет heartbeat > event > message для last event age
   - Таймер обновления ageLabel каждую секунду
   - Корректное отображение статуса на основе pulse.lastHeartbeatTs

### Docker/Infrastructure
7. **`docker-compose.yml`**
   - Исправлены build args для ui: `VITE_API_BASE=/api`, `VITE_WS_URL=/ws`

8. **`ui.env`**
   - Изменено: `VITE_API_BASE=/api`, `VITE_WS_URL=/ws` (относительные пути)

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

## ✅ ЧЕК-ЛИСТ ПРОВЕРКИ "ЖИВОСТИ"

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
```

### 2. WebSocket (через nginx proxy)
```bash
# В браузере DevTools → Network → WS
# Должен быть: ws://localhost:3000/ws
# Status: 101 Switching Protocols
# Heartbeat приходит каждую секунду
```

### 3. UI StatusBar
Откройте http://localhost:3000 и проверьте:
- ✅ **WS**: статус `connected` (зелёный)
- ✅ **Engine**: статус `RUNNING` или `STOPPED` (не `UNKNOWN`)
- ✅ **Last event**: `< 2s ago` (зелёный), обновляется каждую секунду
- ✅ **MOCK MODE**: НЕ показывается (если backend жив)
- ✅ **Backend unreachable**: НЕ показывается

### 4. Real-time данные
- ✅ EventFeed показывает heartbeat события
- ✅ StatusBar обновляется в реальном времени
- ✅ Chart (если есть market data) обновляется через WS topic `market_data`

---

## 🔍 TROUBLESHOOTING

### Проблема: WS reconnecting постоянно
**Причина:** UI пытается подключиться к неправильному URL  
**Решение:**
1. Проверьте `ui.env`: должно быть `VITE_WS_URL=/ws` (относительный путь)
2. Пересоберите UI: `docker compose build --no-cache ui`
3. Проверьте nginx.conf в контейнере: `docker exec hean-ui cat /etc/nginx/conf.d/default.conf`

### Проблема: 502 Bad Gateway на `/api/*`
**Причина:** nginx не может достучаться до `api:8000`  
**Решение:**
1. Проверьте, что api контейнер запущен: `docker compose ps`
2. Проверьте сеть: `docker network inspect hean_hean-network`
3. Проверьте логи nginx: `docker compose logs ui | grep error`

### Проблема: CORS ошибки
**Причина:** Backend не разрешает origin  
**Решение:**
- Backend уже настроен с `allow_origins=["*"]` в `main.py`
- Если всё ещё есть CORS, проверьте, что запросы идут через `/api/*` (nginx proxy), а не напрямую на `api:8000`

### Проблема: MOCK MODE всегда включен
**Причина:** REAL_MODE требует: REST success + WS connected + heartbeat <= 2s  
**Решение:**
1. Проверьте REST: `curl http://localhost:3000/api/telemetry/ping`
2. Проверьте WS в DevTools → Network → WS (должен быть connected)
3. Проверьте heartbeat: в DevTools → Console должны быть heartbeat события каждую секунду
4. Если heartbeat не приходит, проверьте backend логи: `docker compose logs api | grep heartbeat`

### Проблема: Wrong WS URL (ws://api:8000/ws в браузере)
**Причина:** VITE_WS_URL захардкожен в сборке  
**Решение:**
1. Убедитесь, что `ui.env` содержит `VITE_WS_URL=/ws`
2. Пересоберите UI: `docker compose build --no-cache ui`
3. Проверьте, что в браузере используется `ws://localhost:3000/ws` (не `ws://api:8000/ws`)

### Проблема: StatusBar показывает "UNKNOWN" или "OFFLINE"
**Причина:** StatusBar не получает heartbeat через WS  
**Решение:**
1. Проверьте, что WS подключен (DevTools → Network → WS)
2. Проверьте, что подписка на `system_heartbeat` работает (в client.ts автоматически)
3. Проверьте backend: `docker compose logs api | grep heartbeat`
4. Если heartbeat не приходит, проверьте, что engine запущен

---

## 📊 АРХИТЕКТУРА РЕШЕНИЯ

### Схема проксирования (nginx в UI контейнере)
```
Browser → http://localhost:3000/api/* → nginx → http://api:8000/*
Browser → ws://localhost:3000/ws → nginx → ws://api:8000/ws
```

### REAL_MODE логика
```
REAL_MODE = REST_OK && WS_CONNECTED && HEARTBEAT_AGE <= 2s
MOCK_MODE = !REAL_MODE (после 5s без heartbeat)
```

### StatusBar приоритет для last event age
```
1. pulse.lastHeartbeatTs (если есть)
2. pulse.lastEventTs
3. telemetry.last_event_ts
4. ws.lastMessageAt
```

---

## 🎯 РЕЗУЛЬТАТЫ

### До исправлений
- ❌ UI ходил на `http://api:8000` из браузера (DNS ошибка)
- ❌ WS не подключался (`ws://api:8000/ws` недоступен из браузера)
- ❌ StatusBar показывал "UNKNOWN", "OFFLINE"
- ❌ MOCK MODE всегда включен

### После исправлений
- ✅ UI ходит через `/api/*` (nginx proxy)
- ✅ WS подключается через `/ws` (nginx proxy с upgrade)
- ✅ StatusBar показывает реальный статус с live обновлением
- ✅ REAL_MODE работает корректно (только при живом backend + WS + heartbeat)
- ✅ Market data доступен через REST и WS

---

## 📝 ДОПОЛНИТЕЛЬНЫЕ ЗАМЕЧАНИЯ

1. **Market Data**: Если у backend нет реальных market данных, UI покажет "Market feed wiring pending", но heartbeat/telemetry обязаны работать.

2. **Chart Markers**: Маркеры на графике добавляются автоматически при событиях orders/positions через WS topic `orders` и `positions`.

3. **Healthcheck**: UI контейнер может показывать "unhealthy" в docker compose ps, но это не критично, если nginx работает (проверьте `curl http://localhost:3000`).

4. **Dev Mode**: Для разработки используйте `ui-dev` профиль, который запускает Vite dev server с hot-reload.

---

**Конец отчёта**
