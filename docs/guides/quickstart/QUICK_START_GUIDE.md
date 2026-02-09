# 🚀 HEAN Trading System - Быстрый Старт

## ⚡ Запуск за 3 минуты

### Шаг 1: Проверка требований

Убедитесь, что установлено:
- ✅ Docker Desktop (или Docker Engine + Docker Compose)
- ✅ Git (опционально)

Проверка:
```bash
docker --version        # Должно быть >= 20.10
docker compose version  # Должно быть >= 2.0
```

### Шаг 2: Запуск системы

Выполните ONE-LINE команду:

```bash
./docker-deploy.sh
```

Скрипт автоматически:
1. ✅ Проверит Docker и конфигурацию
2. ✅ Остановит старые контейнеры
3. ✅ Соберет все образы (5-10 минут при первом запуске)
4. ✅ Запустит все сервисы
5. ✅ Проверит их работоспособность
6. ✅ Покажет статус и доступные URL

### Шаг 3: Проверка работы

После успешного запуска откройте в браузере:

- **Trading UI:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/health

---

## 🎯 Что работает сразу после запуска

### ✅ API Backend (http://localhost:8000)

**Доступные endpoints:**
- `GET /health` - проверка здоровья
- `GET /docs` - интерактивная документация (Swagger UI)
- `GET /telemetry/summary` - метрики системы
- `GET /portfolio/summary` - состояние портфеля
- `GET /trading/why` - диагностика торговли
- `WS /ws` - WebSocket для real-time обновлений

**Тест из командной строки:**
```bash
# Health check
curl http://localhost:8000/health

# Telemetry
curl http://localhost:8000/telemetry/summary | jq

# Portfolio
curl http://localhost:8000/portfolio/summary | jq
```

### ✅ Trading Command Center (http://localhost:3000)

**Функции:**
- 📊 Dashboard - метрики в реальном времени
- 💼 Positions - управление позициями
- 📝 Orders - история ордеров
- 📈 Strategies - настройка стратегий
- 🎛️ Settings - конфигурация системы
- 📋 Logs - логи в реальном времени

### ✅ SYMBIONT X Trading Bot

**Режим:** Paper Trading на Bybit Testnet
**Символы:** BTC/USDT, ETH/USDT
**Капитал:** $10,000 (виртуальный)

**Проверка статуса:**
```bash
docker compose logs -f symbiont-testnet
```

---

## 📊 Управление Системой

### Просмотр логов

```bash
# Все сервисы
docker compose logs -f

# Только API
docker compose logs -f api

# Только UI
docker compose logs -f ui

# Только Trading Bot
docker compose logs -f symbiont-testnet
```

### Статус контейнеров

```bash
# Список запущенных контейнеров
docker compose ps

# Детальная информация
docker compose ps --format json | jq
```

### Остановка системы

```bash
# Остановка всех сервисов
docker compose down

# Остановка с удалением volumes
docker compose down -v
```

### Перезапуск

```bash
# Перезапуск всех сервисов
docker compose restart

# Перезапуск конкретного сервиса
docker compose restart api
```

---

## 🔧 Полезные Команды

### Быстрый перезапуск после изменений кода

```bash
# Пересборка и запуск
docker compose up -d --build

# Пересборка только конкретного сервиса
docker compose up -d --build api
```

### Очистка и полная пересборка

```bash
# Остановка всех контейнеров
docker compose down

# Удаление всех образов проекта
docker rmi hean-api hean-ui hean-symbiont

# Очистка Docker кэша (опционально)
docker system prune -f

# Полная пересборка
./docker-deploy.sh
```

### Проверка использования ресурсов

```bash
# Статистика в реальном времени
docker stats

# Только контейнеры HEAN
docker stats $(docker compose ps -q)
```

---

## 🐛 Устранение Проблем

### Проблема: Порты уже заняты

**Симптомы:** Ошибки при запуске "port already allocated"

**Решение:**
```bash
# Проверка занятых портов
lsof -i :8000 -i :3000 -i :6379

# Остановка старых контейнеров
docker compose down

# Или убить процессы на портах
sudo lsof -ti:8000 | xargs kill -9
```

### Проблема: Контейнеры не стартуют

**Симптомы:** Container exits immediately

**Решение:**
```bash
# Просмотр логов
docker compose logs api

# Проверка конфигурации
docker compose config

# Пересборка без кэша
docker compose build --no-cache
```

### Проблема: API возвращает 502/503

**Симптомы:** UI показывает "Cannot connect to API"

**Решение:**
```bash
# Проверка статуса API
curl http://localhost:8000/health

# Проверка логов API
docker compose logs api | tail -100

# Проверка Redis
docker compose exec redis redis-cli ping

# Перезапуск API
docker compose restart api
```

### Проблема: UI не загружается

**Симптомы:** Blank page или 404

**Решение:**
```bash
# Проверка логов UI
docker compose logs ui

# Проверка nginx конфигурации
docker compose exec ui cat /etc/nginx/conf.d/default.conf

# Пересборка UI
docker compose up -d --build ui
```

---

## ⚙️ Конфигурация

### Изменение API ключей Bybit

Отредактируйте файл `backend.env`:

```bash
# Testnet (безопасно для тестирования)
BYBIT_API_KEY=your_testnet_key
BYBIT_API_SECRET=your_testnet_secret
BYBIT_TESTNET=true

# Production (ОСТОРОЖНО!)
# BYBIT_API_KEY=your_production_key
# BYBIT_API_SECRET=your_production_secret
# BYBIT_TESTNET=false
# LIVE_CONFIRM=YES
```

После изменения:
```bash
docker compose restart api symbiont-testnet
```

### Изменение торговых символов

В `backend.env`:

```bash
# Добавьте или удалите символы
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT
```

### Включение/Отключение стратегий

В `backend.env`:

```bash
# Базовые стратегии
FUNDING_HARVESTER_ENABLED=true
BASIS_ARBITRAGE_ENABLED=true
IMPULSE_ENGINE_ENABLED=true

# Продвинутые стратегии
HF_SCALPING_ENABLED=false
ENHANCED_GRID_ENABLED=false
MOMENTUM_TRADER_ENABLED=false
```

---

## 📈 Мониторинг и Метрики

### Prometheus (если настроен)

```bash
# Запуск с мониторингом
docker compose --profile monitoring up -d

# Доступ к Prometheus
http://localhost:9091
```

### Grafana (если настроен)

```bash
# Доступ к Grafana
http://localhost:3001
# Login: admin / admin
```

### WebSocket Real-time Updates

```bash
# Test WebSocket connection (Python)
python3 << 'EOF'
import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        # Subscribe to system status
        await ws.send(json.dumps({
            "action": "subscribe",
            "topic": "system_status"
        }))
        # Receive messages
        for _ in range(5):
            msg = await ws.recv()
            print(json.dumps(json.loads(msg), indent=2))

asyncio.run(test_ws())
EOF
```

---

## 🔐 Безопасность

### ⚠️ ВАЖНО для Production

**НЕ ИСПОЛЬЗУЙТЕ** настройки по умолчанию в production!

**Обязательные изменения:**

1. **API Ключи:**
   - Используйте переменные окружения вместо файлов
   - Никогда не коммитьте ключи в Git

2. **HTTPS:**
   ```bash
   # Настройте reverse proxy (nginx/traefik)
   # С Let's Encrypt сертификатами
   ```

3. **Firewall:**
   ```bash
   # Закройте порты кроме 80/443
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw deny 8000/tcp
   sudo ufw deny 3000/tcp
   sudo ufw deny 6379/tcp
   ```

4. **Rate Limiting:**
   - Уже настроен в API (slowapi)
   - Настройте дополнительно на nginx/cloudflare

---

## 📚 Дополнительная Документация

- **Полный отчет аудита:** [PROJECT_AUDIT_REPORT.md](./PROJECT_AUDIT_REPORT.md)
- **Основная документация:** [README.md](./README.md)
- **API документация:** http://localhost:8000/docs
- **Docker инструкции:** [DOCKER_QUICKSTART.md](./DOCKER_QUICKSTART.md)

---

## 🆘 Получить Помощь

### Логи и Диагностика

```bash
# Сохранить все логи в файл
docker compose logs > system_logs.txt

# Сохранить конфигурацию
docker compose config > docker_config.yml

# System info
docker version > system_info.txt
docker compose version >> system_info.txt
```

### Проверка состояния системы

```bash
# Health checks всех сервисов
curl http://localhost:8000/health
curl http://localhost:3000

# Redis
docker compose exec redis redis-cli ping

# Containers status
docker compose ps
```

---

## ✅ Checklist Успешного Запуска

После выполнения `./docker-deploy.sh` проверьте:

- [ ] Все 4 контейнера в состоянии "Up" (healthy)
- [ ] API отвечает на http://localhost:8000/health
- [ ] UI загружается на http://localhost:3000
- [ ] Redis отвечает на ping
- [ ] WebSocket connection работает
- [ ] Логи не показывают ошибок
- [ ] SYMBIONT X подключен к Bybit Testnet

Если все пункты ✅ - **ПОЗДРАВЛЯЮ! Система работает!** 🎉

---

**Готовы торговать? Откройте http://localhost:3000 и начните!**

*Документация актуальна на: 30 января 2026*
