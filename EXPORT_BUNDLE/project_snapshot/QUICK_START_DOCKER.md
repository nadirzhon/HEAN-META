# Быстрый старт с Docker Compose

## Запуск за 3 шага

### 1. Настройте .env файл

```bash
# Минимальная конфигурация для paper trading
TRADING_MODE=paper
INITIAL_CAPITAL=10000.0
```

### 2. Запустите Docker Compose

```bash
# Сборка и запуск
docker-compose up -d

# Или через Makefile
make docker-run
```

### 3. Проверьте логи

```bash
# Просмотр логов
docker-compose logs -f

# Или через Makefile
make docker-logs
```

## Полезные команды

```bash
# Остановка
docker-compose down
# или
make docker-down

# Перезапуск
docker-compose restart
# или
make docker-restart

# Проверка статуса
docker-compose ps

# Health check
curl http://localhost:8080/health
```

## Для live trading

Добавьте в `.env`:

```bash
TRADING_MODE=live
LIVE_CONFIRM=YES
BYBIT_API_KEY=your-key
BYBIT_API_SECRET=your-secret
BYBIT_TESTNET=true
```

Затем перезапустите:

```bash
docker-compose restart
```

## Готово! 🐳

Система запущена и работает в Docker контейнере.

Подробнее: [DOCKER_GUIDE.md](DOCKER_GUIDE.md)

