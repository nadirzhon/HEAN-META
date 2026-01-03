# Docker Guide для HEAN Trading System

## Быстрый старт

### 1. Подготовка

Убедитесь, что у вас есть файл `.env` с необходимыми переменными:

```bash
# Минимальная конфигурация для paper trading
TRADING_MODE=paper
INITIAL_CAPITAL=10000.0

# Для live trading с Bybit
BYBIT_API_KEY=your-key
BYBIT_API_SECRET=your-secret
BYBIT_TESTNET=true
LIVE_CONFIRM=YES
TRADING_MODE=live

# LLM (опционально)
OPENAI_API_KEY=your-openai-key
# или
ANTHROPIC_API_KEY=your-anthropic-key
```

### 2. Запуск с Docker Compose

```bash
# Сборка и запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

### 3. Проверка работы

```bash
# Проверка статуса
docker-compose ps

# Проверка health check
curl http://localhost:8080/health

# Просмотр логов
docker-compose logs -f hean
```

## Команды

### Запуск в разных режимах

```bash
# Paper trading (по умолчанию)
docker-compose up -d

# Live trading (требует настройки .env)
docker-compose up -d

# Backtesting
docker-compose run --rm hean python -m hean.main backtest --days 30
```

### Управление контейнером

```bash
# Перезапуск
docker-compose restart

# Остановка
docker-compose stop

# Удаление контейнера
docker-compose down

# Пересборка образа
docker-compose build --no-cache
```

### Просмотр логов

```bash
# Все логи
docker-compose logs

# Последние 100 строк
docker-compose logs --tail=100

# Следить за логами
docker-compose logs -f

# Логи конкретного сервиса
docker-compose logs hean
```

## Разработка

### Hot Reload

При монтировании `./src:/app/src` изменения в коде применяются без пересборки образа.

```bash
# Перезапуск после изменений
docker-compose restart hean
```

### Отладка

```bash
# Запуск в интерактивном режиме
docker-compose run --rm hean bash

# Выполнение команд внутри контейнера
docker-compose exec hean python -m hean.main backtest --days 7
```

## Конфигурация

### Переменные окружения

Все переменные из `.env` автоматически загружаются в контейнер.

### Порты

- `8080` - Health check endpoint

### Volumes

- `./src:/app/src` - Исходный код (read-only)
- `./logs:/app/logs` - Логи
- `./.env:/app/.env` - Конфигурация

## Health Check

Система автоматически проверяет здоровье через HTTP endpoint:

```bash
curl http://localhost:8080/health
```

Health check проверяет:
- Работает ли система
- Есть ли ошибки
- Статус подключений

## Troubleshooting

### Проблема: Контейнер не запускается

```bash
# Проверьте логи
docker-compose logs hean

# Проверьте конфигурацию
docker-compose config

# Проверьте .env файл
cat .env
```

### Проблема: Ошибки подключения к Bybit

1. Проверьте API ключи в `.env`
2. Проверьте `BYBIT_TESTNET` настройку
3. Проверьте `LIVE_CONFIRM=YES`
4. Проверьте логи: `docker-compose logs hean | grep -i bybit`

### Проблема: Нет логов

```bash
# Проверьте, что логи пишутся
docker-compose exec hean ls -la /app/logs

# Проверьте права доступа
chmod -R 777 logs/
```

### Проблема: Health check не работает

```bash
# Проверьте, что порт открыт
docker-compose ps

# Проверьте вручную
docker-compose exec hean curl http://localhost:8080/health
```

## Production Deployment

### Рекомендации:

1. **Не монтируйте src в production**:
   ```yaml
   # Удалите или закомментируйте:
   # volumes:
   #   - ./src:/app/src:ro
   ```

2. **Используйте secrets для API ключей**:
   ```yaml
   secrets:
     - bybit_api_key
     - bybit_api_secret
   ```

3. **Настройте логирование**:
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "50m"
       max-file: "10"
   ```

4. **Используйте restart policy**:
   ```yaml
   restart: always
   ```

## Мониторинг

### Метрики

Health check endpoint предоставляет базовые метрики:
- Статус системы
- Количество стратегий
- Equity
- Drawdown

### Логи

Логи сохраняются в:
- Контейнер: `/app/logs`
- Хост: `./logs` (если volume настроен)

## Безопасность

### Рекомендации:

1. **Не коммитьте .env файл**
2. **Используйте secrets для production**
3. **Ограничьте права API ключей**
4. **Используйте IP whitelist на Bybit**
5. **Регулярно обновляйте зависимости**

## Примеры использования

### Paper Trading

```bash
# .env
TRADING_MODE=paper
INITIAL_CAPITAL=10000.0

# Запуск
docker-compose up -d
```

### Live Trading на Testnet

```bash
# .env
TRADING_MODE=live
LIVE_CONFIRM=YES
BYBIT_API_KEY=testnet-key
BYBIT_API_SECRET=testnet-secret
BYBIT_TESTNET=true

# Запуск
docker-compose up -d
```

### Backtesting

```bash
# Запуск backtest
docker-compose run --rm hean python -m hean.main backtest --days 30

# Результаты будут в логах
docker-compose logs hean | grep -i "backtest"
```

## Готово!

Теперь вы можете запускать HEAN Trading System с Docker Compose! 🐳

