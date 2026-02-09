# 🚀 БЫСТРЫЙ СТАРТ - BYBIT TESTNET ONLY

**Обновлено:** 30 января 2026
**Версия:** 2.0.0 (No Paper Trading)

---

## ⚡ Что Изменилось?

### ❌ Удалено:
- Paper trading (симуляция)
- Dry run mode
- Synthetic price feed
- Fallback на симуляцию

### ✅ Добавлено:
- **ТОЛЬКО Bybit Testnet** - реальное API
- Обязательное подключение к Bybit
- Реальные ордера с виртуальными деньгами
- Production-ready архитектура

---

## 🎯 Запуск за 3 Минуты

### Шаг 1: Проверьте API Ключи

```bash
# Откройте .env файл
cat .env | grep BYBIT

# Должно быть:
BYBIT_API_KEY=your-testnet-key
BYBIT_API_SECRET=your-testnet-secret
BYBIT_TESTNET=true
```

**Нет ключей?** Получите здесь: https://testnet.bybit.com/app/user/api-management

### Шаг 2: Проверьте Настройки

```bash
# В .env должно быть:
TRADING_MODE=live
LIVE_CONFIRM=YES
BYBIT_TESTNET=true
```

### Шаг 3: Запустите Систему

```bash
# Docker (рекомендуется)
./docker-deploy.sh

# Или напрямую
python -m hean.main run
```

### Шаг 4: Проверьте Логи

Должно быть:
```
🚀 Starting BYBIT TESTNET ONLY router (no paper trading)
✅ Bybit testnet clients connected
✅ Execution router started (Bybit testnet only)
```

---

## ✅ Проверка Работоспособности

### 1. Dashboard

```bash
# Откройте в браузере
open http://localhost:3000
```

### 2. API Health

```bash
# Проверьте API
curl http://localhost:8000/health

# Должно быть:
{"status": "healthy"}
```

### 3. Позиции и Ордера

```bash
# Проверьте позиции
curl http://localhost:8000/positions

# Проверьте ордера
curl http://localhost:8000/orders
```

### 4. Bybit Dashboard

Откройте: https://testnet.bybit.com/trade/spot/BTCUSDT

Вы должны видеть свои ордера там!

---

## ⚠️ Важно!

### ❌ НЕ РАБОТАЕТ:

```bash
# Старые команды:
DRY_RUN=true python -m hean.main run  ❌
TRADING_MODE=paper python -m hean.main run  ❌
```

### ✅ РАБОТАЕТ:

```bash
# Новые команды:
BYBIT_TESTNET=true python -m hean.main run  ✅
./docker-deploy.sh  ✅
```

---

## 🔧 Настройка Стратегий

### .env Параметры

```bash
# Капитал (виртуальный на testnet)
INITIAL_CAPITAL=300.0

# Риск менеджмент
MAX_DAILY_DRAWDOWN_PCT=10.0
MAX_TRADE_RISK_PCT=0.15
MAX_OPEN_POSITIONS=6

# Символы для торговли
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT
```

---

## 🐛 Решение Проблем

### Проблема: "Cannot start without Bybit connection"

**Решение:**
1. Проверьте API ключи в `.env`
2. Проверьте интернет-соединение
3. Убедитесь что ключи testnet (не mainnet)

### Проблема: "Trading mode must be 'live'"

**Решение:**
```bash
# В .env:
TRADING_MODE=live  # НЕ "paper"!
```

### Проблема: Ордера не исполняются

**Решение:**
1. Проверьте баланс testnet
2. Проверьте логи на ошибки
3. Проверьте размер ордера (минимум ~5 USD)

---

## 📊 Мониторинг

### Логи

```bash
# Docker logs
docker compose logs -f hean-api

# Поиск ошибок
docker compose logs | grep ERROR
```

### Метрики

```bash
# Equity и PnL
curl http://localhost:8000/portfolio/summary

# Статистика стратегий
curl http://localhost:8000/telemetry/summary
```

---

## 🎓 Следующие Шаги

1. ✅ Запустите систему на testnet
2. ✅ Проверьте Dashboard
3. ✅ Наблюдайте за ордерами на Bybit
4. ✅ Оптимизируйте параметры
5. ✅ Накопите статистику
6. ⚠️ Переключитесь на mainnet (ОСТОРОЖНО!)

---

## 📚 Документация

- **Полная миграция:** [MIGRATION_TO_BYBIT_TESTNET_ONLY.md](MIGRATION_TO_BYBIT_TESTNET_ONLY.md)
- **Главная документация:** [README.md](README.md)
- **API документация:** http://localhost:8000/docs

---

## 💡 Полезные Ссылки

- **Bybit Testnet:** https://testnet.bybit.com/
- **API Keys:** https://testnet.bybit.com/app/user/api-management
- **Пополнить testnet:** https://testnet.bybit.com/app/user/asset/coin-deposit (бесплатные виртуальные монеты)

---

## ✅ Чеклист Готовности

- [ ] API ключи testnet получены
- [ ] `.env` файл обновлен
- [ ] Docker запущен
- [ ] Dashboard открывается (http://localhost:3000)
- [ ] API health проверен
- [ ] Видны ордера на Bybit testnet

**Все ✅?** Поздравляю! Система готова к работе! 🎉

---

**Вопросы?** Смотрите [MIGRATION_TO_BYBIT_TESTNET_ONLY.md](MIGRATION_TO_BYBIT_TESTNET_ONLY.md)
