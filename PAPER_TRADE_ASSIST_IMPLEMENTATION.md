# Paper Trade Assist Implementation

## Обзор

Реализован режим `PAPER_TRADE_ASSIST` для гарантированного открытия сделок в paper/dry_run режиме. В live режиме безопасность не изменена.

## Изменённые файлы

### 1. Конфигурация
- **src/hean/config.py**
  - Добавлен флаг `paper_trade_assist` с валидацией
  - Добавлены параметры для micro-trade: interval, notional, TP/SL, max_time
  - Валидация: запрещено в live (DRY_RUN=false && LIVE_CONFIRM=YES)

### 2. Paper Trade Assist модуль
- **src/hean/paper_trade_assist.py** (новый)
  - Функции для ослабления фильтров (multipliers, overrides)
  - Функции логирования блокировок (log_block_reason, log_allow_reason)
  - Проверка безопасности режима

### 3. Risk Limits
- **src/hean/risk/limits.py**
  - `check_order_request`: поддержка override max_open_positions
  - `check_daily_attempts`: multiplier для увеличения лимитов
  - `check_cooldown`: multiplier для сокращения cooldown
  - Добавлено логирование блокировок

### 4. Edge Estimator
- **src/hean/execution/edge_estimator.py**
  - `get_min_edge_threshold`: снижение порога на 40% в paper assist
  - Добавлено логирование блокировок

### 5. Impulse Filters
- **src/hean/strategies/impulse_filters.py**
  - `SpreadFilter`: multiplier 2.5x для spread threshold
  - `VolatilityExpansionFilter`: ослабление в paper assist
  - Добавлено логирование

### 6. Main Trading System
- **src/hean/main.py**
  - Добавлен fallback micro-trade loop (`_micro_trade_fallback_loop`)
  - Обработка time-based exit для micro-trade в `_handle_tick_forced_exit`
  - Диагностика блокировок во всех местах проверок
  - Команда `report` для показа статистики блокировок
  - Показ статуса PAPER_TRADE_ASSIST при старте

### 7. Тесты
- **tests/test_paper_trade_assist.py** (новый)
  - Тест запрета в live режиме
  - Тест разрешения в dry_run/testnet
  - Тест multipliers и overrides

## Команды для запуска

### 1. Включить Paper Trade Assist

```bash
# В .env или через переменные окружения
export DRY_RUN=true
export PAPER_TRADE_ASSIST=true

# Или в .env файле:
DRY_RUN=true
PAPER_TRADE_ASSIST=true
```

### 2. Запустить систему

```bash
# CLI режим
python -m hean.main run

# Или через make
make run
```

### 3. Проверить статус

При старте система покажет:
```
Trading mode: paper
DRY_RUN: True
bybit_testnet: False
PAPER_TRADE_ASSIST: True
LIVE_CONFIRM: 
```

### 4. Посмотреть диагностику

```bash
# Показать отчет о блокировках
python -m hean.main report
```

Вывод покажет:
- Количество попыток входа
- Количество блокировок
- Топ причин блокировок
- Статистику по стратегиям и символам
- Статус micro-trade

## Пример ожидаемого лога

### Открытие micro-trade:

```
[PAPER_ASSIST] Opening fallback micro-trade: BTCUSDT size=0.000200 price=$50000.00 TP=$50015.00 SL=$49985.00
[ALLOW] ALLOW symbol=BTCUSDT strategy=paper_assist_micro note=OrderRequest created: size=0.000200
[ORDER_FILLED_HANDLER] Position pos_abc123 created and registered: long 0.000200 BTCUSDT @ 50000.00
```

### Закрытие micro-trade (TP):

```
[PAPER_ASSIST] Micro-trade pos_abc123 hit TP $50015.00, closing
[ORDER_FILLED_HANDLER] Position pos_abc123 closed: PnL=$3.00
```

### Закрытие micro-trade (time-based):

```
[PAPER_ASSIST] Micro-trade pos_abc123 time limit reached (5.0min >= 5min), closing at market
[ORDER_FILLED_HANDLER] Position pos_abc123 closed: PnL=$2.50
```

### Блокировка с диагностикой:

```
[BLOCK] edge_reject symbol=BTCUSDT strategy=impulse_engine measured=3.5000 threshold=5.0000
[ALLOW] BLOCK symbol=BTCUSDT strategy=impulse_engine note=edge_reject: edge=3.5 bps < threshold=5.0 bps
```

## Что изменилось в поведении

### При PAPER_TRADE_ASSIST=true:

1. **Spread gate**: порог увеличен в 2.5 раза
2. **Volatility gate**: ослаблен (min * 0.5, max * 1.5)
3. **Edge threshold**: снижен на 40%
4. **Max positions**: минимум 2 (даже если в конфиге меньше)
5. **Daily attempts**: увеличены в 2 раза
6. **Cooldown**: сокращён в 3 раза
7. **Regime**: разрешены все режимы (включая neutral/chop)
8. **Fallback micro-trade**: каждые 60 секунд (по умолчанию) открывается micro-trade, если нет других сделок

### В LIVE режиме:

- Все проверки остаются строгими
- PAPER_TRADE_ASSIST не может быть включен (валидация падает с ошибкой)
- Fallback micro-trade не работает

## Безопасность

1. **Валидация при старте**: система падает с понятной ошибкой, если пытаются включить в live
2. **Двойная проверка**: `is_paper_assist_enabled()` проверяет безопасность перед использованием
3. **Killswitch**: micro-trade не открывается, если killswitch активен
4. **Risk limits**: не отключаются полностью, только ослабляются

## Тестирование

```bash
# Запустить тесты
pytest tests/test_paper_trade_assist.py -v

# Проверить, что в live режиме нельзя включить
export DRY_RUN=false
export LIVE_CONFIRM=YES
export PAPER_TRADE_ASSIST=true
python -m hean.main run  # Должно упасть с ошибкой
```

## Диагностика проблем

Если сделки всё ещё не открываются:

1. Проверить логи на `[BLOCK]` и `[ALLOW]`
2. Запустить `python -m hean.main report` для статистики
3. Проверить, что `PAPER_TRADE_ASSIST=true` в логах при старте
4. Проверить, что micro-trade loop запущен: `Paper Trade Assist: Fallback micro-trade loop started`
5. Проверить killswitch: micro-trade не работает, если killswitch активен

## Примеры использования

### Минимальный пример для гарантированных сделок:

```bash
# .env
DRY_RUN=true
PAPER_TRADE_ASSIST=true
INITIAL_CAPITAL=400.0
TRADING_SYMBOLS=BTCUSDT,ETHUSDT

# Запуск
python -m hean.main run
```

Через 60 секунд (или меньше, если есть обычные сигналы) должны появиться сделки.

### С кастомными параметрами micro-trade:

```bash
# .env
DRY_RUN=true
PAPER_TRADE_ASSIST=true
PAPER_TRADE_ASSIST_MICRO_TRADE_INTERVAL_SEC=30  # Каждые 30 секунд
PAPER_TRADE_ASSIST_MICRO_TRADE_NOTIONAL_USD=5.0  # 5 USD
PAPER_TRADE_ASSIST_MICRO_TRADE_TP_PCT=0.2  # 0.2% TP
PAPER_TRADE_ASSIST_MICRO_TRADE_SL_PCT=0.2  # 0.2% SL
PAPER_TRADE_ASSIST_MICRO_TRADE_MAX_TIME_MIN=3  # 3 минуты максимум
```

## Важные замечания

1. **Micro-trade открывается только если:**
   - Нет открытых позиций (или меньше 2)
   - Killswitch не активен
   - Прошло достаточно времени с последнего micro-trade для символа

2. **Micro-trade закрывается:**
   - При достижении TP
   - При достижении SL
   - По истечении времени (max_time_min)

3. **Все существующие команды работают:**
   - `python -m hean.main run` - работает
   - `python -m hean.main backtest` - работает
   - `python -m hean.main evaluate` - работает
   - `python -m hean.main process ...` - работает

4. **В live режиме ничего не изменилось:**
   - Все проверки остаются строгими
   - PAPER_TRADE_ASSIST не может быть включен

