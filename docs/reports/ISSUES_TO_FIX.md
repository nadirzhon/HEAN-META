# 🐛 КРИТИЧЕСКИЕ ПРОБЛЕМЫ И ИХ ИСПРАВЛЕНИЕ

**Дата**: 27 января 2026
**Приоритет**: ВЫСОКИЙ

---

## 🔴 ПРОБЛЕМА 1: Debug обходы безопасности в production

### Где находится
**Файл**: `src/hean/strategies/impulse_engine.py`
**Строки**: 336, 371

### Что не так
Проверки безопасности отключены для отладки, но могут работать в продакшене!

### Код проблемы

**Строка 336** (обход кулдауна):
```python
# Check cooldown - TEMPORARILY DISABLED FOR DEBUG
# if not self._check_cooldown(row.symbol, row.timestamp):
#     return None
```

**Строка 371** (обход hard reject):
```python
# Hard reject DISABLED FOR DEBUG
# if self._hard_reject(signal, row):
#     return None
```

### Почему это опасно
- Торговля без кулдаунов → слишком частые сделки
- Нет фильтрации плохих сигналов → убыточные позиции
- Перегрузка API Bybit → возможный бан

### Как исправить

```bash
nano src/hean/strategies/impulse_engine.py
```

**Раскомментируйте проверки:**

```python
# Строка 336 - ИСПРАВЛЕННЫЙ КОД:
if not self._check_cooldown(row.symbol, row.timestamp):
    return None

# Строка 371 - ИСПРАВЛЕННЫЙ КОД:
if self._hard_reject(signal, row):
    return None
```

**Удалите комментарии "DISABLED FOR DEBUG"**

### Проверка
```bash
grep -n "DISABLED FOR DEBUG" src/hean/strategies/impulse_engine.py
# Не должно ничего найти после исправления
```

---

## 🟠 ПРОБЛЕМА 2: Торговля только одним символом

### Где находится
**Файл**: `backend.env`
**Строка**: 13

### Что не так
```bash
TRADING_SYMBOLS=BTCUSDT
```
Торгуется только Bitcoin, хотя система поддерживает 50+ символов!

### Почему это проблема
- Упускаете 98% торговых возможностей
- Все остальные монеты игнорируются
- Multi-symbol сканеры простаивают

### Как исправить

```bash
nano backend.env
```

**Найдите строку**:
```bash
TRADING_SYMBOLS=BTCUSDT
```

**Замените на** (минимум 5 символов для начала):
```bash
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT
```

**Добавьте включение multi-symbol**:
```bash
MULTI_SYMBOL_ENABLED=true
```

**Для агрессивной торговли** (10+ символов):
```bash
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,ADAUSDT,DOTUSDT,MATICUSDT,AVAXUSDT,LINKUSDT
```

### Проверка
```bash
grep TRADING_SYMBOLS backend.env
# Должно показать несколько символов через запятую
```

---

## 🟠 ПРОБЛЕМА 3: Profit Capture отключён

### Где находится
**Файл**: `src/hean/config.py`
**Строка**: 574

### Что не так
```python
profit_capture_enabled: bool = False
```
Нет автоматической фиксации прибыли → деньги теряются при разворотах!

### Почему это проблема
- Прибыль +30% может превратиться в -10% при развороте
- Нет защиты от резких падений
- Психологически сложно вручную фиксировать прибыль

### Как исправить

```bash
nano backend.env
```

**Добавьте в конец файла**:
```bash
# Profit Capture System
PROFIT_CAPTURE_ENABLED=true
PROFIT_CAPTURE_TARGET_PCT=20.0
PROFIT_CAPTURE_TRAIL_PCT=10.0
PROFIT_CAPTURE_MODE=partial
```

### Параметры
- `TARGET_PCT=20.0` - фиксировать при +20% прибыли
- `TRAIL_PCT=10.0` - трейлинг-стоп 10% от пика
- `MODE=partial` - закрывать 50% позиции (или `full` для 100%)

### Проверка
```bash
grep PROFIT_CAPTURE backend.env
# Должны быть все 4 параметра
```

---

## 🟠 ПРОБЛЕМА 4: Process Factory отключён

### Где находится
**Файл**: `src/hean/config.py`
**Строка**: 528

### Что не так
```python
process_factory_enabled: bool = False
```
6 автоматических процессов не работают → упускается пассивный доход!

### Что упускается
1. Capital Parking - деньги лежат без дела вместо Bybit Earn
2. Funding Monitor - не используются позитивные ставки финансирования
3. Fee Monitor - не оптимизируются maker/taker комиссии
4. Opportunity Scanner - не сканируются промо Bybit
5. Contract Monitor - пропускаются листинги новых монет
6. Campaign Monitor - игнорируются бонусы и акции

### Как исправить

```bash
nano backend.env
```

**Добавьте**:
```bash
# Process Factory (6 автоматических процессов)
PROCESS_FACTORY_ENABLED=true
PROCESS_FACTORY_ALLOW_ACTIONS=true
PROCESS_FACTORY_SCAN_INTERVAL_SEC=300
```

### Параметры
- `ENABLED=true` - включить фабрику
- `ALLOW_ACTIONS=true` - разрешить автоматические действия
- `SCAN_INTERVAL_SEC=300` - проверять каждые 5 минут

### Проверка
```bash
# После перезапуска проверьте логи
docker-compose logs api | grep "ProcessFactory"
# Должны быть сообщения о запуске процессов
```

---

## 🟡 ПРОБЛЕМА 5: C++ модули не собраны

### Где находится
**Директория**: `cpp_core/`

### Что не так
Исходники C++ есть, но не скомпилированы → всё работает в 10-100x медленнее!

### Что не работает без C++
- Fast Indicators (RSI, MACD, Bollinger) → медленный Python
- Oracle Engine (предсказание разворотов) → ОТКЛЮЧЕН
- Triangular Arbitrage → медленный поиск
- Graph Engine → нет оптимизации графов

### Как исправить

**Шаг 1: Установите зависимости**
```bash
# macOS
brew install cmake llvm

# Проверьте версию
clang++ --version
# Должно быть 14.0+
```

**Шаг 2: Соберите модули**
```bash
cd /Users/macbookpro/Desktop/HEAN/cpp_core

# Создайте build директорию
mkdir -p build
cd build

# Конфигурация
cmake ..

# Компиляция (используйте все CPU)
make -j$(sysctl -n hw.ncpu)
```

**Шаг 3: Установите библиотеки**
```bash
# Создайте папку для модулей
mkdir -p /Users/macbookpro/Desktop/HEAN/src/hean/cpp_modules

# Скопируйте .dylib файлы
cp *.dylib /Users/macbookpro/Desktop/HEAN/src/hean/cpp_modules/
```

### Проверка
```bash
ls -lh /Users/macbookpro/Desktop/HEAN/src/hean/cpp_modules/
# Должны быть:
# libfast_indicators.dylib
# libgraph_engine.dylib
# (возможно libmetamorphic.dylib)
```

**После сборки перезапустите систему**:
```bash
docker-compose restart api
```

### Ожидаемый результат в логах
```bash
docker-compose logs api | grep -i "c++\|fast_indicators\|graph_engine"
# Должно быть: "C++ indicators loaded successfully"
# Вместо: "Falling back to slower Python implementation"
```

---

## 🟡 ПРОБЛЕМА 6: Стратегии не зарегистрированы

### Где находится
**Файл**: `src/hean/main.py`
**Секция**: Регистрация стратегий (~строки 591-601)

### Что не так
3 прибыльные стратегии реализованы, но не активированы:
1. HF Scalping (40-60 сделок/день)
2. Enhanced Grid (пассивный доход во флэте)
3. Momentum Trader (ловит сильные движения)

### Как исправить

**Шаг 1: Добавьте импорты**

```bash
nano src/hean/main.py
```

Найдите секцию импортов стратегий (около строки 50) и добавьте:

```python
from hean.strategies.hf_scalping import HFScalpingStrategy
from hean.strategies.enhanced_grid import EnhancedGridStrategy
from hean.strategies.momentum_trader import MomentumTrader
```

**Шаг 2: Зарегистрируйте стратегии**

Найдите функцию регистрации стратегий (около строки 591-601):

```python
# После существующих регистраций добавьте:
register_strategy(HFScalpingStrategy)
register_strategy(EnhancedGridStrategy)
register_strategy(MomentumTrader)
```

**Шаг 3: Включите в конфигурации**

```bash
nano backend.env
```

Добавьте:
```bash
# Дополнительные стратегии
HF_SCALPING_ENABLED=true
ENHANCED_GRID_ENABLED=true
MOMENTUM_TRADER_ENABLED=true
```

### Проверка
```bash
# После перезапуска
curl http://localhost:8000/api/v1/strategies | jq .

# Должны быть HFScalping, EnhancedGrid, MomentumTrader в списке
```

---

## 🟡 ПРОБЛЕМА 7: Только Gemini API (нет OpenAI)

### Где находится
**Файл**: `backend.env`

### Что не так
```bash
GEMINI_API_KEY=установлен
OPENAI_API_KEY=не установлен
ANTHROPIC_API_KEY=не установлен
```

AI Factory работает только на Gemini → ограниченная генерация стратегий.

### Почему это проблема
- OpenAI лучше для кодогенерации
- Gemini как единственный провайдер → single point of failure
- AI Factory не может использовать fallback цепочку

### Как исправить

**Вариант A: Добавить OpenAI (платно)**

```bash
nano backend.env
```

Добавьте:
```bash
OPENAI_API_KEY=sk-ваш-ключ-здесь
AI_FACTORY_ENABLED=true
```

**Вариант B: Локальная LLM (бесплатно)**

```bash
# Установите Ollama
brew install ollama

# Скачайте модель
ollama pull mistral

# В backend.env:
AI_FACTORY_ENABLED=true
AI_FACTORY_PROVIDER=local
LOCAL_LLM_MODEL=mistral
```

### Проверка
```bash
grep -E "OPENAI_API_KEY|AI_FACTORY" backend.env
```

---

## 🟢 ПРОБЛЕМА 8: Testnet vs Live режим не ясен

### Где находится
**Файл**: `backend.env`

### Что не так
```bash
BYBIT_TESTNET=false
```
**Это означает РЕАЛЬНУЮ ТОРГОВЛЮ!**

### Рекомендация

**ДЛЯ ТЕСТИРОВАНИЯ** (виртуальные деньги):
```bash
nano backend.env

# Найдите и установите:
BYBIT_TESTNET=true
```

**ДЛЯ РЕАЛЬНОЙ ТОРГОВЛИ** (после тестов):
```bash
BYBIT_TESTNET=false
```

### Проверка режима
```bash
grep BYBIT_TESTNET backend.env

# Если true - тестнет (безопасно)
# Если false - LIVE (будьте осторожны!)
```

### Важно!
**НИКОГДА не переходите на live без**:
1. ✅ Тестов на testnet минимум 1 неделю
2. ✅ Положительных результатов на testnet
3. ✅ Удаления всех debug обходов
4. ✅ Настройки лимитов риска
5. ✅ Понимания всех стратегий

---

## 🟢 ПРОБЛЕМА 9: Income Streams недокапитализированы

### Где находится
**Файл**: `src/hean/config.py`
**Строки**: 576-612

### Текущие аллокации
```python
FundingHarvesterStream: 10%  # Мало!
MakerRebateStream: 5%        # Слишком мало!
BasisHedgeStream: 15%        # OK
VolatilityHarvestStream: 10% # Мало!
```

### Проблема
Слишком мало капитала → мало прибыли от пассивных стратегий

### Как исправить

```bash
nano backend.env
```

Добавьте:
```bash
# Income Streams Optimization
FUNDING_HARVESTER_CAPITAL_PCT=20.0
MAKER_REBATE_CAPITAL_PCT=10.0
BASIS_HEDGE_CAPITAL_PCT=25.0
VOLATILITY_HARVEST_CAPITAL_PCT=15.0

# Больше символов для streams
FUNDING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT
```

### Важно
Убедитесь что сумма аллокаций не превышает 100% доступного капитала!

---

## 🟢 ПРОБЛЕМА 10: Мониторинг не настроен

### Где находится
Нет `docker-compose.monitoring.yml` в корне проекта

### Что не так
Невозможно видеть:
- Графики прибыли в реальном времени
- Метрики производительности
- Алерты по проблемам

### Как исправить

**Временное решение** - используйте встроенный Prometheus:

```bash
# Метрики уже доступны на:
open http://localhost:8000/metrics
```

**Полное решение** - создайте monitoring stack:

```bash
# TODO: Нужно создать docker-compose.monitoring.yml
# Или использовать внешний Grafana
```

---

## 📋 Чек-лист исправлений

### Критичные (исправить немедленно)
- [ ] Удалены debug обходы в impulse_engine.py
- [ ] Проверен режим торговли (testnet/live)
- [ ] Включён Profit Capture

### Важные (исправить сегодня)
- [ ] Включена multi-symbol торговля
- [ ] Включён Process Factory
- [ ] Собраны C++ модули

### Желательные (на этой неделе)
- [ ] Зарегистрированы дополнительные стратегии
- [ ] Настроен AI Factory
- [ ] Оптимизированы Income Streams

---

## 🔧 Скрипт автоматического исправления

```bash
#!/bin/bash
# auto_fix.sh - Автоматическое исправление базовых проблем

echo "=== HEAN Auto-Fix Script ==="
echo ""

# Backup backend.env
cp backend.env backend.env.backup
echo "✅ Backup created: backend.env.backup"

# Fix multi-symbol
if ! grep -q "MULTI_SYMBOL_ENABLED" backend.env; then
    echo "" >> backend.env
    echo "# Multi-Symbol Trading" >> backend.env
    echo "MULTI_SYMBOL_ENABLED=true" >> backend.env
    echo "TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT" >> backend.env
    echo "✅ Multi-symbol enabled"
fi

# Fix profit capture
if ! grep -q "PROFIT_CAPTURE_ENABLED" backend.env; then
    echo "" >> backend.env
    echo "# Profit Capture" >> backend.env
    echo "PROFIT_CAPTURE_ENABLED=true" >> backend.env
    echo "PROFIT_CAPTURE_TARGET_PCT=20.0" >> backend.env
    echo "PROFIT_CAPTURE_TRAIL_PCT=10.0" >> backend.env
    echo "✅ Profit capture enabled"
fi

# Fix process factory
if ! grep -q "PROCESS_FACTORY_ENABLED" backend.env; then
    echo "" >> backend.env
    echo "# Process Factory" >> backend.env
    echo "PROCESS_FACTORY_ENABLED=true" >> backend.env
    echo "PROCESS_FACTORY_ALLOW_ACTIONS=true" >> backend.env
    echo "✅ Process factory enabled"
fi

# Check testnet mode
if grep -q "BYBIT_TESTNET=false" backend.env; then
    echo "⚠️  WARNING: Live trading mode active!"
    echo "   Consider setting BYBIT_TESTNET=true for testing"
fi

echo ""
echo "=== Fix Complete ==="
echo "Review backend.env and restart system:"
echo "  docker-compose restart"
```

**Использование**:
```bash
chmod +x auto_fix.sh
./auto_fix.sh
```

---

**Последнее обновление**: 27 января 2026
