# HEAN SYMBIONT X 🧬

**Самоэволюционирующий живой организм для автономной торговли на крипто-рынках**

SYMBIONT X — это не просто торговый бот. Это **живая система**, которая самостоятельно эволюционирует, адаптируется и защищается от угроз, используя биологические принципы.

---

## 🌟 Что это такое?

HEAN SYMBIONT X — это полностью автономная торговая система, построенная как живой организм со следующими органами:

### 🧠 **1. Market Nervous System** — Нервная система рынка
- **WebSocket потоки**: real-time данные с Bybit
- **Event Envelope**: унифицированный формат всех событий
- **Health Sensors**: мониторинг качества данных (lag, gaps, spread, message rate)

### 🔮 **2. Market Regime Brain** — Мозг классификации режимов
- **Feature Extraction**: извлечение 20+ фич из рынка (trend, volatility, liquidity, microstructure)
- **Regime Classifier**: определяет режим рынка (TREND_UP, TREND_DOWN, RANGE, HIGH_VOL, NEWS_SHOCK и т.д.)
- **Confidence Scoring**: уверенность в классификации

### 🧬 **3. Alpha Genome Lab** — Лаборатория эволюции стратегий
- **Strategy Genome**: представление стратегии как набора генов
- **Mutation Engine**: мутации стратегий (point mutations, gene swaps, duplications, deletions)
- **Crossover Engine**: скрещивание двух родительских стратегий
- **Evolution Engine**: естественный отбор (survival of the fittest)

### 🥊 **4. Adversarial Digital Twin** — Злой экзаменатор
- **Replay World**: backtesting на исторических данных
- **Paper World**: paper trading в реальном времени
- **Micro-Real World**: реальная торговля с микро-суммами ($10-50)
- **Stress Tests**: 10 типов стресс-тестов (flash crash, high vol, thin liquidity и т.д.)
- **Survival Score**: финальная оценка готовности к production (0-100)

### 💰 **5. Capital Allocator** — Дарвиновский распределитель капитала
- **Survival-Weighted Allocation**: больше капитала сильным стратегиям
- **Portfolio Management**: управление портфелем с диверсификацией
- **Auto-Rebalancing**: автоматическая ребалансировка на основе performance
- **Kill Underperformers**: убивает слабых, награждает сильных

### 🛡️ **6. Immune System** — Иммунная система защиты
- **Risk Constitution**: неизменяемые правила безопасности
- **Reflex System**: мгновенные автоматические реакции (safe mode, reduce exposure, close positions)
- **Circuit Breakers**: аварийные выключатели при критических сбоях
- **Kill Switch**: глобальная аварийная остановка

### 📝 **7. Decision Ledger** — Полная память решений
- **Append-Only Log**: все решения записываются навсегда
- **Decision Analysis**: анализ успешных и провальных паттернов
- **Decision Replay**: воспроизведение решений для debugging
- **Decision Chain**: отслеживание связанных решений

### ⚡ **8. Execution Microkernel** — Ультра-быстрое исполнение
- Python wrapper (будет заменён на Rust для максимальной скорости)
- Latency tracking (submission + execution)
- Order validation перед исполнением

---

## 📊 6 KPI — Всегда Видимые Vital Signs

SYMBIONT X постоянно отслеживает 6 ключевых показателей здоровья:

1. **🟢 Survival Score** (0-100)
   - Общая живучесть системы
   - Основан на Sharpe ratio, drawdown, win rate

2. **⚡ Execution Edge** (0-100)
   - Качество исполнения ордеров
   - Slippage + latency

3. **🛡️ Immunity Saves** (0-100)
   - Сколько раз защита спасла от потерь
   - Reflexes + Circuit Breakers

4. **💎 Alpha Production** (0-100)
   - Генерация прибыли
   - ROI-based scoring

5. **🎯 Truth Mode** (0-100)
   - Качество данных и классификации
   - Data health + Regime confidence

6. **🤖 Autonomy Level** (0-100)
   - Уровень автономности
   - Active strategies + Decisions/hour - Manual interventions

---

## 🚀 Быстрый старт

### Установка

```bash
# Клонировать репозиторий
cd HEAN

# Установить зависимости
pip install -r requirements.txt
```

### Минимальный пример

```python
import asyncio
from hean.symbiont_x import HEANSymbiontX

async def main():
    # Конфигурация
    config = {
        'symbols': ['BTCUSDT'],
        'bybit_api_key': 'YOUR_API_KEY',
        'bybit_api_secret': 'YOUR_API_SECRET',
        'initial_capital': 10000,
        'population_size': 50,
    }

    # Создать SYMBIONT X
    symbiont = HEANSymbiontX(config=config)

    # Запустить
    await symbiont.start()

    # Эволюция
    for generation in range(100):
        await symbiont.evolve_generation()
        print(symbiont.get_vital_signs())
        await asyncio.sleep(3600)  # 1 hour

    await symbiont.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧬 Как работает эволюция?

1. **Initialization**: Создаётся случайная популяция из 50 стратегий
2. **Testing**: Каждая стратегия тестируется в 3 мирах (Replay → Paper → Micro-Real)
3. **Stress Testing**: Проходит 10 стресс-тестов
4. **Survival Score**: Вычисляется финальный survival score (0-100)
5. **Selection**: Выживают только стратегии с score > 60
6. **Reproduction**:
   - Top 5 элитных стратегий копируются без изменений
   - 30% новых потомков через crossover (скрещивание)
   - 70% новых потомков через mutations (мутации)
7. **Capital Allocation**: Капитал распределяется Darwinian способом (сильные получают больше)
8. **Repeat**: Цикл повторяется каждый час/день

---

## 🛡️ Защита

SYMBIONT X имеет **три уровня защиты**:

### 1. Constitution (Конституция)
Неизменяемые правила, которые **НЕ МОГУТ** быть нарушены ни при каких условиях:
- Max position size: $10K / 20% capital
- Max leverage: 5x
- Max daily loss: 5%
- Max drawdown: 25%

### 2. Reflexes (Рефлексы)
Мгновенные автоматические реакции:
- **Flash Crash** → Safe Mode
- **Extreme Volatility** → Reduce Exposure
- **Drawdown Limit** → Close Positions
- **Thin Liquidity** → Passive Only
- **Wide Spreads** → Reject Orders

### 3. Circuit Breakers (Автоматы)
Аварийные выключатели:
- API failures (5 за 60 сек) → OPEN
- Order rejections (10 за 60 сек) → OPEN
- Execution errors (3 за 30 сек) → OPEN
- Risk violations (3 за 5 мин) → OPEN

Когда breaker OPEN → торговля останавливается.
После recovery timeout → переход в HALF_OPEN (testing).
После успешных операций → CLOSED (normal).

---

## 📁 Структура проекта

```
src/hean/symbiont_x/
├── __init__.py
├── symbiont.py                  # Главный класс
├── kpi_system.py                # 6 KPI система
│
├── nervous_system/              # Нервная система
│   ├── event_envelope.py
│   ├── ws_connectors.py
│   └── health_sensors.py
│
├── regime_brain/                # Мозг режимов
│   ├── regime_types.py
│   ├── features.py
│   └── classifier.py
│
├── genome_lab/                  # Лаборатория генома
│   ├── genome_types.py
│   ├── mutation_engine.py
│   ├── crossover.py
│   └── evolution_engine.py
│
├── adversarial_twin/            # Тестовые миры
│   ├── test_worlds.py
│   ├── stress_tests.py
│   └── survival_score.py
│
├── capital_allocator/           # Распределитель капитала
│   ├── portfolio.py
│   ├── allocator.py
│   └── rebalancer.py
│
├── immune_system/               # Иммунная система
│   ├── constitution.py
│   ├── reflexes.py
│   └── circuit_breakers.py
│
├── decision_ledger/             # Память решений
│   ├── decision_types.py
│   ├── ledger.py
│   ├── replay.py
│   └── analysis.py
│
└── execution_kernel/            # Ядро исполнения
    └── executor.py
```

---

## ⚙️ Конфигурация

Все параметры настраиваются через config dictionary:

```python
config = {
    # Market data
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    'bybit_api_key': 'YOUR_KEY',
    'bybit_api_secret': 'YOUR_SECRET',

    # Capital
    'initial_capital': 10000,

    # Evolution
    'population_size': 50,
    'elite_size': 5,
    'mutation_rate': 0.1,
    'crossover_rate': 0.3,

    # Allocation
    'allocation_method': 'survival_weighted',  # or 'equal_weight', 'risk_parity'
    'rebalance_interval_hours': 24,

    # Risk constitution
    'risk_constitution': {
        'max_position_size_usd': 10000,
        'max_position_size_pct': 20.0,
        'max_leverage': 5.0,
        'max_daily_loss_pct': 5.0,
        'max_drawdown_pct': 25.0,
        'min_orderbook_depth_usd': 50000,
        'max_spread_bps': 50,
    }
}
```

---

## 🎯 Roadmap

### Phase 1: Core Foundation ✅ (COMPLETED)
- [x] Market Nervous System
- [x] Regime Brain
- [x] Alpha Genome Lab
- [x] Adversarial Twin
- [x] Capital Allocator
- [x] Immune System
- [x] Decision Ledger
- [x] Execution Kernel
- [x] KPI System
- [x] Main Symbiont orchestrator

### Phase 2: Integration & Testing (В ПРОЦЕССЕ)
- [ ] Интеграция с реальным Bybit API
- [ ] Backtesting на исторических данных
- [ ] Paper trading интеграция
- [ ] Unit tests для всех компонентов

### Phase 3: Optimization
- [ ] Rust Execution Microkernel (для ultra-low latency)
- [ ] ML модели для Regime Brain
- [ ] Advanced strategy genomes
- [ ] Multi-exchange support

### Phase 4: UI & Monitoring
- [ ] Web dashboard (React)
- [ ] Real-time KPI visualization
- [ ] Strategy evolution graphs
- [ ] Decision timeline viewer
- [ ] Telegram bot notifications
- [ ] Mobile app (basic)

---

## 🧪 Тестирование

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Backtest на исторических данных
python examples/backtest_symbiont.py
```

---

## 📖 Документация

Полная документация доступна в:
- `HEAN_SYMBIONT_X_ARCHITECTURE.md` — детальная архитектура
- `docs/` — подробные гайды по компонентам

---

## ⚠️ Disclaimer

**ВАЖНО**: SYMBIONT X — это экспериментальная система. Торговля криптовалютами связана с высоким риском.

- Используйте только те средства, которые можете позволить себе потерять
- Начинайте с paper trading
- Тестируйте на micro-real с минимальными суммами
- Всегда мониторьте KPI и логи
- Constitution и Circuit Breakers — ваша последняя защита

---

## 🤝 Contributing

Приветствуются pull requests! Пожалуйста:
1. Форкните репозиторий
2. Создайте feature branch
3. Напишите тесты
4. Создайте pull request

---

## 📝 License

MIT License - смотрите LICENSE file

---

## 💬 Контакты

Вопросы? Идеи? Найденные баги?
- Открывайте Issues на GitHub
- Или пишите: [ваш email]

---

## 🌟 Философия

SYMBIONT X основан на простой идее: **рынки эволюционируют, и стратегии должны эволюционировать вместе с ними**.

Вместо статичных правил, SYMBIONT X **адаптируется**:
- Слабые стратегии умирают
- Сильные мутируют и размножаются
- Капитал течёт к победителям
- Защита активируется автоматически
- Память сохраняет каждое решение

Это не бот. Это **организм**. 🧬

---

**Да начнётся эволюция! 🚀**
