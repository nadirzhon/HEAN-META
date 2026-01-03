й # Система генерации торговых агентов

Система для автоматической генерации торговых агентов с использованием LLM (Large Language Models).

## Установка

Система поддерживает два LLM провайдера:

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
```

### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

## Использование

### CLI утилита

#### Генерация начального агента
```bash
python generate_agent.py initial -o generated_agent.py
```

#### Генерация нескольких агентов
```bash
python generate_agent.py initial --count 10 -o generated_agents/
```

#### Эволюция агента на основе лучших/худших
```bash
python generate_agent.py evolution \
  --best-agents "Agent1: PF=2.5, WR=60%" \
  --worst-agents "Agent2: PF=0.8, WR=40%" \
  --market-conditions "High volatility, trending market" \
  --performance-metrics "Average PF: 1.5, Average WR: 50%" \
  -o evolved_agent.py
```

#### Мутация существующего агента
```bash
python generate_agent.py mutation \
  --agent-code "$(cat existing_agent.py)" \
  --profit-factor 1.2 \
  --total-pnl 1000.0 \
  --max-drawdown 15.0 \
  --win-rate 55.0 \
  --issues "Low win rate, high drawdown" \
  -o improved_agent.py
```

#### Генерация агента для рыночных условий
```bash
python generate_agent.py market_conditions \
  --volatility-level "high" \
  --volatility-value 0.05 \
  --trend-direction "bullish" \
  --trend-strength "strong" \
  --volume-level "high" \
  --market-regime "IMPULSE" \
  --spread-bps 5.0 \
  --historical-summary "Recent 30-day data shows strong uptrend" \
  --suggested-style "momentum" \
  --suggested-timeframe "short" \
  --suggested-size "medium" \
  --risk-approach "tight_stops" \
  -o specialized_agent.py
```

#### Гибридный агент
```bash
python generate_agent.py hybrid \
  --agent1-code "$(cat agent1.py)" \
  --pf1 2.5 --pnl1 5000.0 \
  --agent2-code "$(cat agent2.py)" \
  --pf2 1.8 --wr2 65.0 \
  --agent3-code "$(cat agent3.py)" \
  --pf3 2.0 --sharpe3 1.5 \
  -o hybrid_agent.py
```

#### Агент для решения проблемы
```bash
python generate_agent.py problem_focused \
  --problem "Low win rate in volatile markets" \
  --current-pf 1.1 \
  --problem-areas "Entry timing, stop loss placement" \
  --failed-patterns "Trading during high volatility spikes" \
  --focus1 "Better entry filters" \
  --focus2 "Adaptive stop losses" \
  --focus3 "Volatility-based position sizing" \
  -o problem_solver_agent.py
```

#### Оценка и улучшение агента
```bash
python generate_agent.py evaluation \
  --agent-code "$(cat agent.py)" \
  --pf 1.3 \
  --pnl 2000.0 \
  --dd 12.0 \
  --wr 52.0 \
  --sharpe 0.8 \
  --trades 150 \
  -o improved_agent.py
```

#### Креативный агент
```bash
python generate_agent.py creative -o creative_agent.py
```

### Программное использование

```python
from hean.agent_generation import AgentGenerator

# Инициализация генератора
generator = AgentGenerator()

# Генерация начального агента
code = generator.generate_agent(
    prompt_type="initial",
    output_path="my_agent.py"
)

# Генерация нескольких агентов
codes = generator.generate_initial_agents(
    count=10,
    output_dir="generated_agents"
)

# Эволюция агента
code = generator.evolve_agent(
    best_agents_info="Agent1: PF=2.5, WR=60%",
    worst_agents_info="Agent2: PF=0.8, WR=40%",
    market_conditions="High volatility, trending market",
    performance_metrics="Average PF: 1.5, Average WR: 50%",
    output_path="evolved_agent.py"
)

# Мутация агента
code = generator.mutate_agent(
    agent_code=existing_code,
    profit_factor=1.2,
    total_pnl=1000.0,
    max_drawdown_pct=15.0,
    win_rate=55.0,
    issues="Low win rate, high drawdown",
    output_path="improved_agent.py"
)
```

## Типы промптов

### 1. Initial (Начальный)
Генерация нового агента с нуля. Используется для создания начального набора агентов.

### 2. Evolution (Эволюция)
Создание улучшенного агента на основе лучших и худших практик существующих агентов.

### 3. Mutation (Мутация)
Точечное улучшение существующего агента на основе его метрик производительности.

### 4. Market Conditions (Рыночные условия)
Создание специализированного агента для конкретных рыночных условий.

### 5. Hybrid (Гибрид)
Комбинирование нескольких успешных агентов в один гибридный.

### 6. Problem Focused (Фокус на проблеме)
Создание агента, специально решающего конкретную проблему.

### 7. Evaluation (Оценка)
Анализ агента и генерация улучшенной версии.

### 8. Creative (Креативный)
Создание инновационного агента с нестандартным подходом.

### 9. Analytical (Аналитический)
Создание аналитического агента для анализа рынка (не торговая стратегия).

## Рекомендации по использованию

### Начальная генерация
1. Используйте `initial` промпт для создания первых 10-20 агентов
2. Тестируйте их на исторических данных
3. Отбирайте лучших по метрикам (PF, WR, Sharpe)

### Эволюция
1. Каждые 7 дней анализируйте производительность агентов
2. Используйте `evolution` промпт с данными лучших/худших
3. Тестируйте новые агенты перед добавлением в продакшн

### Специализация
1. Анализируйте текущие рыночные условия
2. Используйте `market_conditions` для создания специализированных агентов
3. Активируйте их только в соответствующих условиях

### Решение проблем
1. Выявляйте системные проблемы в торговле
2. Используйте `problem_focused` для создания целевых решений
3. Тестируйте на проблемных сценариях

### Гибриды
1. Комбинируйте лучших агентов по разным метрикам
2. Используйте `hybrid` для создания супер-агентов
3. Валидируйте, что гибрид превосходит исходные

## Структура сгенерированного кода

Все агенты следуют стандартной структуре:

```python
from hean.core.bus import EventBus
from hean.core.types import Event, Signal, Tick, FundingRate
from hean.core.regime import Regime
from hean.strategies.base import BaseStrategy
from hean.logging import get_logger

logger = get_logger(__name__)

class GeneratedAgent(BaseStrategy):
    """Описание агента."""
    
    def __init__(self, bus: EventBus, symbols: list[str] | None = None) -> None:
        super().__init__("agent_id", bus)
        # Инициализация
        
    async def on_tick(self, event: Event) -> None:
        """Обработка тиков."""
        # Логика
        
    async def on_funding(self, event: Event) -> None:
        """Обработка фандинга."""
        # Логика
        
    async def on_regime_update(self, event: Event) -> None:
        """Обработка смены режима."""
        # Логика
```

## Интеграция в систему

После генерации агента:

1. Сохраните код в `src/hean/strategies/`
2. Добавьте импорт в `src/hean/strategies/__init__.py`
3. Зарегистрируйте в основной системе торговли
4. Протестируйте на исторических данных
5. Мониторьте производительность

## Примеры использования

### Автоматическая генерация начального набора
```bash
# Генерируем 20 агентов
python generate_agent.py initial --count 20 -o generated_agents/

# Тестируем каждого
for agent in generated_agents/*.py; do
    python -m pytest tests/test_backtest.py --agent "$agent"
done
```

### Еженедельная эволюция
```bash
# Собираем метрики лучших/худших агентов
python collect_metrics.py > metrics.json

# Генерируем эволюционированного агента
python generate_agent.py evolution \
  --best-agents "$(jq -r '.best' metrics.json)" \
  --worst-agents "$(jq -r '.worst' metrics.json)" \
  --market-conditions "$(jq -r '.market' metrics.json)" \
  --performance-metrics "$(jq -r '.metrics' metrics.json)" \
  -o evolved_agent.py
```

## Примечания

- Все сгенерированные агенты проходят валидацию синтаксиса
- Код извлекается из markdown блоков, если LLM возвращает форматированный ответ
- Система автоматически определяет тип LLM клиента (OpenAI/Anthropic)
- Рекомендуется тестировать все сгенерированные агенты перед использованием

