# Быстрый старт: Генерация агентов

## Установка

```bash
# Установить опциональные зависимости для LLM
pip install -e ".[llm]"

# Или вручную:
pip install openai anthropic
```

## Настройка API ключа

```bash
# Для OpenAI
export OPENAI_API_KEY="your-api-key"

# Или для Anthropic (Claude)
export ANTHROPIC_API_KEY="your-api-key"
```

## Базовое использование

### 1. Генерация одного агента

```bash
python generate_agent.py initial -o my_agent.py
```

### 2. Генерация нескольких агентов

```bash
python generate_agent.py initial --count 10 -o agents/
```

### 3. Эволюция агента

```bash
python generate_agent.py evolution \
  --best-agents "Лучший агент: PF=2.5, WR=60%" \
  --worst-agents "Худший агент: PF=0.8, WR=40%" \
  --market-conditions "Высокая волатильность, восходящий тренд" \
  --performance-metrics "Средний PF: 1.5, Средний WR: 50%" \
  -o evolved_agent.py
```

### 4. Улучшение существующего агента

```bash
python generate_agent.py mutation \
  --agent-code "$(cat existing_agent.py)" \
  --profit-factor 1.2 \
  --total-pnl 1000.0 \
  --max-drawdown 15.0 \
  --win-rate 55.0 \
  --issues "Низкий win rate, высокий drawdown" \
  -o improved_agent.py
```

## Программное использование

```python
from hean.agent_generation import AgentGenerator

# Инициализация
generator = AgentGenerator()

# Генерация агента
code = generator.generate_agent(
    prompt_type="initial",
    output_path="agent.py"
)

# Генерация нескольких
codes = generator.generate_initial_agents(
    count=10,
    output_dir="agents"
)
```

## Типы промптов

- `initial` - Начальная генерация нового агента
- `evolution` - Эволюция на основе лучших/худших агентов
- `mutation` - Улучшение существующего агента
- `market_conditions` - Агент для конкретных рыночных условий
- `hybrid` - Гибридный агент из нескольких
- `problem_focused` - Агент для решения проблемы
- `evaluation` - Оценка и улучшение
- `creative` - Креативный инновационный агент
- `analytical` - Аналитический агент (не торговая стратегия)

## Интеграция в систему

После генерации агента:

1. Сохраните в `src/hean/strategies/`
2. Добавьте импорт в `src/hean/strategies/__init__.py`
3. Зарегистрируйте в основной системе
4. Протестируйте на исторических данных

## Примеры

См. `examples/generate_agent_example.py` для полных примеров.

## Документация

Полная документация: [AGENT_GENERATION.md](AGENT_GENERATION.md)

