# VectorBT + Optuna Backtesting System

Супер-быстрый backtesting движок с автоматической оптимизацией параметров для торговых стратегий.

## Возможности

### 🚀 Производительность
- **100x ускорение** через векторизацию (VectorBT)
- Использование Numba JIT компиляции
- Параллельная оптимизация на всех CPU ядрах
- Тестирование на 3+ годах данных за секунды

### 🎯 Оптимизация
- **Optuna** для умного поиска параметров
- Multi-objective optimization (максимизация прибыли + минимизация drawdown)
- Поддержка Pareto-оптимальных решений
- Early pruning для плохих trial'ов

### 📊 Walk-Forward Analysis
- Предотвращение overfitting
- Rolling и anchored окна
- Автоматическая реоптимизация
- Метрики стабильности параметров

### 📈 Визуализация
- Графики equity curve и drawdown
- Heatmap'ы параметров
- Trade distribution
- Walk-forward analysis charts
- Optimization history

### 📐 Статистика
- **Risk metrics**: Sharpe, Sortino, Calmar, Omega
- **Drawdown analysis**: Max DD, Avg DD, DD duration
- **Trade stats**: Win rate, Profit factor, Expectancy
- **Advanced metrics**: VaR, CVaR, Risk of Ruin

## Установка

```bash
pip install -e ".[backtesting]"
```

Или отдельно:
```bash
pip install vectorbt optuna matplotlib seaborn
```

## Быстрый старт

### 1. Простой backtesting

```python
from hean.backtesting import VectorBTEngine, BacktestConfig
from hean.backtesting.vectorbt_engine import create_simple_ma_crossover_signals

# Настройка
config = BacktestConfig(
    initial_capital=10000,
    commission=0.0006,
    slippage=0.0002,
)

# Создание движка
engine = VectorBTEngine(config)

# Генерация сигналов
entries, exits = create_simple_ma_crossover_signals(
    data,
    fast_period=10,
    slow_period=50
)

# Запуск backtest
result = engine.backtest(data, entries, exits)

print(result)
# BacktestResult(
#   Total Return: 45.23%
#   Sharpe: 1.85
#   Max DD: -12.4%
#   Win Rate: 58.3%
#   ...
# )
```

### 2. Оптимизация параметров

```python
from hean.backtesting import OptunaOptimizer, OptimizationConfig

# Настройка оптимизации
opt_config = OptimizationConfig(
    n_trials=100,
    n_jobs=-1,  # Все CPU
)

# Создание оптимизатора
optimizer = OptunaOptimizer(engine, data, opt_config)

# Определение пространства параметров
param_space = {
    'fast_period': ('int', 5, 30),
    'slow_period': ('int', 20, 100),
}

# Запуск оптимизации
result = optimizer.optimize_strategy(
    create_simple_ma_crossover_signals,
    param_space,
    objective_metric='sharpe_ratio',
)

print(result.best_params)
# {'fast_period': 12, 'slow_period': 48}
```

### 3. Multi-objective оптимизация

```python
# Максимизация прибыли + минимизация drawdown
result = optimizer.optimize_multi_objective(
    strategy_func,
    param_space,
    objectives=['sharpe_ratio', 'max_drawdown'],
)

# Получить Pareto-оптимальные решения
pareto_trials = result.study.best_trials
for trial in pareto_trials[:5]:
    print(f"Sharpe: {trial.values[0]:.3f}, DD: {trial.values[1]:.2%}")
```

### 4. Walk-Forward Analysis

```python
from hean.backtesting import WalkForwardAnalysis, WalkForwardConfig

# Настройка
wf_config = WalkForwardConfig(
    train_window_months=6,
    test_window_months=2,
    step_months=1,
    anchored=False,
)

# Создание анализатора
wfa = WalkForwardAnalysis(engine, wf_config)

# Запуск
wf_result = wfa.run(data, strategy_func, param_space)

print(wf_result)
# WalkForwardResult(
#   Windows: 30
#   Avg Train Sharpe: 2.1
#   Avg Test Sharpe: 1.6
#   Overfitting Ratio: 0.76  (хорошо!)
#   ...
# )
```

### 5. Визуализация

```python
from hean.backtesting import BacktestVisualizer

visualizer = BacktestVisualizer()

# Backtest summary
visualizer.plot_backtest_summary(result)

# Optimization results
visualizer.plot_optimization_results(opt_result)

# Walk-forward analysis
visualizer.plot_walk_forward_results(wf_result)

# Parameter heatmap
visualizer.plot_parameter_heatmap(
    results_df,
    'fast_period',
    'slow_period',
    metric='sharpe_ratio',
)
```

## Примеры

Полный пример использования: [examples/backtesting_example.py](../../../examples/backtesting_example.py)

```bash
python examples/backtesting_example.py
```

## Архитектура

```
src/hean/backtesting/
├── __init__.py              # Экспорт основных классов
├── vectorbt_engine.py       # Основной backtesting движок
├── optuna_optimizer.py      # Оптимизация параметров
├── walk_forward.py          # Walk-forward analysis
├── metrics.py               # Расчет статистики
├── visualization.py         # Визуализация
└── README.md               # Документация
```

## Основные классы

### VectorBTEngine
Основной backtesting движок на базе VectorBT.

**Методы:**
- `backtest(data, entries, exits)` - базовый backtest
- `backtest_custom_strategy(data, strategy_func, params)` - кастомная стратегия
- `backtest_indicator(data, indicator, thresholds)` - индикаторная стратегия
- `run_monte_carlo(result, n_simulations)` - Monte Carlo симуляция

### OptunaOptimizer
Оптимизация параметров с помощью Optuna.

**Методы:**
- `optimize_strategy(strategy_func, param_space, objective_metric)` - оптимизация
- `optimize_multi_objective(strategy_func, param_space, objectives)` - multi-objective
- `grid_search(strategy_func, param_grid)` - grid search

### WalkForwardAnalysis
Walk-forward analysis для предотвращения overfitting.

**Методы:**
- `run(data, strategy_func, param_space)` - запуск WFA
- `get_summary_df(result)` - summary таблица

### BacktestVisualizer
Визуализация результатов.

**Методы:**
- `plot_backtest_summary(result)` - полный отчет
- `plot_optimization_results(result)` - optimization history
- `plot_walk_forward_results(result)` - WFA charts
- `plot_parameter_heatmap(df, param1, param2)` - heatmap

## Метрики

### Returns
- Total Return
- Annualized Return
- CAGR
- Monthly Returns

### Risk
- Volatility (annualized)
- Max Drawdown
- Average Drawdown
- Downside Deviation
- VaR (95%)
- CVaR (95%)

### Risk-Adjusted
- **Sharpe Ratio** - (Return - RiskFree) / Volatility
- **Sortino Ratio** - (Return - RiskFree) / Downside Deviation
- **Calmar Ratio** - Return / Max Drawdown
- **Omega Ratio** - Gains / Losses

### Trades
- Total Trades
- Win Rate
- Profit Factor
- Average Win/Loss
- Expectancy
- Payoff Ratio

## Best Practices

### 1. Предотвращение Overfitting
```python
# ✅ Всегда используйте walk-forward analysis
wfa = WalkForwardAnalysis(engine, config)
wf_result = wfa.run(data, strategy_func, param_space)

# ✅ Проверяйте overfitting ratio
if wf_result.avg_overfitting_ratio < 0.5:
    print("WARNING: Severe overfitting detected!")

# ✅ Используйте out-of-sample тестирование
train_data = data[:int(len(data) * 0.7)]
test_data = data[int(len(data) * 0.7):]
```

### 2. Выбор параметров
```python
# ✅ Используйте разумные границы
param_space = {
    'period': ('int', 5, 100),  # Не слишком широкий диапазон
    'threshold': ('float', 0.1, 0.9),
}

# ❌ Избегайте слишком широких диапазонов
param_space = {
    'period': ('int', 1, 1000),  # Слишком широко
}
```

### 3. Multi-objective optimization
```python
# ✅ Балансируйте прибыль и риск
objectives = ['sharpe_ratio', 'max_drawdown']

# ✅ Проверяйте Pareto frontier
pareto_trials = result.study.best_trials
# Выбирайте решение в зависимости от risk tolerance
```

### 4. Производительность
```python
# ✅ Используйте параллелизацию
config = OptimizationConfig(
    n_jobs=-1,  # Все CPU
    enable_pruning=True,  # Early stopping
)

# ✅ Используйте Numba
engine_config = BacktestConfig(use_numba=True)
```

## Benchmark

Пример производительности на 3 годах часовых данных (~26,000 баров):

| Метод | Время |
|-------|-------|
| Простой backtest | 0.05s |
| 100 trials optimization | 5s |
| Walk-forward (30 windows) | 150s |
| Grid search (100 combos) | 2s |

**Система**: AMD Ryzen 9 5950X (16 cores), 64GB RAM

## Troubleshooting

### Ошибка: "vectorbt not found"
```bash
pip install vectorbt
```

### Ошибка: "Numba compilation failed"
```python
# Отключите Numba
config = BacktestConfig(use_numba=False)
```

### Медленная оптимизация
```python
# Уменьшите n_trials
config = OptimizationConfig(n_trials=50)  # Вместо 100

# Включите pruning
config = OptimizationConfig(enable_pruning=True)

# Используйте параллелизацию
config = OptimizationConfig(n_jobs=-1)
```

## Roadmap

- [ ] Поддержка futures и options
- [ ] Интеграция с live trading
- [ ] Reinforcement Learning оптимизация
- [ ] Portfolio backtesting (multiple strategies)
- [ ] Real-time performance tracking

## License

MIT

## Credits

- **VectorBT**: https://github.com/polakowo/vectorbt
- **Optuna**: https://github.com/optuna/optuna

Developed as part of HEAN-META trading system.
