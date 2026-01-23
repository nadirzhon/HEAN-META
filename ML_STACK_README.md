# Bitcoin Price Prediction ML Stack

Полный ML стек для предсказания цены Bitcoin с использованием ансамбля моделей.

## 🎯 Особенности

- **Ensemble модель**: LightGBM + XGBoost + CatBoost
- **50+ фич**: RSI, MACD, Volume, Orderbook, Sentiment и др.
- **Target**: Предсказание движения цены через 5 минут (up/down)
- **Auto-retraining**: Автоматическое переобучение каждые 24 часа
- **Backtesting**: Тестирование на исторических данных
- **Production-ready**: Готовый inference для production

## 📁 Структура

```
src/hean/ml/
├── __init__.py
├── features/                  # Feature Engineering
│   ├── feature_engineer.py    # Главный модуль (50+ фич)
│   ├── technical_indicators.py # Технические индикаторы
│   ├── volume_features.py     # Volume фичи
│   ├── orderbook_features.py  # Orderbook фичи
│   └── sentiment_features.py  # Sentiment фичи
├── models/                    # ML Models
│   ├── ensemble.py            # Ensemble модель
│   ├── lightgbm_model.py      # LightGBM
│   ├── xgboost_model.py       # XGBoost
│   └── catboost_model.py      # CatBoost
├── training/                  # Training Pipeline
│   ├── trainer.py             # Главный trainer
│   └── data_splitter.py       # Разделение данных
├── inference/                 # Production Inference
│   └── predictor.py           # ML Predictor
├── backtesting/              # Backtesting
│   └── backtester.py          # Backtester
├── metrics/                   # Метрики
│   └── evaluator.py           # Model Evaluator
└── auto_retrain/             # Auto-Retraining
    └── scheduler.py           # Retraining Scheduler
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Основные ML библиотеки
pip install lightgbm xgboost catboost scikit-learn

# Уже установленные зависимости
# numpy, pandas (уже в pyproject.toml)
```

### 2. Обучение модели

```bash
python scripts/train_ml_model.py
```

Этот скрипт:
- Загружает данные
- Создает 50+ фич
- Обучает ensemble модель
- Оценивает на train/val/test
- Сохраняет модель в `models/bitcoin_predictor/`

### 3. Запуск предсказаний

```bash
python scripts/run_ml_predictions.py
```

Этот скрипт:
- Загружает обученную модель
- Делает предсказание на свежих данных
- Показывает вероятность и confidence
- Выдает торговые рекомендации

## 📊 Feature Engineering (50+ фич)

### Technical Indicators (20+ фич)
- **RSI**: Relative Strength Index (14, 21 периодов)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Upper, Middle, Lower bands + %B
- **Moving Averages**: SMA и EMA (5, 10, 20, 50, 100, 200)
- **Stochastic**: %K и %D
- **ADX**: Average Directional Index
- **ATR**: Average True Range
- **CCI**: Commodity Channel Index
- **Williams %R**
- **Ichimoku Cloud**

### Volume Features (10+ фич)
- Volume changes и trends
- **OBV**: On-Balance Volume
- **VWAP**: Volume-Weighted Average Price
- **MFI**: Money Flow Index
- Volume oscillators
- Volume ratios

### Orderbook Features (10+ фич)
- Bid-Ask spread
- Order imbalance
- Liquidity metrics
- Market depth
- Buying/Selling pressure

### Sentiment Features (5+ фич)
- Fear & Greed Index (если доступен)
- Synthetic sentiment из price/volume
- Momentum sentiment
- Volatility sentiment

### Price Action Features (5+ фич)
- High-Low range
- Body size
- Shadows
- Gaps
- Bullish/Bearish patterns

### Volatility Features
- Rolling standard deviation
- Historical volatility
- True Range

### Momentum Features
- Rate of Change (ROC)
- Momentum
- Acceleration
- Velocity

### Time Features
- Hour, Day of week
- Weekend flag
- Cyclical encoding

## 🤖 Ensemble Model

### LightGBM
```python
params = {
    'objective': 'binary',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8
}
```

### XGBoost
```python
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### CatBoost
```python
params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'Logloss'
}
```

### Weighted Voting
Веса моделей обновляются на основе validation performance:

```python
weights = {
    'lightgbm': accuracy_lgb / total_accuracy,
    'xgboost': accuracy_xgb / total_accuracy,
    'catboost': accuracy_cat / total_accuracy
}
```

## 📈 Метрики оценки

### ML Metrics
- **Accuracy**: Общая точность
- **Precision**: Точность положительных предсказаний
- **Recall**: Полнота (sensitivity)
- **F1 Score**: Гармоническое среднее precision и recall
- **ROC AUC**: Area Under ROC Curve
- **Confusion Matrix**: TP, TN, FP, FN
- **MCC**: Matthews Correlation Coefficient

### Trading Metrics
- **Total Return**: Общая доходность
- **Win Rate**: Процент прибыльных сделок
- **Sharpe Ratio**: Риск-скорректированная доходность
- **Sortino Ratio**: Downside risk metric
- **Max Drawdown**: Максимальная просадка
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Return / Max Drawdown

## 🔄 Auto-Retraining

Автоматическое переобучение каждые 24 часа:

```python
from hean.ml.auto_retrain import RetrainingScheduler

scheduler = RetrainingScheduler({
    'retrain_interval_hours': 24,
    'min_accuracy': 0.55,
    'max_performance_drop': 0.05
})

await scheduler.start(data_provider_function)
```

Features:
- Scheduled retraining (default: 24h)
- Performance-based triggers
- Safe model replacement
- Automatic backup/restore
- Training history tracking

## 🎓 Использование в коде

### Training

```python
from hean.ml.training import ModelTrainer

trainer = ModelTrainer(config)

results = trainer.train(
    ohlcv_data,
    orderbook_data=None,
    sentiment_data=None
)

trainer.save_model('models/bitcoin_predictor')
```

### Inference

```python
from hean.ml.inference import MLPredictor

predictor = MLPredictor('models/bitcoin_predictor')

result = predictor.predict(
    ohlcv_data,
    return_probabilities=True
)

print(f"Direction: {result['direction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Backtesting

```python
from hean.ml.backtesting import Backtester

backtester = Backtester({
    'trading_fee': 0.001,
    'initial_capital': 10000
})

results = backtester.backtest(
    predictions,
    prices,
    timestamps
)

backtester.print_results()
```

## 🔧 Конфигурация

Пример полной конфигурации:

```python
config = {
    'features': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2
    },
    'model': {
        'lightgbm': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8
        },
        'xgboost': {
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'catboost': {
            'depth': 6,
            'learning_rate': 0.05,
            'iterations': 1000
        }
    },
    'data_split': {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    },
    'backtesting': {
        'trading_fee': 0.001,
        'slippage': 0.0005,
        'initial_capital': 10000
    },
    'auto_retrain': {
        'retrain_interval_hours': 24,
        'min_accuracy': 0.55,
        'max_performance_drop': 0.05
    }
}
```

## 📊 Пример вывода

### Training Output
```
==============================================================================
Model Evaluation Results - test
==============================================================================
Timestamp: 2026-01-23T21:00:00
Samples: 3000

Core Metrics
--------------------------------------------------------------
Accuracy:  0.5850
Precision: 0.5920
Recall:    0.5780
F1 Score:  0.5849
ROC AUC:   0.6340

Confusion Matrix
--------------------------------------------------------------
True Negatives:    820  |  False Positives:  680
False Negatives:   635  |  True Positives:   865

Top 20 Important Features:
feature                        importance
rsi                                 1250.5
macd_hist                          1180.2
bb_pct                             1050.8
...
```

### Prediction Output
```
==============================================================================
Bitcoin Price Prediction - ML Inference
==============================================================================

Prediction Results:
  Direction: UP
  Probability: 68.50%
  Confidence: 74.00%
  Inference Time: 45.32ms

Model Ensemble Breakdown:
  LightGBM: 0.6720
  XGBoost:  0.6850
  CatBoost: 0.6980

Trading Recommendation:
==============================================================================
  🟢 BUY signal
  Confidence Level: 74.0%
  Expected Movement: UP
```

## 🎯 Best Practices

1. **Data Quality**: Используйте качественные данные с достаточной историей (минимум 200 свечей)
2. **Feature Engineering**: Настройте параметры индикаторов под свой таймфрейм
3. **Model Validation**: Всегда проверяйте модель на out-of-sample данных
4. **Backtesting**: Тестируйте на исторических данных перед production
5. **Monitoring**: Отслеживайте performance модели в production
6. **Retraining**: Регулярно переобучайте модель на свежих данных

## 🔍 Troubleshooting

### Модель не обучается
- Проверьте достаточность данных (минимум 10000 сэмплов)
- Убедитесь что установлены все зависимости
- Проверьте логи на ошибки

### Низкая точность
- Увеличьте количество данных
- Настройте параметры моделей
- Добавьте дополнительные фичи
- Попробуйте другой split метод

### Медленный inference
- Уменьшите количество фич
- Используйте более простые модели
- Оптимизируйте feature engineering

## 📝 TODO / Roadmap

- [ ] Интеграция с реальным API биржи
- [ ] Реальные orderbook данные
- [ ] Sentiment analysis из Twitter/Reddit
- [ ] Deep Learning модели (LSTM, Transformer)
- [ ] Multi-timeframe predictions
- [ ] Portfolio optimization
- [ ] Risk management integration

## 📄 License

MIT License - часть проекта HEAN

## 🙏 Credits

Разработано с использованием:
- LightGBM, XGBoost, CatBoost
- Scikit-learn
- Pandas, NumPy
- Claude AI (Anthropic)
