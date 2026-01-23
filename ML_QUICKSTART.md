# ML Stack - Быстрый старт

## 1. Установка зависимостей

```bash
# Установить ML зависимости
pip install -e ".[ml]"

# Или установить отдельно
pip install lightgbm xgboost catboost scikit-learn pandas
```

## 2. Обучить модель

```bash
python scripts/train_ml_model.py
```

Результат:
- Модель сохранена в `models/bitcoin_predictor/`
- Метрики: accuracy, precision, recall, F1
- Feature importance

## 3. Запустить предсказания

```bash
python scripts/run_ml_predictions.py
```

Результат:
- Направление: UP/DOWN
- Вероятность: 0-100%
- Confidence: 0-100%
- Торговая рекомендация

## 4. Использование в коде

### Training
```python
from hean.ml.training import ModelTrainer

trainer = ModelTrainer()
results = trainer.train(ohlcv_data)
trainer.save_model('models/my_model')
```

### Prediction
```python
from hean.ml.inference import MLPredictor

predictor = MLPredictor('models/bitcoin_predictor')
result = predictor.predict(ohlcv_data)

print(f"Direction: {result['direction']}")
print(f"Probability: {result['probability']:.2%}")
```

### Backtesting
```python
from hean.ml.backtesting import Backtester

backtester = Backtester()
results = backtester.backtest(predictions, prices)
backtester.print_results()
```

## 5. Auto-Retraining

```python
from hean.ml.auto_retrain import RetrainingScheduler

async def data_provider():
    # Fetch fresh data
    return ohlcv_data, orderbook_data, sentiment_data

scheduler = RetrainingScheduler({'retrain_interval_hours': 24})
await scheduler.start(data_provider)
```

## Подробная документация

См. [ML_STACK_README.md](ML_STACK_README.md)
