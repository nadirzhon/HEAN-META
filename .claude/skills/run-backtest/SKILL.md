---
name: run-backtest
description: Запускает бэктест для одной конкретной стратегии на исторических данных.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash,Read
argument-hint: [strategy_name, days_back]
---
## Objective
Запустить бэктест для стратегии `$ARGUMENTS[0]` на данных за последние `$ARGUMENTS[1]` дней.

## Workflow
1.  Создай временный Python-скрипт.
2.  Скрипт должен:
    - Загрузить OHLCV данные за указанный период с помощью `DuckDBStore`.
    - Создать геном для указанной стратегии (потребуется найти или создать функцию для этого).
    - Запустить `BacktestEngine.run_backtest()` с этим геномом и данными.
    - Вывести результаты.
3.  Выполни и затем удали скрипт.
