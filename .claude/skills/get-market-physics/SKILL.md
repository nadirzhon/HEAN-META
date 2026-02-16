---
name: get-market-physics
description: Получает и отображает текущие "физические" метрики рынка (температура, энтропия, фаза).
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
argument-hint: [symbol]
---
## Objective
Получить данные от компонента "Physics" для символа `$ARGUMENTS[0]` (по умолчанию BTCUSDT).

## Command
Символ для запроса: `${1:-BTCUSDT}`
```bash
curl http://localhost:8000/api/v1/physics/state?symbol=${1:-BTCUSDT}
```
