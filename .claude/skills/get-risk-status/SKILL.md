---
name: get-risk-status
description: Запрашивает и отображает текущий статус модулей управления рисками.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---
## Objective
Получить актуальную информацию о состоянии `RiskGovernor` и `KillSwitch`.

## Commands
Выполни две команды и покажи их вывод.

1. **Статус Risk Governor:**
```bash
curl http://localhost:8000/api/v1/risk/governor/status
```

2. **Статус KillSwitch:**
```bash
curl http://localhost:8000/api/v1/risk/killswitch/status
```
