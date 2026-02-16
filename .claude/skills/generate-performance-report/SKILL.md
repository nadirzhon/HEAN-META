---
name: generate-performance-report
description: Собирает данные о производительности и генерирует отчет в формате Markdown.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash,Read
argument-hint: [period]
---
## Objective
Создать отчет о производительности системы за период `$ARGUMENTS[0]` (например, 'daily', 'weekly').

## Workflow
1.  Собери данные из различных источников:
    - Общий PnL и Sharpe из API.
    - Производительность топ-5 и топ-3 худших стратегий.
    - Количество срабатываний `KillSwitch` и `RiskGovernor`.
2.  Сформируй из этих данных красивый и читаемый Markdown-отчет.
3.  Выведи отчет в консоль.
