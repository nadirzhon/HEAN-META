---
name: query-ticks
description: Выполняет прямой SQL-запрос к DuckDB для получения последних тиков.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash,Read
argument-hint: [symbol, limit]
---
## Objective
Получить последние `$ARGUMENTS[1]` тиков для символа `$ARGUMENTS[0]` из базы данных.

## Workflow
1.  Создай временный Python-скрипт.
2.  Скрипт должен:
    - Импортировать `DuckDBStore`.
    - Вызвать метод `query_ticks(symbol, limit)`.
    - Напечатать результат в удобном формате.
3.  Выполни и затем удали скрипт.
