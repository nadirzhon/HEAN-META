---
name: integrate-google-trends
description: Устанавливает зависимости и запускает тесты для интеграции Google Trends.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---
## Objective
Выполнить первые шаги по интеграции Google Trends в качестве нового источника данных для Oracle.

## Workflow
1.  Установи зависимости: `pip install -r requirements_google_trends.txt`
2.  Проверь, существуют ли тесты для этого модуля, и запусти их.
3.  Прочитай `GOOGLE_TRENDS_GUIDE.md` и предоставь краткую сводку о том, как этот модуль должен быть интегрирован в `ContextAggregator`.
