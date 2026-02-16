---
name: full-system-diagnostics
description: Запускает комплексную диагностику всей системы и формирует сводный отчет.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash,Read
---
## Objective
Собрать полную диагностическую информацию о системе HEAN в один отчет.

## Workflow
Это мета-скилл. Последовательно вызови следующие скиллы и объедини их выводы в единый отчет:
1.  `/check-system-health`
2.  `/get-risk-status`
3.  `/get-market-physics`
4.  `/get-brain-analysis`
5.  Выполни `docker-compose ps` для отображения статуса всех контейнеров.
