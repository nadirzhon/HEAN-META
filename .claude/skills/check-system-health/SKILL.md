---
name: check-system-health
description: Проводит быструю проверку "здоровья" всей системы.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---
## Objective
Выполнить серию команд для быстрой диагностики состояния системы HEAN.

## Workflow
Выполни следующие команды последовательно и предоставь сводный отчет.
1.  **Проверка качества кода:** `make lint`
2.  **Быстрые тесты:** `make test-quick`
3.  **Статус Docker-контейнеров:** `docker-compose ps`
4.  **Статус API движка:** `curl http://localhost:8000/api/v1/engine/status` (если система запущена).
