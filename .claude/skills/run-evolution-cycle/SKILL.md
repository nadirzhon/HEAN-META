---
name: run-evolution-cycle
description: Запускает цикл эволюционного бэктестинга для Symbiont X.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---

## Objective
Запустить полный цикл эволюционного бэктестинга для системы `Symbiont X`.

## Workflow
Скрипт `run_evolution_test.py`, находящийся в директории `scripts/` этого скилла, выполнит весь процесс:
1. Загрузит исторические данные.
2. Создаст популяцию случайных стратегий.
3. Запустит бэктест для оценки каждой стратегии.
4. Выведет итоговый отчет с рейтингом производительности.

## Command
Выполни следующую команду в терминале:

```bash
python3 .claude/skills/run-evolution-cycle/scripts/run_evolution_test.py
```
