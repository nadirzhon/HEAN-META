---
name: get-brain-analysis
description: Запрашивает и отображает последний анализ рынка от компонента Brain (Claude AI).
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---
## Objective
Получить последний отчет о состоянии рынка от `Brain`.

## Command
```bash
curl http://localhost:8000/api/v1/brain/analysis
```
