---
name: view-logs
description: Показывает логи для указанного Docker-сервиса.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
argument-hint: [service_name]
---
## Objective
Показать последние логи для сервиса `$ARGUMENTS[0]`. Если сервис не указан, показать для всех.

## Command
```bash
docker-compose logs --tail=100 -f $ARGUMENTS
```
