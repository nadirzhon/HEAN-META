---
name: clean-docker
description: Безопасно останавливает и удаляет все Docker-контейнеры и тома проекта.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---
## Objective
Полностью очистить Docker-окружение проекта, используя команду из Makefile.

## Command
```bash
make docker-clean
```
