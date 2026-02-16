---
name: deploy-to-staging
description: Запускает процесс развертывания проекта в тестовое (staging) окружение.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---
## Objective
Выполнить деплой последней версии кода на staging-сервер.

## Command
```bash
# Предполагается, что существует скрипт для деплоя
# ./scripts/deploy.sh --env staging
echo "Скрипт для деплоя еще не создан. Это - плейсхолдер для будущей CI/CD функциональности."
```
