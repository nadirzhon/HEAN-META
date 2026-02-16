---
name: reset-killswitch
description: Попытка сбросить KillSwitch после его срабатывания.
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---
## Objective
Отправить команду на сброс `KillSwitch`, если он был активирован.

## Warning
Используйте эту команду только после анализа и устранения причины, вызвавшей срабатывание KillSwitch.

## Command
```bash
# Эта команда должна быть реализована в API
# curl -X POST http://localhost:8000/api/v1/risk/killswitch/reset
echo "Функционал сброса KillSwitch через API еще не реализован. Это - плейсхолдер."
```
