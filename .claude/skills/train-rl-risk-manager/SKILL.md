---
name: train-rl-risk-manager
description: Запускает скрипт для тренировки AI-модели управления рисками (Reinforcement Learning).
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash
---
## Objective
Запустить тренировку PPO-агента для `RLRiskManager`.

## Command
```bash
python3 scripts/train_rl_risk.py --timesteps 50000
```
