# HEAN Docker Build Guide

## Содержание

1. [Требования](#1-требования)
2. [Структура Docker-файлов](#2-структура-docker-файлов)
3. [Настройка окружения](#3-настройка-окружения)
4. [Быстрый старт (Development)](#4-быстрый-старт-development)
5. [Запуск отдельных сервисов](#5-запуск-отдельных-сервисов)
6. [Production-сборка](#6-production-сборка)
7. [Мониторинг (Prometheus + Grafana)](#7-мониторинг-prometheus--grafana)
8. [Архитектура и потоки данных](#8-архитектура-и-потоки-данных)
9. [Порты и сети](#9-порты-и-сети)
10. [Проверка здоровья](#10-проверка-здоровья)
11. [Управление и обслуживание](#11-управление-и-обслуживание)
12. [Решение проблем](#12-решение-проблем)
13. [Kubernetes (продвинутый)](#13-kubernetes-продвинутый)

---

## 1. Требования

| Компонент | Минимальная версия | Проверка |
|-----------|-------------------|----------|
| Docker | 24.0+ | `docker --version` |
| Docker Compose | v2.20+ | `docker compose version` |
| RAM | 8 GB (рекомендовано) | Суммарный лимит контейнеров: ~6.5 GB |
| Диск | 5 GB свободно | Для образов и данных |
| Порты | 8000, 3000, 6379 | Должны быть свободны |

Убедитесь, что Docker Desktop запущен:

```bash
# macOS
open -a "Docker Desktop"

# Проверка готовности
docker info > /dev/null 2>&1 && echo "Docker готов" || echo "Docker не запущен"
```

---

## 2. Структура Docker-файлов

```
HEAN/
├── docker-compose.yml                # Основной стек (8 сервисов)
├── docker-compose.production.yml     # Production (3 реплики API, TLS)
├── docker-compose.monitoring.yml     # Мониторинг отдельно
│
├── api/
│   ├── Dockerfile                    # API backend (multi-stage, Python 3.11)
│   └── Dockerfile.optimized          # API production (то же + комментарии)
│
├── apps/ui/
│   ├── Dockerfile                    # UI frontend (Node 20 → nginx)
│   ├── Dockerfile.dev                # UI dev с hot-reload (port 5173)
│   ├── Dockerfile.optimized          # UI production (hardened nginx)
│   └── nginx.conf                    # Proxy /api → api:8000, /ws → WebSocket
│
├── Dockerfile                        # Symbiont X (общий demo)
├── Dockerfile.testnet                # Symbiont Bybit Testnet (live trading)
├── Dockerfile.symbiont               # Symbiont с тестами
│
├── services/
│   ├── collector/Dockerfile          # Сбор рыночных данных (Python 3.12)
│   ├── physics/Dockerfile            # Расчёт физики рынка (Python 3.12)
│   ├── brain/Dockerfile              # AI принятие решений (Python 3.12)
│   └── risk/Dockerfile               # Риск-менеджмент (Python 3.12)
│
├── redis.conf                        # Конфигурация Redis (для production)
├── monitoring/
│   ├── prometheus/prometheus.yml      # Конфигурация Prometheus
│   └── grafana/                      # Дашборды и datasources Grafana
│
└── k8s/                              # Kubernetes манифесты
```

---

## 3. Настройка окружения

### 3.1 Создание файлов окружения

Перед первой сборкой нужно создать три env-файла:

```bash
# 1. Основной .env (используется сервисом brain)
cp .env.example .env

# 2. Backend API конфигурация
cp backend.env.example backend.env

# 3. Symbiont testnet конфигурация
cp .env.symbiont.example .env.symbiont
```

### 3.2 Обязательные переменные

Отредактируйте файлы и заполните:

**`.env`** — минимально необходимое:
```env
# Bybit Testnet API ключи (получить на testnet.bybit.com)
BYBIT_API_KEY=ваш_ключ
BYBIT_API_SECRET=ваш_секрет
BYBIT_TESTNET=true

# Начальный капитал (USDT)
INITIAL_CAPITAL=300

# AI провайдер (опционально, один из трёх)
ANTHROPIC_API_KEY=sk-ant-...    # для Claude
# OPENAI_API_KEY=sk-...         # для GPT
# GEMINI_API_KEY=AI...           # для Gemini
```

**`backend.env`** — конфигурация API:
```env
BYBIT_API_KEY=ваш_ключ
BYBIT_API_SECRET=ваш_секрет
BYBIT_TESTNET=true
INITIAL_CAPITAL=300
LIVE_CONFIRM=YES

# Стратегии
FUNDING_HARVESTER_ENABLED=true
BASIS_ARBITRAGE_ENABLED=true
IMPULSE_ENGINE_ENABLED=true

# Символы для торговли
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT

# Redis (внутри Docker)
REDIS_URL=redis://redis:6379/0
```

**`.env.symbiont`** — конфигурация Symbiont:
```env
BYBIT_API_KEY=ваш_ключ
BYBIT_API_SECRET=ваш_секрет
BYBIT_TESTNET=true
INITIAL_CAPITAL=300
REDIS_URL=redis://redis:6379/0
```

### 3.3 Проверка конфигурации

```bash
# Проверить наличие всех env-файлов
ls -la .env backend.env .env.symbiont

# Проверить что ключи заполнены (не placeholder)
grep -c "your_" .env backend.env .env.symbiont
# Должно вернуть 0 для каждого файла
```

---

## 4. Быстрый старт (Development)

### 4.1 Полный стек — одна команда

```bash
docker compose up -d --build
```

Эта команда собирает и запускает **8 сервисов**:

| Сервис | Описание | Порт |
|--------|----------|------|
| `redis` | Хранилище состояний | 6379 |
| `api` | FastAPI бэкенд | **8000** |
| `ui` | React фронтенд (nginx) | **3000** |
| `collector` | Сбор данных с Bybit WS | внутренний |
| `physics` | Расчёт рыночной физики | внутренний |
| `brain` | AI принятие решений | внутренний |
| `risk-svc` | Контроль рисков | внутренний |
| `symbiont-testnet` | Торговля на Bybit Testnet | внутренний |

### 4.2 Порядок запуска (автоматический)

```
redis (стартует первым, healthcheck: redis-cli ping)
  │
  ├── api          (ждёт redis healthy)
  │   └── ui       (ждёт api)
  │
  ├── collector    (ждёт redis healthy)
  ├── physics      (ждёт redis healthy)
  ├── brain        (ждёт redis healthy)
  ├── risk-svc     (ждёт redis healthy)
  └── symbiont     (ждёт redis healthy)
```

### 4.3 Проверка после запуска

```bash
# Статус всех контейнеров
docker compose ps

# API здоров?
curl -s http://localhost:8000/health | python3 -m json.tool

# UI доступен?
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000

# Логи в реальном времени
docker compose logs -f

# Логи конкретного сервиса
docker compose logs -f api
```

Ожидаемый результат `curl http://localhost:8000/health`:
```json
{
    "status": "healthy",
    "components": {
        "api": "healthy",
        "event_bus": "running",
        "redis": "connected",
        "engine": "running"
    }
}
```

### 4.4 Доступ к сервисам

| URL | Что открывается |
|-----|-----------------|
| http://localhost:3000 | Торговый дашборд (React UI) |
| http://localhost:8000/docs | API документация (Swagger) |
| http://localhost:8000/health | Healthcheck API |
| http://localhost:8000/api/v1/engine/status | Статус торгового движка |

---

## 5. Запуск отдельных сервисов

### 5.1 Только API + Redis (без микросервисов)

```bash
docker compose up -d redis api
```

### 5.2 API + Redis + UI (без торговли)

```bash
docker compose up -d redis api ui
```

### 5.3 Полный стек без Symbiont (наблюдение без торговли)

```bash
docker compose up -d redis api ui collector physics brain risk-svc
```

### 5.4 Только Symbiont (live trading)

```bash
docker compose up -d redis symbiont-testnet
```

### 5.5 Пересборка одного сервиса

```bash
# Пересобрать только API
docker compose up -d --build api

# Пересобрать только UI
docker compose up -d --build ui

# Пересобрать все микросервисы
docker compose up -d --build collector physics brain risk-svc
```

---

## 6. Production-сборка

### 6.1 Подготовка

```bash
# Создать production env
cp .env.production.example .env.production
```

Отредактируйте `.env.production`:
```env
BYBIT_TESTNET=false        # ВНИМАНИЕ: реальная торговля!
REDIS_PASSWORD=надёжный_пароль
GRAFANA_ADMIN_PASSWORD=admin_пароль
```

### 6.2 Запуск

```bash
docker compose -f docker-compose.production.yml up -d --build
```

Отличия от development:
- **3 реплики API** с rolling update
- **3 отдельные сети**: frontend, backend (internal), monitoring
- **Redis с паролем** и конфигурацией из `redis.conf`
- **Сжатые логи** (50 MB макс)
- **Prometheus + Grafana** доступны через профиль `monitoring`

### 6.3 Production с мониторингом

```bash
docker compose -f docker-compose.production.yml --profile monitoring up -d --build
```

Это добавит:
- Prometheus на порту 9090
- Grafana на порту 3001

---

## 7. Мониторинг (Prometheus + Grafana)

### 7.1 Standalone мониторинг (без production compose)

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

| Сервис | Порт | Логин |
|--------|------|-------|
| Prometheus | 9091 | — |
| Grafana | 3001 | admin / admin |

### 7.2 Что мониторится

Prometheus скрейпит `http://api:8000/metrics` каждые 10 секунд.

Предустановленные Grafana дашборды:
- **HEAN Trading System Overview**: API request rate, response time (p95/p99), memory, CPU
- **HEAN Trading System**: equity, drawdown, PnL breakdown, execution metrics (slippage, maker/taker ratio, latency), safety metrics

### 7.3 Доступ

```
http://localhost:3001          # Grafana
http://localhost:9091          # Prometheus (standalone)
http://localhost:9090          # Prometheus (production compose)
```

---

## 8. Архитектура и потоки данных

### 8.1 Общая схема

```
                        ┌──────────────────────────────────────────────────┐
                        │              Docker Network (hean-network)       │
                        │                                                  │
  Bybit WebSocket ─────►│  collector ──► Redis Streams ──► physics         │
                        │                     │               │            │
                        │                     │               ▼            │
                        │                     │             brain          │
                        │                     │               │            │
                        │                     │               ▼            │
                        │                     │           risk-svc         │
                        │                     │                            │
                        │              ┌──────┴──────┐                     │
  Пользователь:3000 ───►│  UI (nginx) ─►│  API (FastAPI)│◄──► Redis       │
                        │              │   port 8000   │                   │
                        │              └──────┬──────┘                     │
                        │                     │                            │
                        │              symbiont-testnet                    │
                        │                     │                            │
                        │                     ▼                            │
                        │              Bybit Testnet API                   │
                        └──────────────────────────────────────────────────┘
```

### 8.2 Два режима работы

**Монолитный режим (API):**
```
API → Engine Facade → EventBus → Strategies/Risk/Execution → Bybit
```
Весь торговый цикл внутри одного контейнера `api`.

**Микросервисный режим (Collector → Physics → Brain → Risk):**
```
Collector → Redis Stream (market:{symbol})
                ↓
Physics → Redis Stream (physics:{symbol})
                ↓
Brain → Redis Stream (brain:signals)
                ↓
Risk-svc → Redis Stream (risk:approved)
```
Распределённый pipeline через Redis Streams.

### 8.3 Потоки данных Redis

| Redis Stream | Производитель | Потребитель | Данные |
|-------------|---------------|-------------|--------|
| `market:{symbol}` | collector | physics | Тики, orderbook |
| `physics:{symbol}` | physics | brain | Температура, энтропия, фаза |
| `brain:signals` | brain | risk-svc | Торговые сигналы |
| `risk:approved` | risk-svc | symbiont | Одобренные ордера |

---

## 9. Порты и сети

### 9.1 Карта портов (Development)

| Порт | Сервис | Протокол | Описание |
|------|--------|----------|----------|
| **3000** | ui | HTTP | React дашборд |
| **8000** | api | HTTP/WS | REST API + WebSocket |
| **6379** | redis | TCP | Redis (для отладки) |

### 9.2 Карта портов (Production)

| Порт | Сервис | Протокол |
|------|--------|----------|
| **80** | ui | HTTP |
| **443** | ui | HTTPS |
| **9090** | prometheus | HTTP (profile: monitoring) |
| **3001** | grafana | HTTP (profile: monitoring) |

Redis и API **не** выставлены наружу в production.

### 9.3 Сети

**Development:** одна сеть `hean-network` (bridge)

**Production:** три изолированные сети:
```
frontend  (172.20.0.0/24) — ui, api
backend   (172.21.0.0/24, internal) — api, redis
monitoring (172.22.0.0/24) — api, prometheus, grafana
```

### 9.4 Nginx Proxy (UI)

UI проксирует запросы к API:
```
/api/*  →  http://api:8000/*     (REST, таймаут 30с)
/ws     →  http://api:8000/ws    (WebSocket, таймаут 3600с)
/*      →  index.html            (SPA fallback)
```

---

## 10. Проверка здоровья

### 10.1 Healthcheck каждого сервиса

| Сервис | Healthcheck | Интервал | Старт |
|--------|------------|----------|-------|
| redis | `redis-cli ping` | 10s | 10s |
| api | `curl -f http://localhost:8000/health` | 30s | 15s |
| collector | Python `redis.ping()` | 30s | 10s |
| physics | Python `redis.ping()` | 30s | 10s |
| brain | Python `redis.ping()` | 30s | 10s |
| risk-svc | Python `redis.ping()` | 30s | 10s |
| symbiont | Проверка процесса + heartbeat файл (< 300s) | 30s | 60s |

### 10.2 Ручная проверка

```bash
# Статус всех контейнеров с healthcheck
docker compose ps

# Healthcheck конкретного контейнера
docker inspect --format='{{.State.Health.Status}}' hean-api

# API endpoints для диагностики
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/engine/status
curl http://localhost:8000/api/v1/risk/governor/status
curl http://localhost:8000/api/v1/trading/metrics
```

### 10.3 Smoke Test

```bash
# Запуск полного smoke test (API должен быть запущен)
./scripts/smoke_test.sh
```

Smoke test проверяет 10 категорий: REST API, AI catalyst, market data, risk governor, WebSocket, engine control, multi-symbol, Bybit integration, mock data detection, trading funnel.

---

## 11. Управление и обслуживание

### 11.1 Основные команды

```bash
# Запуск
docker compose up -d --build

# Остановка (контейнеры удаляются)
docker compose down

# Остановка с удалением данных Redis
docker compose down -v

# Перезапуск одного сервиса
docker compose restart api

# Логи
docker compose logs -f              # все сервисы
docker compose logs -f api brain    # конкретные сервисы
docker compose logs --tail=100 api  # последние 100 строк
```

### 11.2 Обновление кода

```bash
# Пересобрать и перезапустить всё
docker compose up -d --build

# Пересобрать только изменённый сервис
docker compose up -d --build api

# Принудительная пересборка без кеша
docker compose build --no-cache api
docker compose up -d api
```

### 11.3 Очистка

```bash
# Удалить остановленные контейнеры
docker compose down

# Удалить контейнеры + тома (Redis данные)
docker compose down -v

# Удалить все образы HEAN
docker rmi $(docker images "hean-*" -q) 2>/dev/null

# Полная очистка Docker (осторожно — удаляет ВСЁ!)
docker system prune -af --volumes
```

### 11.4 Доступ внутрь контейнера

```bash
# Зайти в API контейнер
docker exec -it hean-api bash

# Зайти в Redis CLI
docker exec -it hean-redis redis-cli

# Проверить Python внутри API
docker exec -it hean-api python -c "from hean.config import settings; print(settings.bybit_testnet)"
```

### 11.5 Ресурсы

Лимиты ресурсов (настроены в `docker-compose.yml`):

| Сервис | CPU лимит | RAM лимит | CPU резерв | RAM резерв |
|--------|----------|----------|-----------|-----------|
| api | 2 | 2 GB | 0.5 | 512 MB |
| redis | 1 | 512 MB | 0.25 | 256 MB |
| symbiont | 2 | 2 GB | 0.5 | 512 MB |
| collector | 0.5 | 256 MB | 0.1 | 128 MB |
| physics | 1 | 512 MB | 0.25 | 256 MB |
| brain | 1 | 512 MB | 0.25 | 256 MB |
| risk-svc | 0.5 | 256 MB | 0.1 | 128 MB |
| ui | 0.5 | 256 MB | 0.1 | 64 MB |
| **Итого** | **7.5** | **6.5 GB** | | |

---

## 12. Решение проблем

### 12.1 Docker не запускается

```bash
# Проверить что Docker демон работает
docker info

# macOS: открыть Docker Desktop
open -a "Docker Desktop"
# Подождать ~30 секунд до готовности
```

### 12.2 Порт занят

```bash
# Найти кто занимает порт
lsof -i :8000
lsof -i :3000
lsof -i :6379

# Убить процесс (PID из предыдущей команды)
kill -9 <PID>
```

### 12.3 Контейнер не стартует

```bash
# Посмотреть логи упавшего контейнера
docker compose logs api

# Посмотреть почему перезапускается
docker inspect hean-api | grep -A 10 "State"

# Частая причина: отсутствие env-файлов
ls -la .env backend.env .env.symbiont
```

### 12.4 API возвращает ошибки

```bash
# Проверить что Redis доступен из API
docker exec hean-api python -c "import redis; r=redis.from_url('redis://redis:6379'); print(r.ping())"

# Проверить переменные окружения
docker exec hean-api env | grep BYBIT

# Перезапустить API
docker compose restart api
```

### 12.5 UI показывает пустой экран

```bash
# Проверить что nginx работает
docker exec hean-ui nginx -t

# Проверить проксирование к API
docker exec hean-ui curl -s http://api:8000/health

# Пересобрать UI
docker compose up -d --build ui
```

### 12.6 Redis не подключается

```bash
# Проверить здоровье Redis
docker exec hean-redis redis-cli ping
# Ожидаемый ответ: PONG

# Проверить память Redis
docker exec hean-redis redis-cli info memory | grep used_memory_human
```

### 12.7 Symbiont не торгует

```bash
# Проверить логи
docker compose logs -f symbiont-testnet

# Проверить healthcheck
docker inspect --format='{{.State.Health.Status}}' hean-symbiont-testnet

# Проверить API ключи
docker exec hean-symbiont-testnet env | grep BYBIT_API_KEY
# Должен показать реальный ключ, НЕ placeholder
```

### 12.8 Ошибки сборки образов

```bash
# Пересборка без кеша
docker compose build --no-cache

# Проблема с зависимостями Python
docker compose build --no-cache api

# Проблема с npm
docker compose build --no-cache ui

# Проверить .dockerignore — не исключает ли нужные файлы
cat .dockerignore
```

### 12.9 Нехватка места

```bash
# Проверить использование Docker
docker system df

# Очистить неиспользуемые образы и кеш
docker system prune -f

# Агрессивная очистка (удалит все неиспользуемые образы)
docker system prune -af
```

### 12.10 Диагностический скрипт

```bash
# Встроенный скрипт диагностики
./diagnose_docker.sh
```

---

## 13. Kubernetes (продвинутый)

Манифесты в `k8s/`:

```bash
# Создать namespace
kubectl apply -f k8s/namespace.yaml

# Создать секреты (отредактируйте значения!)
kubectl apply -f k8s/secret.yaml

# Применить конфигурацию
kubectl apply -f k8s/configmap.yaml

# Запустить сервисы
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/ui-deployment.yaml

# Проверить статус
kubectl -n hean-production get pods
kubectl -n hean-production get svc
```

Kubernetes настроен с:
- **HPA** — автоскейлинг API от 3 до 10 реплик (по CPU 70%, RAM 80%)
- **Anti-affinity** — поды распределяются по разным нодам
- **Security context** — non-root, read-only filesystem
- **Ingress** — nginx с TLS (cert-manager), rate limiting 100 rps

---

## Шпаргалка

```bash
# Первый запуск
cp .env.example .env && cp backend.env.example backend.env && cp .env.symbiont.example .env.symbiont
# Отредактировать .env, backend.env, .env.symbiont — вписать API ключи
docker compose up -d --build

# Проверить
docker compose ps
curl http://localhost:8000/health
open http://localhost:3000

# Логи
docker compose logs -f

# Остановить
docker compose down

# Полная пересборка
docker compose down && docker compose up -d --build

# Production
docker compose -f docker-compose.production.yml up -d --build

# Production + мониторинг
docker compose -f docker-compose.production.yml --profile monitoring up -d --build
```
