# Запуск нового веб-интерфейса (apps/ui)

## Быстрый старт

```bash
# Собрать и запустить API + UI + Redis
docker-compose up -d --build

# Или только UI (если API уже запущен)
docker-compose up -d --build ui
```

Интерфейс доступен на **http://localhost:3000**.

## Режим разработки (hot reload)

```bash
# Поднять API + UI-dev (порт 5173) с профилем dev
docker-compose --profile dev up -d --build api ui-dev
```

UI-dev использует `Dockerfile.dev` и монтирует исходники из `apps/ui` для живой перезагрузки.

## Ручная сборка без Compose

```bash
cd apps/ui

# Сборка статического бандла с нужными API/WS адресами
docker build \
  --build-arg VITE_API_BASE=http://localhost:8000 \
  --build-arg VITE_WS_URL=ws://localhost:8000/ws \
  -t hean-ui .

# Запуск
docker run -d -p 3000:80 --name hean-ui hean-ui
```

## Проверка и управление

```bash
docker ps | grep hean-ui        # статус
docker logs -f hean-ui          # логи
docker stop hean-ui             # остановить
docker start hean-ui            # запустить
docker rm -f hean-ui            # удалить контейнер
docker rmi hean-ui              # удалить образ
```

## Порт

Меняем внешний порт через `docker-compose.yml`:

```yaml
ports:
  - "8080:80"   # 8080 снаружи -> 80 внутри контейнера nginx
```

## Если изменения не видны

1. Очистите кэш браузера (Cmd/Ctrl + Shift + R).
2. Пересоберите образ без кэша:
   ```bash
   docker-compose build --no-cache ui
   docker-compose up -d ui
   ```
