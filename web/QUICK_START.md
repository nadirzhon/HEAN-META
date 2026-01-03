# 🚀 Быстрый запуск веб-сайта через Docker

## Самый простой способ

```bash
# Из корня проекта HEAN
docker-compose up -d web
```

Откройте браузер: **http://localhost:3000**

## Альтернативные способы

### Способ 1: Из папки web/

```bash
cd web
docker-compose up -d
```

### Способ 2: Используя скрипт

```bash
cd web
./start.sh
```

### Способ 3: Ручная сборка

```bash
cd web
docker build -t hean-website .
docker run -d -p 3000:80 --name hean-website hean-website
```

## Проверка

```bash
# Статус
docker ps | grep hean-website

# Логи
docker logs hean-website

# Health check
curl http://localhost:3000/health
```

## Остановка

```bash
# Остановить
docker stop hean-website

# Или через docker-compose
docker-compose stop web
```

## Удаление

```bash
# Удалить контейнер
docker rm -f hean-website

# Удалить образ
docker rmi hean-website
```

## Проблемы?

1. **Порт занят?** Измените порт в `docker-compose.yml` (строка с `3000:80`)
2. **Не запускается?** Проверьте логи: `docker logs hean-website`
3. **Нужно пересобрать?** `docker-compose build --no-cache web && docker-compose up -d web`

