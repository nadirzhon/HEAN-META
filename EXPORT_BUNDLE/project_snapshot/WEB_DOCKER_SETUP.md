# Запуск веб-сайта через Docker

## Быстрый старт

### Из корня проекта

```bash
# Запустить только веб-сайт
docker-compose up -d web

# Или запустить все сервисы (HEAN + веб-сайт)
docker-compose up -d
```

Сайт будет доступен на **http://localhost:3000**

### Из папки web/

```bash
cd web

# Используя docker-compose
docker-compose up -d

# Или используя скрипт
./start.sh

# Или вручную
docker build -t hean-website .
docker run -d -p 3000:80 --name hean-website hean-website
```

## Проверка работы

```bash
# Проверить статус контейнера
docker ps | grep hean-website

# Просмотреть логи
docker logs -f hean-website

# Проверить health check
curl http://localhost:3000/health
```

## Управление

```bash
# Остановить
docker stop hean-website

# Запустить
docker start hean-website

# Перезапустить
docker restart hean-website

# Удалить контейнер
docker rm -f hean-website

# Удалить образ
docker rmi hean-website
```

## Структура Docker файлов

```
web/
├── Dockerfile          # Docker образ на базе nginx:alpine
├── nginx.conf          # Конфигурация nginx
├── docker-compose.yml  # Docker Compose для веб-сайта
├── start.sh            # Скрипт быстрого запуска
└── .dockerignore       # Исключения для сборки
```

## Порт

По умолчанию веб-сайт доступен на порту **3000**. 

Чтобы изменить порт, отредактируйте `docker-compose.yml`:

```yaml
ports:
  - "8080:80"  # Измените 8080 на нужный порт
```

## Особенности

- ✅ Легковесный образ на базе nginx:alpine
- ✅ Gzip сжатие для лучшей производительности
- ✅ Кэширование статических файлов
- ✅ Security headers
- ✅ Health check endpoint
- ✅ Автоматический перезапуск при сбоях

## Troubleshooting

### Порт уже занят

Если порт 3000 занят, измените его в `docker-compose.yml`:

```yaml
ports:
  - "8080:80"  # Используйте другой порт
```

### Контейнер не запускается

```bash
# Проверьте логи
docker logs hean-website

# Проверьте, что порт свободен
lsof -i :3000

# Пересоберите образ
docker-compose build --no-cache web
docker-compose up -d web
```

### Изменения не отображаются

После изменения файлов сайта нужно пересобрать образ:

```bash
docker-compose build web
docker-compose up -d web
```

Или для разработки можно использовать volume mount (добавить в docker-compose.yml):

```yaml
volumes:
  - ./web:/usr/share/nginx/html:ro
```

