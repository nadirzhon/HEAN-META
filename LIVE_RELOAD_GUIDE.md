# 🔥 Live Reload Development Guide

## Быстрый старт

### Запуск в режиме разработки с hot-reload:

```bash
./start-dev.sh
```

Или вручную:

```bash
docker compose --profile dev up --build
```

## 📊 Что включено?

### 🔹 Backend API (Python/FastAPI)
- **Hot-reload через uvicorn**: Автоматическая перезагрузка при изменении `.py` файлов
- **Volume mounting**: Код в `./src` монтируется в контейнер
- **Порт**: `8000`
- **URL**: http://localhost:8000
- **Docs**: http://localhost:8000/docs

### 🔹 Frontend UI (React/Vite)
- **Hot Module Replacement (HMR)**: Мгновенное обновление без перезагрузки страницы
- **Volume mounting**: Код в `./apps/ui/src` монтируется в контейнер
- **Порт**: `5173`
- **URL**: http://localhost:5173

## 🚀 Режимы работы

### Production режим (по умолчанию):
```bash
docker compose up
```
- Оптимизированные образы
- Nginx для UI
- Без hot-reload

### Development режим:
```bash
docker compose --profile dev up
```
- Образы с dev инструментами
- Hot-reload для API и UI
- Volume mounting для мгновенных изменений

## 📝 Как это работает?

### Backend API
1. **Dockerfile.dev** - образ с watchdog и dev зависимостями
2. **uvicorn --reload** - отслеживает изменения в `/app/src`
3. **Volume mounting** - `./src:/app/src` проксирует ваши локальные изменения в контейнер

При изменении любого `.py` файла в `./src`, uvicorn автоматически перезапустит приложение (~1-2 секунды).

### Frontend UI
1. **Dockerfile.dev** - Node.js образ с Vite dev server
2. **Vite HMR** - Hot Module Replacement от Vite
3. **Volume mounting** - монтируются:
   - `./apps/ui/src`
   - `./apps/ui/index.html`
   - `./apps/ui/vite.config.ts`
   - `./apps/ui/postcss.config.mjs`

При изменении любого файла в `./apps/ui/src`, Vite мгновенно обновляет браузер без перезагрузки.

## 🔧 Troubleshooting

### API не перезапускается при изменениях?

**Проверьте логи:**
```bash
docker compose logs -f api-dev
```

**Убедитесь что volume mounted:**
```bash
docker compose exec api-dev ls -la /app/src
```

### UI не обновляется?

**Проверьте Vite dev server:**
```bash
docker compose logs -f ui-dev
```

**Проверьте что порт 5173 открыт:**
```bash
curl http://localhost:5173
```

### Медленная перезагрузка?

**Для API:** uvicorn --reload отслеживает все файлы. Можно ограничить:
```yaml
command: uvicorn hean.api.main:app --reload --reload-dir /app/src/hean
```

**Для UI:** Vite оптимизирован из коробки, но можно улучшить:
- Исключите `node_modules` из watching
- Добавьте `.gitignore` паттерны в Vite config

## 📁 Структура

```
HEAN-META/
├── api/
│   ├── Dockerfile          # Production образ
│   └── Dockerfile.dev      # Development образ с hot-reload
├── apps/ui/
│   ├── Dockerfile          # Production (nginx)
│   └── Dockerfile.dev      # Development (Vite dev server)
├── docker-compose.yml      # Конфигурация с профилями
├── start-dev.sh           # Скрипт для быстрого старта dev режима
└── src/                   # Backend код (монтируется в контейнер)
```

## ⚙️ Дополнительные команды

### Пересобрать только API:
```bash
docker compose --profile dev up --build api-dev
```

### Пересобрать только UI:
```bash
docker compose --profile dev up --build ui-dev
```

### Остановить dev окружение:
```bash
docker compose --profile dev down
```

### Посмотреть логи в реальном времени:
```bash
# Все сервисы
docker compose --profile dev logs -f

# Только API
docker compose --profile dev logs -f api-dev

# Только UI
docker compose --profile dev logs -f ui-dev
```

## 🎯 Best Practices

1. **Используйте dev режим для разработки**: Не нужно пересобирать образы после каждого изменения
2. **Production режим для тестирования**: Проверяйте изменения в production режиме перед деплоем
3. **Коммитьте регулярно**: Hot-reload не заменяет git
4. **Следите за логами**: Они покажут ошибки сразу при сохранении файла

## 🔗 Полезные ссылки

- [Uvicorn Auto-reload](https://www.uvicorn.org/#command-line-options)
- [Vite HMR API](https://vitejs.dev/guide/api-hmr.html)
- [Docker Compose Profiles](https://docs.docker.com/compose/profiles/)

---

**Happy coding! 🚀**
