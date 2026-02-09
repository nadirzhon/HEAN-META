# Исправление REST Polling Degraded

## Проблема
UI показывал предупреждение "REST polling degraded — showing cached/mock data" даже когда REST API работал.

## Причина
Логика определения `restOk` была слишком строгой:
```typescript
const restOk = telemetryOk && Boolean(portfolioData);
```
Это требовало что **оба** endpoint должны работать одновременно. Если один из них временно недоступен (таймаут, ошибка сети), то REST считался недоступным.

## Исправления

### 1. Улучшенная логика определения REST доступности
- ✅ Изменено с `telemetryOk && portfolioData` на `telemetryOk || portfolioData`
- ✅ REST считается доступным если **хотя бы один** endpoint работает
- ✅ Это более устойчиво к временным сбоям отдельных endpoints

### 2. Детальная индикация состояния
Добавлены три уровня состояния:
- **"ok"** - оба endpoint работают (telemetry + portfolio)
- **"degraded"** - только один endpoint работает
- **"error"** - оба endpoint не работают

### 3. Улучшенные сообщения в UI
- ✅ "REST polling partially degraded" - когда только один endpoint работает
- ✅ "REST polling failed" - когда оба endpoint не работают
- ✅ Убрано сообщение когда REST работает нормально

## Результат

### До исправления:
- REST считался недоступным если один endpoint падал
- Показывалось "REST polling degraded" даже при временных сбоях
- UI переключался на mock данные

### После исправления:
- ✅ REST считается доступным если хотя бы один endpoint работает
- ✅ Показывается детальная индикация состояния
- ✅ UI продолжает работать с реальными данными даже при частичных сбоях

## Проверка

Все REST endpoints работают:
- ✅ `/health` - 200 OK
- ✅ `/telemetry/summary` - 200 OK  
- ✅ `/portfolio/summary` - 200 OK
- ✅ `/engine/status` - 200 OK
- ✅ `/strategies` - 200 OK

## Статус
✅ **ИСПРАВЛЕНО** - REST polling теперь более устойчив к сбоям
