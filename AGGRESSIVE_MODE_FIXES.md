# Исправления для Агрессивного Режима - Полный Отчет

**Дата**: 2026-01-XX  
**Цель**: Устранить все блокировки сигналов в DEBUG_MODE для максимальной активности торговли

---

## 🔍 Найденные проблемы

### 1. **Порог движения цены слишком высокий**
- **Файл**: `src/hean/strategies/impulse_engine.py:49`
- **Было**: `_impulse_threshold = 0.005` (0.5%)
- **Проблема**: Синтетический feed может не генерировать такое движение
- **✅ Исправлено**: `0.001` (0.1%) в DEBUG_MODE

### 2. **Требование volume spike**
- **Файл**: `src/hean/strategies/impulse_engine.py:302`
- **Было**: Требовался объем на 20% выше среднего
- **Проблема**: В синтетическом feed это редко происходит
- **✅ Исправлено**: Всегда `True` в DEBUG_MODE

### 3. **Spread check блокирует сигналы**
- **Файл**: `src/hean/strategies/impulse_engine.py:162-180`
- **Было**: Проверка spread всегда активна
- **Проблема**: Даже в DEBUG_MODE блокирует сигналы
- **✅ Исправлено**: Полностью обойдено в DEBUG_MODE

### 4. **Regime gating**
- **Файл**: `src/hean/strategies/impulse_engine.py:277-283`
- **Было**: Только логирование, но проверка режима все равно могла блокировать
- **Проблема**: Стратегия могла работать только в IMPULSE режиме
- **✅ Исправлено**: 
  - В DEBUG_MODE разрешены все режимы (IMPULSE, NORMAL, RANGE)
  - Проверка режима полностью обойдена

### 5. **Maker edge check**
- **Файл**: `src/hean/strategies/impulse_engine.py:495-516`
- **Было**: Проверка maker edge в IMPULSE режиме
- **Проблема**: Блокировал сигналы даже в DEBUG_MODE
- **✅ Исправлено**: Полностью обойдено в DEBUG_MODE

### 6. **Фильтры все еще проверяются**
- **Файл**: `src/hean/strategies/impulse_engine.py:454-465`
- **Было**: `if not settings.debug_mode:` - фильтры обходились, но проверка была
- **Проблема**: Если фильтры возвращали False, сигнал блокировался
- **✅ Исправлено**: Принудительно устанавливается `filter_result = True` в DEBUG_MODE

### 7. **Edge estimator блокирует сигналы**
- **Файл**: `src/hean/strategies/impulse_engine.py:542-547`
- **Было**: Проверка edge estimator активна
- **Проблема**: Мог блокировать сигналы с низким edge
- **✅ Исправлено**: Обойдено в DEBUG_MODE

### 8. **Edge confirmation loop (2-step entry)**
- **Файл**: `src/hean/strategies/impulse_engine.py:552-585`
- **Было**: Требовал подтверждения на втором импульсе
- **Проблема**: Удваивал время до генерации сигнала
- **✅ Исправлено**: Полностью обойдено в DEBUG_MODE - сигналы публикуются сразу

### 9. **Интервал forced signal слишком большой**
- **Файл**: `src/hean/strategies/impulse_engine.py:80`
- **Было**: 300,000 тиков (~2 дня)
- **Проблема**: Очень редко генерировал принудительные сигналы
- **✅ Исправлено**: 1,000 тиков в DEBUG_MODE (очень часто для тестирования)

---

## ✅ Примененные исправления

### 1. Порог движения цены
```python
# БЫЛО:
self._impulse_threshold = 0.005  # 0.5%

# СТАЛО:
self._impulse_threshold = 0.001 if settings.debug_mode else 0.005  # 0.1% в debug
```

### 2. Volume spike requirement
```python
# БЫЛО:
volume_spike = recent_volume > avg_volume * 1.2

# СТАЛО:
if settings.debug_mode:
    volume_spike = True  # Always pass
else:
    volume_spike = recent_volume > avg_volume * 1.2
```

### 3. No-trade zone (spread check)
```python
# БЫЛО:
async def _check_no_trade_zone(self, tick: Tick) -> bool:
    if tick.bid and tick.ask:
        spread = (tick.ask - tick.bid) / tick.price
        if spread > self._spread_gate:
            return True  # Блокировать
    return False

# СТАЛО:
async def _check_no_trade_zone(self, tick: Tick) -> bool:
    if settings.debug_mode:
        return False  # Полностью обойти
    # ... остальная проверка
```

### 4. Regime gating
```python
# БЫЛО:
if settings.impulse_allow_normal:
    self._allowed_regimes = {Regime.IMPULSE, Regime.NORMAL}
else:
    self._allowed_regimes = {Regime.IMPULSE}

# СТАЛО:
if settings.debug_mode:
    self._allowed_regimes = {Regime.IMPULSE, Regime.NORMAL, Regime.RANGE}  # Все режимы
elif settings.impulse_allow_normal:
    self._allowed_regimes = {Regime.IMPULSE, Regime.NORMAL}
else:
    self._allowed_regimes = {Regime.IMPULSE}
```

### 5. Maker edge check
```python
# БЫЛО:
if current_regime == Regime.IMPULSE:
    if maker_edge_bps < reduced_threshold:
        return  # Блокировать

# СТАЛО:
if settings.debug_mode:
    logger.debug(f"[AGGRESSIVE] Maker edge check bypassed")
elif current_regime == Regime.IMPULSE:
    # ... проверка
```

### 6. Filter pipeline
```python
# БЫЛО:
if not settings.debug_mode:
    filter_result = self._filter_pipeline.allow(tick, context)
    if not filter_result:
        return  # Блокировать
else:
    logger.debug(f"[DEBUG] Filters bypassed")

# СТАЛО:
if settings.debug_mode:
    filter_result = True  # Принудительно разрешить
    logger.debug(f"[AGGRESSIVE] All filters completely bypassed")
else:
    filter_result = self._filter_pipeline.allow(tick, context)
    if not filter_result:
        return
```

### 7. Edge estimator
```python
# БЫЛО:
if not settings.debug_mode:
    edge_allowed = self._edge_estimator.should_emit_signal(...)
    if not edge_allowed:
        return  # Блокировать

# СТАЛО:
if settings.debug_mode:
    edge_allowed = True  # Принудительно разрешить
    logger.debug(f"[AGGRESSIVE] Edge estimator check bypassed")
else:
    edge_allowed = self._edge_estimator.should_emit_signal(...)
    if not edge_allowed:
        return
```

### 8. Edge confirmation loop
```python
# БЫЛО:
if not settings.debug_mode:
    confirmed_signal = self._edge_confirmation.confirm_or_update(...)
    if confirmed_signal is None:
        return  # Ждать подтверждения
    await self._publish_signal(confirmed_signal)
else:
    await self._publish_signal(signal)  # Но все равно был вызов после

# СТАЛО:
if settings.debug_mode:
    logger.debug(f"[AGGRESSIVE] Edge confirmation bypassed - emitting immediately")
    await self._publish_signal(signal)  # Только один раз, сразу
    self._last_trade_time[symbol] = datetime.utcnow()
else:
    # ... оригинальная логика с подтверждением
```

### 9. Forced signal interval
```python
# БЫЛО:
self._force_signal_interval = 300000  # 300k тиков

# СТАЛО:
if settings.debug_mode:
    self._force_signal_interval = 1000  # 1k тиков (очень часто)
else:
    self._force_signal_interval = 300000
```

---

## 📊 Итоговые изменения

| Компонент | Было | Стало (DEBUG_MODE=True) |
|-----------|------|-------------------------|
| Порог движения цены | 0.5% | 0.1% (-80%) |
| Volume spike | Требуется 20% | Всегда True |
| Spread check | Активен | Полностью обойден |
| Regime gating | IMPULSE only | Все режимы разрешены |
| Maker edge check | Активен | Полностью обойден |
| Filter pipeline | Проверяется | Принудительно True |
| Edge estimator | Проверяется | Принудительно True |
| Edge confirmation | 2-step required | Немедленная публикация |
| Forced signal interval | 300k тиков | 1k тиков (-99.7%) |

---

## 🎯 Ожидаемый результат

После этих исправлений, сигналы должны генерироваться **МНОГО ЧАЩЕ**, потому что:

1. ✅ Порог движения снижен в 5 раз (0.5% → 0.1%)
2. ✅ Volume spike не требуется
3. ✅ Spread не блокирует
4. ✅ Режим рынка не важен
5. ✅ Maker edge не проверяется
6. ✅ Все фильтры обойдены
7. ✅ Edge estimator обойден
8. ✅ Подтверждение не требуется (немедленная публикация)
9. ✅ Принудительные сигналы каждые 1k тиков (вместо 300k)

---

## 🧪 Проверка

После применения исправлений проверьте логи:

```bash
docker-compose logs -f afo-engine | grep -E "AGGRESSIVE|Impulse detected|FORCED_PUBLISH"
```

Ожидаемые сообщения:
- `[AGGRESSIVE] No-trade zone completely bypassed`
- `[AGGRESSIVE] Regime gating completely bypassed`
- `[AGGRESSIVE] Volume spike requirement bypassed`
- `[AGGRESSIVE] All filters completely bypassed`
- `[AGGRESSIVE] Edge estimator check bypassed`
- `[AGGRESSIVE] Maker edge check bypassed`
- `[AGGRESSIVE] Edge confirmation bypassed - emitting immediately`
- `[FORCED] Impulse detected`
- `[FORCED_PUBLISH] Publishing signal`

---

## ⚠️ Важные замечания

1. **ТОЛЬКО ДЛЯ ТЕСТИРОВАНИЯ**: Эти изменения работают только когда `DEBUG_MODE=True`
2. **БЕЗОПАСНОСТЬ**: В production режиме все проверки остаются активными
3. **МОНИТОРИНГ**: Следите за количеством сигналов и блокировок в логах
4. **ОТКАТ**: После тестирования верните `DEBUG_MODE=False`

---

## 📝 Следующие шаги

1. ✅ Исправления применены
2. ⏳ Перезапустите движок: `docker-compose restart afo-engine`
3. ⏳ Проверьте логи на наличие `[AGGRESSIVE]` сообщений
4. ⏳ Убедитесь, что сигналы генерируются чаще
5. ⏳ Мониторьте количество открытых позиций и ордеров

---

**Все исправления применены и готовы к тестированию!**
